#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "parallel.h"

#define THETA_STEPS 180
#define SHIFT 11
#define SCALE (1 << SHIFT)


// Parametri di Cache Blocking (Tuning per CPU moderne / Apple Silicon)
#define TILE_P 256  // Quanti pixel processare assieme
#define TILE_T 32   // Quanti angoli processare assieme


// ––– Pure MPI –––
// The image is divided into (height / size) rows and scattered across each node
Lines* HoughLines_Parallel_MPI(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int theta = THETA_STEPS;
    int rho = compute_rho(width, height);

    // Calculate workload offsets
    int base = (size > 0) ? (height / size) : 0;
    int rem  = (size > 0) ? (height % size) : 0;
    int local_rows = base + ((rank < rem) ? 1 : 0);
    int y_start    = rank * base + ((rank < rem) ? rank : rem);

    int *sc_counts = NULL;
    int *sc_displs = NULL;
    
    // Rank 0 handles distribution setup
    if (rank == 0) {
        sc_counts = malloc(size * sizeof(int));
        sc_displs = malloc(size * sizeof(int));
        if (!sc_counts || !sc_displs) {
            fprintf(stderr, "Rank 0: Failed to allocate scatter arrays.\n");
            MPI_Abort(comm, 1);
        }
        
        int disp = 0;
        for (int r = 0; r < size; r++) {
            sc_counts[r] = (base + ((r < rem) ? 1 : 0)) * width;
            sc_displs[r] = disp;
            disp += sc_counts[r];
        }
    }

    int local_elems = local_rows * width;
    unsigned char *local_edge = malloc((local_elems > 0 ? local_elems : 1) * sizeof(unsigned char));
    if (!local_edge) MPI_Abort(comm, 1);

    // Scatter the flattened 1D array
    MPI_Scatterv(edge_img, sc_counts, sc_displs, MPI_UNSIGNED_CHAR,
                 local_edge, local_elems, MPI_UNSIGNED_CHAR, 0, comm);

    if (rank == 0) { free(sc_counts); free(sc_displs); }

    // Precompute trigonometric tables locally
    float cos_table[THETA_STEPS], sin_table[THETA_STEPS];
    for (int t = 0; t < theta; t++) {
        float rad = (float)t * 3.1415926535f / (float)THETA_STEPS;
        cos_table[t] = cosf(rad);
        sin_table[t] = sinf(rad);
    }

    // Allocate local accumulator
    size_t acc_len = (size_t)rho * (size_t)theta;
    int *local_acc = calloc(acc_len, sizeof(int));
    if (!local_acc) MPI_Abort(comm, 1);

    // OPTIMIZATION: Single-pass Hough accumulator generation
    for (int ly = 0; ly < local_rows; ly++) {
        int global_y = y_start + ly;
        for (int x = 0; x < width; x++) {
            if (local_edge[ly * width + x] > 0) {
                float fx = (float)x;
                float fy = (float)global_y;
                for (int t = 0; t < theta; t++) {
                    int r = (int)roundf(fx * cos_table[t] + fy * sin_table[t]) + rho / 2;
                    if (r >= 0 && r < rho) {
                        local_acc[r * theta + t]++;
                    }
                }
            }
        }
    }
    free(local_edge); 

    int *global_acc = calloc(acc_len, sizeof(int));
    if (!global_acc) MPI_Abort(comm, 1); 
    
    // Use Allreduce so everyone gets the fully summed Hough Space
    MPI_Allreduce(local_acc, global_acc, (int)acc_len, MPI_INT, MPI_SUM, comm);
    free(local_acc);

    // Setup 2D pointer array for NMS
    int **acc2d = malloc(rho * sizeof(int*));
    for (int r = 0; r < rho; r++) acc2d[r] = &global_acc[r * theta];

    // 1. Calculate NMS workload distribution for THIS rank
    int nms_base = rho / size;
    int nms_rem  = rho % size;
    int my_rho_count = nms_base + ((rank < nms_rem) ? 1 : 0);
    int my_rho_start = rank * nms_base + ((rank < nms_rem) ? rank : nms_rem);
    int my_rho_end   = my_rho_start + my_rho_count;

    // 2. Perform NMS only on this rank's assigned rows
    // (Because we used Allreduce, we can safely check neighbors even at the boundaries 
    // my_rho_start and my_rho_end, as the data exists in memory!)
    
    int capacity = 1000;
    Line *local_lines = malloc(capacity * sizeof(Line));
    int local_count = 0;

    for (int r = my_rho_start; r < my_rho_end; r++) {
        for (int t = 0; t < theta; t++) {
            if (acc2d[r][t] > threshold && NMS_max(acc2d, r, t, rho, theta)) {
                if (local_count >= capacity) {
                    capacity *= 2;
                    local_lines = realloc(local_lines, capacity * sizeof(Line));
                }
                local_lines[local_count].r = r - rho / 2;
                local_lines[local_count].t = t;
                local_count++;
            }
        }
    }

    // 3. Gather all the local lines back to Rank 0
    // Because each process found a different number of lines, we need MPI_Gatherv
    
    int *ga_counts = NULL;
    int *ga_displs = NULL;
    if (rank == 0) {
        ga_counts = malloc(size * sizeof(int));
        ga_displs = malloc(size * sizeof(int));
    }

    // First, tell Rank 0 how many lines each process found
    MPI_Gather(&local_count, 1, MPI_INT, ga_counts, 1, MPI_INT, 0, comm);

    Lines *final_result = NULL;

    if (rank == 0) {
        final_result = malloc(sizeof(Lines));
        final_result->count = 0;
        for (int i = 0; i < size; i++) {
            ga_displs[i] = final_result->count;
            final_result->count += ga_counts[i];
        }
        final_result->lines = malloc(final_result->count * sizeof(Line));
    }

    // Create an MPI Datatype for your 'Line' struct (or just send it as raw bytes)
    // Assuming Line is exactly 2 ints (r and t):
    int bytes_per_line = sizeof(Line); 
    
    // Gather the actual line data onto Rank 0
    // Note: MPI_Gather/v usually expects element counts, so we multiply by sizeof(Line) 
    // and send as MPI_BYTE to avoid needing to create a custom MPI_Datatype for now.
    
    // Convert counts and displs to bytes for Rank 0
    if (rank == 0) {
        for(int i=0; i<size; i++) {
            ga_counts[i] *= bytes_per_line;
            ga_displs[i] *= bytes_per_line;
        }
    }

    MPI_Gatherv(local_lines, local_count * bytes_per_line, MPI_BYTE,
                (rank == 0) ? final_result->lines : NULL, ga_counts, ga_displs, MPI_BYTE, 
                0, comm);

    // Cleanup
    free(local_lines);
    free(acc2d);
    free(global_acc);
    if (rank == 0) {
        free(ga_counts);
        free(ga_displs);
    }

    return final_result; // Only valid on Rank 0, NULL elsewhere if handled properly
}

// ––– Dense Hybrid (MPI + OpenMP) –––
Lines* HoughLines_Hybrid_Dense(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int theta = THETA_STEPS;
    const int rho   = compute_rho(width, height);

    const int base = (size > 0) ? ((int)height / size) : 0;
    const int rem  = (size > 0) ? ((int)height % size) : 0;
    const int local_rows = base + ((rank < rem) ? 1 : 0);
    const int y_start    = rank * base + ((rank < rem) ? rank : rem);

    int *counts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        counts = (int*)malloc((size_t)size * sizeof(int));
        displs = (int*)malloc((size_t)size * sizeof(int));
        int disp = 0;
        for (int r = 0; r < size; r++) {
            counts[r] = (base + ((r < rem) ? 1 : 0)) * width;
            displs[r] = disp;
            disp += counts[r];
        }
    }

    unsigned char *local_edge = NULL;
    const int local_elems = local_rows * (int)width;
    const size_t local_bytes = (local_elems > 0) ? (size_t)local_elems : 1u;
    local_edge = (unsigned char*)malloc(local_bytes * sizeof(unsigned char));

    MPI_Scatterv(edge_img, counts, displs, MPI_UNSIGNED_CHAR, local_edge, local_elems, MPI_UNSIGNED_CHAR, 0, comm);

    if (rank == 0) { free(counts); free(displs); }

    double cos_table[THETA_STEPS], sin_table[THETA_STEPS];
    const double pi = 3.14159265358979323846;
    for (int t = 0; t < theta; t++) {
        const double rad = (double)t * pi / (float)THETA_STEPS;
        cos_table[t] = cos(rad);
        sin_table[t] = sin(rad);
    }

    const size_t acc_len = (size_t)rho * (size_t)theta;
    int *local_acc = (int*)malloc(acc_len * sizeof(int));

    int **thread_accs = NULL;
    int nthreads = 0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            nthreads = omp_get_num_threads();
            thread_accs = (int**)malloc((size_t)nthreads * sizeof(int*));
        }

        const int tid = omp_get_thread_num();
        thread_accs[tid] = (int*)calloc(acc_len, sizeof(int));
        int *myacc = thread_accs[tid];

        #pragma omp for schedule(static)
        for (int ly = 0; ly < local_rows; ly++) {
            const int y = y_start + ly;
            const unsigned char *row = local_edge + (size_t)ly * (size_t)width;
            for (int x = 0; x < (int)width; x++) {
                if (row[x] > 0) {
                    for (int t = 0; t < theta; t++) {
                        const int r = (int)lrint((double)x * cos_table[t] + (double)y * sin_table[t]) + rho / 2;
                        if ((unsigned)r < (unsigned)rho) {
                            myacc[(size_t)r * (size_t)theta + (size_t)t]++;
                        }
                    }
                }
            }
        }

        #pragma omp barrier

        #pragma omp for schedule(static)
        for (size_t i = 0; i < acc_len; i++) {
            int sum = 0;
            for (int k = 0; k < nthreads; k++) sum += thread_accs[k][i];
            local_acc[i] = sum;
        }

        free(myacc);
    }
    
    free(thread_accs); free(local_edge);

    int *global_acc = NULL;
    if (rank == 0) global_acc = (int*)calloc(acc_len, sizeof(int));

    MPI_Reduce(local_acc, global_acc, (int)acc_len, MPI_INT, MPI_SUM, 0, comm);
    free(local_acc);

    if (rank != 0) return NULL;

    int **acc2d = (int**)malloc((size_t)rho * sizeof(int*));
    for (int r = 0; r < rho; r++) acc2d[r] = &global_acc[(size_t)r * (size_t)theta];

    Lines *result = (Lines*)malloc(sizeof(Lines));
    result->count = 0;
    int capacity = 64;
    result->lines = (Line*)malloc((size_t)capacity * sizeof(Line));

    for (int r = 0; r < rho; r++) {
        for (int t = 0; t < theta; t++) {
            if (acc2d[r][t] > (int)threshold && NMS_max(acc2d, r, t, rho, theta)) {
                if (result->count >= capacity) {
                    capacity *= 2;
                    result->lines = (Line*)realloc(result->lines, (size_t)capacity * sizeof(Line));
                }
                result->lines[result->count].r = r - rho / 2;
                result->lines[result->count].t = t;
                result->count++;
            }
        }
    }

    free(acc2d); free(global_acc);
    return result;
}

// ––– Sparse Hybrid (MPI + OpenMP) –––
Lines* HoughLines_Hybrid_Sparse(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int theta = THETA_STEPS;
    int rho = compute_rho(width, height);

    int base = (size > 0) ? (height / size) : 0;
    int rem  = (size > 0) ? (height % size) : 0;
    int local_rows = base + ((rank < rem) ? 1 : 0);
    int y_start    = rank * base + ((rank < rem) ? rank : rem);

    int *counts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int disp = 0;
        for (int r = 0; r < size; r++) {
            counts[r] = (base + ((r < rem) ? 1 : 0)) * width;
            displs[r] = disp;
            disp += counts[r];
        }
    }

    int local_elems = local_rows * width;
    unsigned char *local_edge = malloc((local_elems > 0 ? local_elems : 1) * sizeof(unsigned char));

    MPI_Scatterv(edge_img, counts, displs, MPI_UNSIGNED_CHAR, local_edge, local_elems, MPI_UNSIGNED_CHAR, 0, comm);

    if (rank == 0) { free(counts); free(displs); }

    float cos_table[THETA_STEPS], sin_table[THETA_STEPS];
    for (int t = 0; t < theta; t++) {
        float rad = (float)t * 3.1415926535f / (float)THETA_STEPS;
        cos_table[t] = cosf(rad);
        sin_table[t] = sinf(rad);
    }

    int num_edges = 0;
    for (int i = 0; i < local_elems; i++) {
        if (local_edge[i] > 0) num_edges++;
    }

    int* x_coords = malloc((num_edges > 0 ? num_edges : 1) * sizeof(int));
    int* y_coords = malloc((num_edges > 0 ? num_edges : 1) * sizeof(int));

    int edge_idx = 0;
    for (int ly = 0; ly < local_rows; ly++) {
        int global_y = y_start + ly;
        for (int x = 0; x < width; x++) {
            if (local_edge[ly * width + x] > 0) {
                x_coords[edge_idx] = x;
                y_coords[edge_idx] = global_y;
                edge_idx++;
            }
        }
    }
    free(local_edge); 

    size_t acc_len = (size_t)rho * (size_t)theta;
    int *local_acc = calloc(acc_len, sizeof(int));
    int **thread_accs = NULL;
    int nthreads = 1;

    #pragma omp parallel
    {
        #pragma omp single
        {
            nthreads = omp_get_num_threads();
            thread_accs = malloc(nthreads * sizeof(int*));
        }

        int tid = omp_get_thread_num();
        thread_accs[tid] = calloc(acc_len, sizeof(int));
        int *myacc = thread_accs[tid];

        #pragma omp for schedule(static)
        for (int i = 0; i < num_edges; i++) {
            float x = (float)x_coords[i];
            float y = (float)y_coords[i];
            for (int t = 0; t < theta; t++) {
                int r = (int)roundf(x * cos_table[t] + y * sin_table[t]) + rho / 2;
                if (r >= 0 && r < rho) {
                    myacc[r * theta + t]++;
                }
            }
        }

        #pragma omp barrier

        #pragma omp for schedule(static)
        for (size_t i = 0; i < acc_len; i++) {
            int sum = 0;
            for (int k = 0; k < nthreads; k++) sum += thread_accs[k][i];
            local_acc[i] = sum;
        }

        free(myacc);
    }
    
    free(thread_accs); free(x_coords); free(y_coords);

    int *global_acc = NULL;
    if (rank == 0) global_acc = calloc(acc_len, sizeof(int));

    MPI_Reduce(local_acc, global_acc, (int)acc_len, MPI_INT, MPI_SUM, 0, comm);
    free(local_acc);

    if (rank != 0) return NULL;

    int **acc2d = malloc(rho * sizeof(int*));
    for (int r = 0; r < rho; r++) acc2d[r] = &global_acc[r * theta];

    Lines *result = malloc(sizeof(Lines));
    result->count = 0;
    int capacity = 1000;
    result->lines = malloc(capacity * sizeof(Line));

    for (int r = 0; r < rho; r++) {
        for (int t = 0; t < theta; t++) {
            if (acc2d[r][t] > threshold && NMS_max(acc2d, r, t, rho, theta)) {
                if (result->count >= capacity) {
                    capacity *= 2;
                    result->lines = realloc(result->lines, capacity * sizeof(Line));
                }
                result->lines[result->count].r = r - rho / 2;
                result->lines[result->count].t = t;
                result->count++;
            }
        }
    }

    free(acc2d); free(global_acc);
    return result;
}

Lines* HoughLines_Hybrid_Optimized(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int theta_dim = THETA_STEPS;
    int rho_dim = compute_rho(width, height);

    // ---------------------------------------------------------
    // 1. SCATTER MPI (Distribuzione delle righe dell'immagine)
    // ---------------------------------------------------------
    int base = (size > 0) ? (height / size) : 0;
    int rem  = (size > 0) ? (height % size) : 0;
    int local_rows = base + ((rank < rem) ? 1 : 0);
    int y_start    = rank * base + ((rank < rem) ? rank : rem);

    int *sc_counts = NULL;
    int *sc_displs = NULL;
    
    if (rank == 0) {
        sc_counts = malloc(size * sizeof(int));
        sc_displs = malloc(size * sizeof(int));
        int disp = 0;
        for (int r = 0; r < size; r++) {
            sc_counts[r] = (base + ((r < rem) ? 1 : 0)) * width;
            sc_displs[r] = disp;
            disp += sc_counts[r];
        }
    }

    int local_elems = local_rows * width;
    unsigned char *local_edge = malloc((local_elems > 0 ? local_elems : 1) * sizeof(unsigned char));

    MPI_Scatterv(edge_img, sc_counts, sc_displs, MPI_UNSIGNED_CHAR,
                 local_edge, local_elems, MPI_UNSIGNED_CHAR, 0, comm);

    if (rank == 0) { free(sc_counts); free(sc_displs); }

    // ---------------------------------------------------------
    // 2. PRE-CALCOLO TABELLE TRIGONOMETRICHE (PUNTO FISSO)
    // ---------------------------------------------------------
    int cos_fixed[THETA_STEPS], sin_fixed[THETA_STEPS];
    for (int t = 0; t < theta_dim; t++) {
        float rad = t * 3.1415926535f / (float)THETA_STEPS;
        // Moltiplichiamo per 2048 e arrotondiamo a intero
        cos_fixed[t] = (int)roundf(cosf(rad) * SCALE);
        sin_fixed[t] = (int)roundf(sinf(rad) * SCALE);
    }

    // ---------------------------------------------------------
    // 3. ESTRAZIONE SPARSE DELLE COORDINATE
    // ---------------------------------------------------------
    // Contiamo i pixel bianchi per allocare esattamente la memoria necessaria
    int local_edges_count = 0;
    for (int i = 0; i < local_elems; i++) {
        if (local_edge[i] > 0) local_edges_count++;
    }

    int alloc_edges = local_edges_count > 0 ? local_edges_count : 1;
    int *x_coords = malloc(alloc_edges * sizeof(int));
    int *y_coords = malloc(alloc_edges * sizeof(int));

    int idx = 0;
    for (int ly = 0; ly < local_rows; ly++) {
        int global_y = y_start + ly;
        for (int x = 0; x < width; x++) {
            if (local_edge[ly * width + x] > 0) {
                x_coords[idx] = x;
                y_coords[idx] = global_y;
                idx++;
            }
        }
    }
    free(local_edge); // La matrice immagine non serve più!

    // ---------------------------------------------------------
    // 4. CALCOLO OPENMP: ACCUMULATORI PRIVATI E PUNTO FISSO
    // ---------------------------------------------------------
    size_t acc_len = (size_t)rho_dim * (size_t)theta_dim;
    int nthreads = omp_get_max_threads();
    
    // Matrice gigante piatta che contiene un accumulatore 1D per ogni thread
    int *thread_accs = calloc(nthreads * acc_len, sizeof(int));
    int *local_acc = calloc(acc_len, sizeof(int));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        // Puntatore diretto all'accumulatore esclusivo per questo thread
        int *my_acc = &thread_accs[tid * acc_len]; 

        // Poiché x_coords è denso (nessuno zero!), il carico è perfettamente bilanciato
        #pragma omp for schedule(static)
        for (int i = 0; i < local_edges_count; i++) {
            int px = x_coords[i];
            int py = y_coords[i];

            for (int t = 0; t < theta_dim; t++) {
                // IL CUORE DELL'OTTIMIZZAZIONE: Solo somme, moltiplicazioni e uno shift bit a bit (>> 11)
                // Nessun cast, nessuna FPU (Floating Point Unit) utilizzata!
                int r = ((px * cos_fixed[t] + py * sin_fixed[t]) >> SHIFT) + (rho_dim / 2);
                if (r >= 0 && r < rho_dim) {
                    my_acc[r * theta_dim + t]++;
                }
            }
        }

        // Riduzione finale: il nodo somma i risultati di tutti i suoi thread privati
        #pragma omp for schedule(static)
        for (size_t j = 0; j < acc_len; j++) {
            int sum = 0;
            for (int k = 0; k < nthreads; k++) {
                sum += thread_accs[k * acc_len + j];
            }
            local_acc[j] = sum;
        }
    }

    free(x_coords); 
    free(y_coords); 
    free(thread_accs);

    // ---------------------------------------------------------
    // 5. MPI ALLREDUCE E NON-MAXIMUM SUPPRESSION PARALLELA
    // ---------------------------------------------------------
    int *global_acc = calloc(acc_len, sizeof(int));
    MPI_Allreduce(local_acc, global_acc, (int)acc_len, MPI_INT, MPI_SUM, comm);
    free(local_acc);

    int **acc2d = malloc(rho_dim * sizeof(int*));
    for (int r = 0; r < rho_dim; r++) acc2d[r] = &global_acc[r * theta_dim];

    // Distribuzione del carico per NMS (Ogni rank fa una fetta di RHO)
    int nms_base = rho_dim / size;
    int nms_rem  = rho_dim % size;
    int my_rho_count = nms_base + ((rank < nms_rem) ? 1 : 0);
    int my_rho_start = rank * nms_base + ((rank < nms_rem) ? rank : nms_rem);
    int my_rho_end   = my_rho_start + my_rho_count;

    int capacity = 1000;
    Line *local_lines = malloc(capacity * sizeof(Line));
    int local_count = 0;

    for (int r = my_rho_start; r < my_rho_end; r++) {
        for (int t = 0; t < theta_dim; t++) {
            if (acc2d[r][t] > threshold && NMS_max(acc2d, r, t, rho_dim, theta_dim)) {
                if (local_count >= capacity) {
                    capacity *= 2;
                    Line *temp = realloc(local_lines, capacity * sizeof(Line));
                    if (!temp) break; 
                    local_lines = temp;
                }
                local_lines[local_count].r = r - (rho_dim / 2);
                local_lines[local_count].t = t;
                local_count++;
            }
        }
    }

    // ---------------------------------------------------------
    // 6. GATHER FINALE SU RANK 0
    // ---------------------------------------------------------
    int *ga_counts = NULL;
    int *ga_displs = NULL;
    if (rank == 0) {
        ga_counts = malloc(size * sizeof(int));
        ga_displs = malloc(size * sizeof(int));
    }

    MPI_Gather(&local_count, 1, MPI_INT, ga_counts, 1, MPI_INT, 0, comm);

    Lines *final_result = NULL;
    if (rank == 0) {
        final_result = malloc(sizeof(Lines));
        final_result->count = 0;
        for (int i = 0; i < size; i++) {
            ga_displs[i] = final_result->count;
            final_result->count += ga_counts[i];
        }
        final_result->lines = malloc((final_result->count > 0 ? final_result->count : 1) * sizeof(Line));
    }

    int bytes_per_line = sizeof(Line); 
    if (rank == 0) {
        for(int i = 0; i < size; i++) {
            ga_counts[i] *= bytes_per_line;
            ga_displs[i] *= bytes_per_line;
        }
    }

    MPI_Gatherv(local_lines, local_count * bytes_per_line, MPI_BYTE,
                (rank == 0) ? final_result->lines : NULL, ga_counts, ga_displs, MPI_BYTE, 
                0, comm);

    free(local_lines); 
    free(acc2d); 
    free(global_acc);
    if (rank == 0) { 
        free(ga_counts); 
        free(ga_displs); 
    }

    return final_result;
}

Lines* HoughLines_Hybrid_Tiled(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int rho_dim = compute_rho(width, height);
    int theta_dim = THETA_STEPS;

    // 1. SCATTER MPI (Ometto i dettagli per brevità, identico a prima)
    int base = (size > 0) ? (height / size) : 0;
    int rem  = (size > 0) ? (height % size) : 0;
    int local_rows = base + ((rank < rem) ? 1 : 0);
    int y_start    = rank * base + ((rank < rem) ? rank : rem);

    int *sc_counts = NULL;
    int *sc_displs = NULL;
    if (rank == 0) {
        sc_counts = malloc(size * sizeof(int));
        sc_displs = malloc(size * sizeof(int));
        int disp = 0;
        for (int r = 0; r < size; r++) {
            sc_counts[r] = (base + ((r < rem) ? 1 : 0)) * width;
            sc_displs[r] = disp;
            disp += sc_counts[r];
        }
    }
    int local_elems = local_rows * width;
    unsigned char *local_edge = malloc((local_elems > 0 ? local_elems : 1) * sizeof(unsigned char));
    MPI_Scatterv(edge_img, sc_counts, sc_displs, MPI_UNSIGNED_CHAR, local_edge, local_elems, MPI_UNSIGNED_CHAR, 0, comm);
    if (rank == 0) { free(sc_counts); free(sc_displs); }

    // 2. PRE-CALCOLO TABELLE (Con BUG-FIX per supportare theta_dim variabili!)
    int *cos_fixed = malloc(theta_dim * sizeof(int));
    int *sin_fixed = malloc(theta_dim * sizeof(int));
    for (int t = 0; t < theta_dim; t++) {
        // Mappa dinamicamente [0, theta_dim] -> [0, PI]
        float rad = (float)t * 3.1415926535f / (float)theta_dim; 
        cos_fixed[t] = (int)roundf(cosf(rad) * SCALE);
        sin_fixed[t] = (int)roundf(sinf(rad) * SCALE);
    }

    // 3. ESTRAZIONE SPARSE
    int local_edges_count = 0;
    for (int i = 0; i < local_elems; i++) {
        if (local_edge[i] > 0) local_edges_count++;
    }
    int alloc_edges = local_edges_count > 0 ? local_edges_count : 1;
    int *x_coords = malloc(alloc_edges * sizeof(int));
    int *y_coords = malloc(alloc_edges * sizeof(int));
    int idx = 0;
    for (int ly = 0; ly < local_rows; ly++) {
        int global_y = y_start + ly;
        for (int x = 0; x < width; x++) {
            if (local_edge[ly * width + x] > 0) {
                x_coords[idx] = x;
                y_coords[idx++] = global_y;
            }
        }
    }
    free(local_edge);

    // 4. CALCOLO OPENMP: ACCUMULATORI PRIVATI + CACHE BLOCKING
    size_t acc_len = (size_t)rho_dim * (size_t)theta_dim;
    int nthreads = omp_get_max_threads();
    int *thread_accs = calloc(nthreads * acc_len, sizeof(int));
    int *local_acc = calloc(acc_len, sizeof(int));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int *my_acc = &thread_accs[tid * acc_len]; 

        // Il parallelismo si applica sui "blocchi di pixel"
        #pragma omp for schedule(dynamic)
        for (int p_base = 0; p_base < local_edges_count; p_base += TILE_P) {
            int p_end = (p_base + TILE_P < local_edges_count) ? p_base + TILE_P : local_edges_count;

            // Iteriamo a blocchi di angoli
            for (int t_base = 0; t_base < theta_dim; t_base += TILE_T) {
                int t_end = (t_base + TILE_T < theta_dim) ? t_base + TILE_T : theta_dim;

                // MAGIA DEL CACHE HIT: Ciclo interno sui PIXEL, non sugli angoli.
                // Teniamo 't' fisso, così scriviamo sempre sulla stessa colonna dell'array!
                for (int t = t_base; t < t_end; t++) {
                    for (int p = p_base; p < p_end; p++) {
                        int r = ((x_coords[p] * cos_fixed[t] + y_coords[p] * sin_fixed[t]) >> SHIFT) + (rho_dim / 2);
                        if (r >= 0 && r < rho_dim) {
                            my_acc[r * theta_dim + t]++;
                        }
                    }
                }
            }
        }

        // Riduzione locale
        #pragma omp for schedule(static)
        for (size_t j = 0; j < acc_len; j++) {
            int sum = 0;
            for (int k = 0; k < nthreads; k++) sum += thread_accs[k * acc_len + j];
            local_acc[j] = sum;
        }
    }

    free(x_coords); free(y_coords); free(thread_accs);
    free(cos_fixed); free(sin_fixed);

    // 5. MPI ALLREDUCE, NMS e GATHER (Identici a Optimized)
    int *global_acc = calloc(acc_len, sizeof(int));
    MPI_Allreduce(local_acc, global_acc, (int)acc_len, MPI_INT, MPI_SUM, comm);
    free(local_acc);

    int **acc2d = malloc(rho_dim * sizeof(int*));
    for (int r = 0; r < rho_dim; r++) acc2d[r] = &global_acc[r * theta_dim];

    int nms_base = rho_dim / size;
    int nms_rem  = rho_dim % size;
    int my_rho_start = rank * nms_base + ((rank < nms_rem) ? rank : nms_rem);
    int my_rho_end   = my_rho_start + nms_base + ((rank < nms_rem) ? 1 : 0);

    int capacity = 1000;
    Line *local_lines = malloc(capacity * sizeof(Line));
    int local_count = 0;

    for (int r = my_rho_start; r < my_rho_end; r++) {
        for (int t = 0; t < theta_dim; t++) {
            if (acc2d[r][t] > threshold && NMS_max(acc2d, r, t, rho_dim, theta_dim)) {
                if (local_count >= capacity) {
                    capacity *= 2;
                    Line *temp = realloc(local_lines, capacity * sizeof(Line));
                    if (!temp) break; 
                    local_lines = temp;
                }
                local_lines[local_count].r = r - (rho_dim / 2);
                local_lines[local_count].t = t;
                local_count++;
            }
        }
    }

    int *ga_counts = NULL; int *ga_displs = NULL;
    if (rank == 0) { ga_counts = malloc(size * sizeof(int)); ga_displs = malloc(size * sizeof(int)); }
    MPI_Gather(&local_count, 1, MPI_INT, ga_counts, 1, MPI_INT, 0, comm);

    Lines *final_result = NULL;
    if (rank == 0) {
        final_result = malloc(sizeof(Lines));
        final_result->count = 0;
        for (int i = 0; i < size; i++) {
            ga_displs[i] = final_result->count;
            final_result->count += ga_counts[i];
        }
        final_result->lines = malloc((final_result->count > 0 ? final_result->count : 1) * sizeof(Line));
        for(int i = 0; i < size; i++) { ga_counts[i] *= sizeof(Line); ga_displs[i] *= sizeof(Line); }
    }

    MPI_Gatherv(local_lines, local_count * sizeof(Line), MPI_BYTE,
                (rank == 0) ? final_result->lines : NULL, ga_counts, ga_displs, MPI_BYTE, 0, comm);

    free(local_lines); free(acc2d); free(global_acc);
    if (rank == 0) { free(ga_counts); free(ga_displs); }

    return final_result;
}

// ==========================================
// PURO MPI (CERCHI) - CORRETTO
// ==========================================
Circle* HoughCircles_PureMPI(int* x_coords, int* y_coords, int num_edges, 
                             int width, int height, 
                             int r_min, int r_max, int threshold, 
                             MPI_Comm comm, int* out_count) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int total_r = r_max - r_min + 1;
    int base_r = total_r / size;
    int rem_r = total_r % size;
    int local_r_count = base_r + ((rank < rem_r) ? 1 : 0);
    int local_r_start = r_min + rank * base_r + ((rank < rem_r) ? rank : rem_r);
    int local_r_end = local_r_start + local_r_count - 1;

    // FIX: Usiamo 'capacity' per gestire l'allocazione dinamica
    int capacity = 1000;
    Circle* local_circles = malloc(capacity * sizeof(Circle));
    int local_count = 0;

    int *local_acc2D = malloc(width * height * sizeof(int));

    for (int r = local_r_start; r <= local_r_end; r++) {
        for (int i = 0; i < width * height; i++) local_acc2D[i] = 0;

        for (int e = 0; e < num_edges; e++) {
            bresenham_vote(local_acc2D, width, height, x_coords[e], y_coords[e], r);
        }

        int dyn_thresh = (int)(2.0 * 3.14159265 * r * 0.50); 
        if (dyn_thresh < 25) { 
            dyn_thresh = 25; 
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (local_acc2D[y * width + x] >= dyn_thresh) {
                    
                    // FIX: Allocazione dinamica ripristinata
                    if (local_count >= capacity) {
                        capacity *= 2;
                        Circle* temp = realloc(local_circles, capacity * sizeof(Circle));
                        if (temp) local_circles = temp;
                    }
                    local_circles[local_count++] = (Circle){x, y, r, local_acc2D[y * width + x]};
                }
            }
        }
    }
    free(local_acc2D);

    int *ga_counts = NULL; int *ga_displs = NULL;
    if (rank == 0) {
        ga_counts = malloc(size * sizeof(int));
        ga_displs = malloc(size * sizeof(int));
    }

    MPI_Gather(&local_count, 1, MPI_INT, ga_counts, 1, MPI_INT, 0, comm);

    Circle *final_result = NULL;
    int total_circles = 0;

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            ga_displs[i] = total_circles;
            total_circles += ga_counts[i];
            ga_counts[i] *= sizeof(Circle);
            ga_displs[i] *= sizeof(Circle);
        }
        final_result = malloc((total_circles > 0 ? total_circles : 1) * sizeof(Circle));
    }

    MPI_Gatherv(local_circles, local_count * sizeof(Circle), MPI_BYTE,
                final_result, ga_counts, ga_displs, MPI_BYTE, 0, comm);

    free(local_circles);
    if (rank == 0) { free(ga_counts); free(ga_displs); }

    *out_count = total_circles;
    return final_result;
}


// ==========================================
// IBRIDO MPI+OMP (CERCHI) - CORRETTO
// ==========================================
Circle* HoughCircles_Hybrid(int* x_coords, int* y_coords, int num_edges, 
                            int width, int height, 
                            int r_min, int r_max, int threshold, 
                            MPI_Comm comm, int* out_count) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int total_r = r_max - r_min + 1;
    int base_r = total_r / size;
    int rem_r = total_r % size;
    int local_r_count = base_r + ((rank < rem_r) ? 1 : 0);
    int local_r_start = r_min + rank * base_r + ((rank < rem_r) ? rank : rem_r);
    int local_r_end = local_r_start + local_r_count - 1;

    // FIX: Variabile coerente per l'allocazione
    int capacity = 1000;
    Circle* local_circles = malloc(capacity * sizeof(Circle));
    int local_count = 0;

    #pragma omp parallel
    {
        int *thread_acc2D = malloc(width * height * sizeof(int));

        #pragma omp for schedule(dynamic)
        for (int r = local_r_start; r <= local_r_end; r++) {
            
            for (int i = 0; i < width * height; i++) thread_acc2D[i] = 0;

            for (int e = 0; e < num_edges; e++) {
                bresenham_vote(thread_acc2D, width, height, x_coords[e], y_coords[e], r);
            }

            int dyn_thresh = (int)(2.0 * 3.14159265 * r * 0.45); 
            
            // FIX: Soglia minima corretta contro l'OOM
            if (dyn_thresh < 25) {
                dyn_thresh = 25;
            }

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (thread_acc2D[y * width + x] >= dyn_thresh) {
                        #pragma omp critical
                        {
                            // FIX: La condizione usa 'capacity', non 'dyn_thresh'
                            if (local_count >= capacity) {
                                capacity *= 2;
                                Circle* temp = realloc(local_circles, capacity * sizeof(Circle));
                                if (temp) local_circles = temp;
                            }
                            local_circles[local_count++] = (Circle){x, y, r, thread_acc2D[y * width + x]};
                        }
                    }
                }
            }
        }
        free(thread_acc2D);
    }

    int *ga_counts = NULL; int *ga_displs = NULL;
    if (rank == 0) { ga_counts = malloc(size * sizeof(int)); ga_displs = malloc(size * sizeof(int)); }

    MPI_Gather(&local_count, 1, MPI_INT, ga_counts, 1, MPI_INT, 0, comm);

    Circle *final_result = NULL;
    int total_circles = 0;

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            ga_displs[i] = total_circles;
            total_circles += ga_counts[i];
            ga_counts[i] *= sizeof(Circle);
            ga_displs[i] *= sizeof(Circle);
        }
        final_result = malloc((total_circles > 0 ? total_circles : 1) * sizeof(Circle));
    }

    MPI_Gatherv(local_circles, local_count * sizeof(Circle), MPI_BYTE,
                final_result, ga_counts, ga_displs, MPI_BYTE, 0, comm);

    free(local_circles);
    if (rank == 0) { free(ga_counts); free(ga_displs); }

    *out_count = total_circles;
    return final_result;
}
