#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "parallel.h"

#define THETA_STEPS 180


// ––– Pure MPI –––
// The image is divided into rows (height / size) and scattered across each node
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
        float rad = (float)t * 3.1415926535f / 180.0f;
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

    // Allocate global accumulator on Rank 0
    // int *global_acc = NULL;
    // if (rank == 0) {
    //     global_acc = calloc(acc_len, sizeof(int));
    //     if (!global_acc) {
    //         free(local_acc);
    //         MPI_Abort(comm, 1);
    //     }
    // }

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
        const double rad = (double)t * pi / 180.0;
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
        float rad = (float)t * 3.1415926535f / 180.0f;
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
