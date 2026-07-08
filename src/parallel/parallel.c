#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "parallel.h"


Circle* CHT_Parameter_MPI(int* x_coords, int* y_coords, int num_edges, 
                                 int width, int height, 
                                 int r_min, int r_max, float threshold, float* theta_coords,
                                 MPI_Comm comm, int* out_count) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Creazione Datatype MPI per la struct
    MPI_Datatype mpi_circle_type;
    MPI_Type_contiguous(sizeof(Circle), MPI_BYTE, &mpi_circle_type);
    MPI_Type_commit(&mpi_circle_type);

    // Radius load balance with Round-Robin
    int num_local_radii = 0;
    for (int r = r_min + rank; r <= r_max; r += size) {
        num_local_radii++;
    }

    int capacity = 100;
    Circle* local_circles = malloc(capacity * sizeof(Circle));
    int local_count = 0;
    int *local_acc2D = calloc(width * height, sizeof(int));

    // Pre-calculate sine and cosine for every edge point
    float* cos_theta = malloc(num_edges * sizeof(float));
    float* sin_theta = malloc(num_edges * sizeof(float));
    for (int e = 0; e < num_edges; e++) {
        cos_theta[e] = cosf(theta_coords[e]);
        sin_theta[e] = sinf(theta_coords[e]);
    }

    for (int i = 0; i < num_local_radii; i++) {
        int r = r_min + rank + i * size;
        memset(local_acc2D, 0, width * height * sizeof(int));

        for (int e = 0; e < num_edges; e++) {
            int x = x_coords[e];
            int y = y_coords[e];
            
            // Calcolo dei due possibili centri candidati lungo la normale
            int xc = (int)(x - r * cos_theta[e] + 0.5f);
            int yc = (int)(y - r * sin_theta[e] + 0.5f);
            
            // Voto 1: Controllo dei confini e incremento cache-friendly,
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = xc + dx;
                    int ny = yc + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        local_acc2D[ny * width + nx]++;
                    }
                }
            }
        }

        // Getting the maximum of acc values'
        int max_v = 0;
        for(int i=0; i<width*height; i++) if(local_acc2D[i] > max_v) max_v = local_acc2D[i];

        int min_absolute_votes = 3;
        if (max_v < min_absolute_votes) {
            continue;
        }

        float threshold_n = max_v * threshold;
        // printf("DEBUG Rank %d, Raggio %d -> max_v: %d, threshold_float: %f, threshold_n: %f\n", rank, r, max_v, threshold, threshold_n);

        // Estrazione dei massimi
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (local_acc2D[y * width + x] >= threshold_n) {
                    if (NMS_max_circles(local_acc2D, x, y, width, height)) {
                        if (local_count >= capacity) {
                            capacity *= 2;
                            Circle* temp = realloc(local_circles, capacity * sizeof(Circle));
                            if (temp) local_circles = temp;
                            else { MPI_Abort(comm, 1); }
                        }
                        local_circles[local_count++] = (Circle){x, y, r, local_acc2D[y * width + x]};
                    }
                }    
            }
        }
    }
    
    free(local_acc2D);
    free(cos_theta);
    free(sin_theta);

    // MPI Sync 
    int *ga_counts = NULL; 
    int *ga_displs = NULL;
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
        }
        final_result = malloc((total_circles > 0 ? total_circles : 1) * sizeof(Circle));
    }

    MPI_Gatherv(local_circles, local_count, mpi_circle_type,
                final_result, ga_counts, ga_displs, mpi_circle_type, 0, comm);
    if (rank == 0) {filter_by_statistics(final_result, total_circles, threshold, out_count);}


    free(local_circles);
    MPI_Type_free(&mpi_circle_type);
    if (rank == 0) { free(ga_counts); free(ga_displs); }

    return final_result;
}

Circle* CHT_Parameter_MPI_opt(int* x_coords, int* y_coords, int num_edges, 
                                 int width, int height, 
                                 int r_min, int r_max, float threshold, float* theta_coords,
                                 MPI_Comm comm, int* out_count) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MPI_Datatype mpi_circle_type;
    MPI_Type_contiguous(sizeof(Circle), MPI_BYTE, &mpi_circle_type);
    MPI_Type_commit(&mpi_circle_type);

    int num_local_radii = 0;
    for (int r = r_min + rank; r <= r_max; r += size) {
        num_local_radii++;
    }

    int capacity = 100;
    Circle* local_circles = malloc(capacity * sizeof(Circle));
    int local_count = 0;
    
    int *local_acc2D = calloc((size_t)width * height, sizeof(int));

    // Array per tracciare le celle modificate (Sparse Matrix Pattern)
    size_t touched_cap = (size_t)num_edges * 9; 
    size_t wh = (size_t)width * height;
    if (touched_cap > wh) touched_cap = wh;
    if (touched_cap == 0) touched_cap = 1;
    int* touched = malloc(touched_cap * sizeof(int));

    float* cos_theta = malloc(num_edges * sizeof(float));
    float* sin_theta = malloc(num_edges * sizeof(float));
    for (int e = 0; e < num_edges; e++) {
        cos_theta[e] = cosf(theta_coords[e]);
        sin_theta[e] = sinf(theta_coords[e]);
    }

    for (int i = 0; i < num_local_radii; i++) {
        int r = r_min + rank + i * size;
        int touched_count = 0;

        for (int e = 0; e < num_edges; e++) {
            int xc = (int)(x_coords[e] - r * cos_theta[e] + 0.5f);
            int yc = (int)(y_coords[e] - r * sin_theta[e] + 0.5f);
            
            // Branchless Voting
            // Se il centro 3x3 è lontano dai bordi, evitiamo del tutto gli IF
            if (xc >= 1 && xc < width - 1 && yc >= 1 && yc < height - 1) {
                for (int dy = -1; dy <= 1; dy++) {
                    int base_idx = (yc + dy) * width + xc;
                    for (int dx = -1; dx <= 1; dx++) {
                        int idx = base_idx + dx;
                        if (local_acc2D[idx] == 0) {
                            touched[touched_count++] = idx;
                        }
                        local_acc2D[idx]++;
                    }
                }
            } else {
                // Per i rari pixel vicino ai bordi dell'immagine
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = xc + dx;
                        int ny = yc + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            int idx = ny * width + nx;
                            if (local_acc2D[idx] == 0) {
                                touched[touched_count++] = idx;
                            }
                            local_acc2D[idx]++;
                        }
                    }
                }
            }
        }

        // Trova il massimo solo nelle celle touched
        int max_v = 0;
        for (int k = 0; k < touched_count; k++) {
            int v = local_acc2D[touched[k]];
            if (v > max_v) max_v = v;
        }

        int min_absolute_votes = 3;
        if (max_v >= min_absolute_votes) {
            float threshold_n = max_v * threshold;

            // Estrazione picchi solo nelle celle touched
            for (int k = 0; k < touched_count; k++) {
                int idx = touched[k];
                if (local_acc2D[idx] >= threshold_n) {
                    int x = idx % width;
                    int y = idx / width;
                    if (NMS_max_circles(local_acc2D, x, y, width, height)) {
                        if (local_count >= capacity) {
                            capacity *= 2;
                            Circle* temp = realloc(local_circles, capacity * sizeof(Circle));
                            if (temp) local_circles = temp;
                            else { MPI_Abort(comm, 1); }
                        }
                        local_circles[local_count++] = (Circle){x, y, r, local_acc2D[idx]};
                    }
                }
            }
        }

        // Invece del memset da O(W*H), azzeriamo solo a O(N_voti)        
        for (int k = 0; k < touched_count; k++) {
            local_acc2D[touched[k]] = 0;
        }
    }
    
    free(touched);
    free(local_acc2D);
    free(cos_theta);
    free(sin_theta);

    // --- MPI Sync ---
    int *ga_counts = NULL; 
    int *ga_displs = NULL;
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
        }
        final_result = malloc((total_circles > 0 ? total_circles : 1) * sizeof(Circle));
    }

    MPI_Gatherv(local_circles, local_count, mpi_circle_type,
                final_result, ga_counts, ga_displs, mpi_circle_type, 0, comm);
    
    if (rank == 0) { filter_by_statistics(final_result, total_circles, threshold, out_count); }

    free(local_circles);
    MPI_Type_free(&mpi_circle_type);
    if (rank == 0) { free(ga_counts); free(ga_displs); }

    return final_result;
}

Circle* CHT_Parameter_Hybrid(int* x_coords, int* y_coords, int num_edges, 
                                 int width, int height, 
                                 int r_min, int r_max, float threshold, float* theta_coords,
                                 MPI_Comm comm, int* out_count) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Creazione Datatype MPI per la struct
    MPI_Datatype mpi_circle_type;
    MPI_Type_contiguous(sizeof(Circle), MPI_BYTE, &mpi_circle_type);
    MPI_Type_commit(&mpi_circle_type);

    // Bilanciamento Round-Robin dei raggi
    int num_local_radii = 0;
    for (int r = r_min + rank; r <= r_max; r += size) num_local_radii++;

    int capacity = 100;
    Circle* local_circles = malloc(capacity * sizeof(Circle));
    int local_count = 0;

   
    // Pre-calcolo di seno e coseno per ogni edge.
    float* cos_theta = malloc(num_edges * sizeof(float));
    float* sin_theta = malloc(num_edges * sizeof(float));
    #pragma omp parallel for
    for (int e = 0; e < num_edges; e++) {
        cos_theta[e] = cosf(theta_coords[e]);
        sin_theta[e] = sinf(theta_coords[e]);
    }
    
    #pragma omp parallel
    {
        int *local_acc2D = calloc(width * height, sizeof(int));

        int thread_capacity = 20; 
        Circle* thread_circles = malloc(thread_capacity * sizeof(Circle));
        int thread_count = 0;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_local_radii; i++) {
            int r = r_min + rank + i * size;
            memset(local_acc2D, 0, width * height * sizeof(int));

            for (int e = 0; e < num_edges; e++) {
                int x = x_coords[e];
                int y = y_coords[e];
                
                int xc = (int)(x - r * cos_theta[e] + 0.5f);
                int yc = (int)(y - r * sin_theta[e] + 0.5f);
                
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = xc + dx;
                        int ny = yc + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            local_acc2D[ny * width + nx]++;
                        }
                    }
                }
            }

            int max_v = 0;
            for(int i=0; i<width*height; i++) if(local_acc2D[i] > max_v) max_v = local_acc2D[i];

            int min_absolute_votes = 3;
            if (max_v < min_absolute_votes) {
                continue;
            }

            float threshold_n = max_v * threshold;

            // Peak Extraction
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (local_acc2D[y * width + x] >= threshold_n) {
                        if (NMS_max_circles(local_acc2D, x, y, width, height)) {
                                if (thread_count >= thread_capacity) {
                                    thread_capacity *= 2;
                                    thread_circles = realloc(thread_circles, thread_capacity * sizeof(Circle));  
                                }
                                thread_circles[thread_count++] = (Circle){x, y, r, local_acc2D[y * width + x]};
                            
                        }
                    }    
                }
            }
        }

        #pragma omp critical
        {
            for (int k = 0; k < thread_count; k++) {
                if (local_count >= capacity) {
                    capacity *= 2;
                    local_circles = realloc(local_circles, capacity * sizeof(Circle));
                }
                local_circles[local_count++] = thread_circles[k];
            }
        }
        free(thread_circles);
        free(local_acc2D);
    }

    free(cos_theta);
    free(sin_theta);

    int *ga_counts = NULL; 
    int *ga_displs = NULL;
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
        }
        final_result = malloc((total_circles > 0 ? total_circles : 1) * sizeof(Circle));
    }

    MPI_Gatherv(local_circles, local_count, mpi_circle_type,
                final_result, ga_counts, ga_displs, mpi_circle_type, 0, comm);
    if (rank == 0) {filter_by_statistics(final_result, total_circles, threshold, out_count);}

    free(local_circles);
    MPI_Type_free(&mpi_circle_type);
    if (rank == 0) { free(ga_counts); free(ga_displs); }

    // *out_count = total_circles;
    return final_result;
}
