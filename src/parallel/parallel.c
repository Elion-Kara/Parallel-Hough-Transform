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

        int min_absolute_votes = 10;
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

            int min_absolute_votes = 10;
            if (max_v < min_absolute_votes) {
                continue;
            }

            float threshold_n = max_v * threshold;
            // printf("DEBUG Rank %d, Raggio %d -> max_v: %d, threshold_float: %f, threshold_n: %f\n", rank, r, max_v, threshold, threshold_n);

            // Estrazione dei massimi
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (local_acc2D[y * width + x] >= threshold_n) {
                        // printf("votes:%d\n", local_acc2D[y * width + x]);
                        if (NMS_max_circles(local_acc2D, x, y, width, height)) {
                            #pragma omp critical
                            {
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
        }
        
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

Circle* CHT_Domain_MPI(int* x_coords, int* y_coords, int num_edges, 
                               int width, int height, 
                               int r_min, int r_max, float threshold, float* theta_coords,
                               MPI_Comm comm, int* out_count) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 1. SPATIAL DECOMPOSITION: Dividiamo l'immagine (gli edges) tra i rank
    int edges_per_rank = num_edges / size;
    int start_edge = rank * edges_per_rank;
    int end_edge = (rank == size - 1) ? num_edges : start_edge + edges_per_rank;

    int capacity = 100;
    Circle* local_circles = malloc(capacity * sizeof(Circle));
    int local_count = 0;
    
    // Accumulatore locale per i voti del sottoinsieme di edges
    int *local_acc2D = calloc(width * height, sizeof(int));
    
    // Accumulatore globale (serve solo al Rank 0 per unire i risultati)
    int *global_acc2D = NULL;
    if (rank == 0) {
        global_acc2D = calloc(width * height, sizeof(int));
    }

    // Pre-calcolo seno e coseno solo per gli edge di competenza
    float* cos_theta = malloc(num_edges * sizeof(float));
    float* sin_theta = malloc(num_edges * sizeof(float));
    for (int e = start_edge; e < end_edge; e++) {
        cos_theta[e] = cosf(theta_coords[e]);
        sin_theta[e] = sinf(theta_coords[e]);
    }

    // 2. TUTTI i rank devono iterare su TUTTI i raggi
    for (int r = r_min; r <= r_max; r++) {
        memset(local_acc2D, 0, width * height * sizeof(int));

        // 3. Ogni rank vota SOLO per la sua "fetta" di immagine
        for (int e = start_edge; e < end_edge; e++) {
            int x = x_coords[e];
            int y = y_coords[e];
            
            int xc1 = (int)(x - r * cos_theta[e] + 0.5f);
            int yc1 = (int)(y - r * sin_theta[e] + 0.5f);

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = xc1 + dx;
                    int ny = yc1 + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        local_acc2D[ny * width + nx]++;
                    }
                }
            }
        }

        // 4. IL COLLO DI BOTTIGLIA (NETWORK OVERHEAD)
        // Poiché i cerchi oltrepassano i confini della "fetta" di immagine assegnata al rank,
        // dobbiamo unire gli accumulatori locali in un accumulatore globale per OGNI raggio.
        MPI_Reduce(local_acc2D, global_acc2D, width * height, MPI_INT, MPI_SUM, 0, comm);

        // 5. Solo il Rank 0 ha la visione globale dell'accumulatore e può estrarre i picchi
        if (rank == 0) {
            int max_v = 0;
            for(int i = 0; i < width * height; i++) {
                if(global_acc2D[i] > max_v) max_v = global_acc2D[i];
            }

            int min_absolute_votes = 3;
            if (max_v >= min_absolute_votes) {
                float threshold_n = max_v * threshold;

                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        if (global_acc2D[y * width + x] >= threshold_n) {
                            if (NMS_max_circles(global_acc2D, x, y, width, height)) {
                                if (local_count >= capacity) {
                                    capacity *= 2;
                                    local_circles = realloc(local_circles, capacity * sizeof(Circle));
                                }
                                local_circles[local_count++] = (Circle){x, y, r, global_acc2D[y * width + x]};
                            }
                        }    
                    }
                }
            }
        }
    }
    
    free(local_acc2D);
    if (rank == 0) free(global_acc2D);
    free(cos_theta);
    free(sin_theta);

    // Dato che solo il Rank 0 ha estratto i cerchi, non serve l'MPI_Gatherv finale.
    // Il Rank 0 applica la statistica e restituisce il risultato.
    if (rank == 0) {
        filter_by_statistics(local_circles, local_count, threshold, out_count);
        return local_circles;
    } else {
        free(local_circles);
        *out_count = 0;
        return NULL;
    }
}

Circle* CHT_Domain_MPI_Optimized(int* x_coords, int* y_coords, int num_edges, 
                                        int width, int height, 
                                        int r_min, int r_max, float threshold, float* theta_coords,
                                        MPI_Comm comm, int* out_count) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Creazione Datatype MPI per la struct Circle
    MPI_Datatype mpi_circle_type;
    MPI_Type_contiguous(sizeof(Circle), MPI_BYTE, &mpi_circle_type);
    MPI_Type_commit(&mpi_circle_type);

    // 1. SPATIAL SPLIT (Sull'Accumulatore, non sugli edges)
    // Dividiamo l'altezza dell'immagine in "fette" orizzontali
    int slice_h = height / size;
    int y_start = rank * slice_h;
    int y_end = (rank == size - 1) ? height : y_start + slice_h;
    int local_height = y_end - y_start;

    // L'accumulatore ora è molto più piccolo! Risparmio enorme di RAM.
    int *local_slice_acc = calloc(width * local_height, sizeof(int));
    
    int capacity = 100;
    Circle* local_circles = malloc(capacity * sizeof(Circle));
    int local_count = 0;

    // Pre-calcolo seno e coseno per TUTTI gli edge, dato che ogni rank 
    // deve processare l'intera lista per capire se il centro cade nella sua fetta.
    float* cos_theta = malloc(num_edges * sizeof(float));
    float* sin_theta = malloc(num_edges * sizeof(float));
    for (int e = 0; e < num_edges; e++) {
        cos_theta[e] = cosf(theta_coords[e]);
        sin_theta[e] = sinf(theta_coords[e]);
    }

    // TUTTI i rank processano TUTTI i raggi
    for (int r = r_min; r <= r_max; r++) {
        memset(local_slice_acc, 0, width * local_height * sizeof(int));

        // TUTTI i rank iterano su TUTTI gli edges
        // (Qui puoi usare OpenMP in modo sicuro: #pragma omp parallel for private(x, y, xc1, yc1) etc.)
        for (int e = 0; e < num_edges; e++) {
            int x = x_coords[e];
            int y = y_coords[e];
            
            int xc1 = (int)(x - r * cos_theta[e] + 0.5f);
            int yc1 = (int)(y - r * sin_theta[e] + 0.5f);

            // IL FILTRO GEOMETRICO: Vota solo se il centro cade nella tua "Fetta" (Slice)
            if (yc1 >= y_start && yc1 < y_end && xc1 >= 0 && xc1 < width) {
                // Mappa la Y globale nella Y locale della fetta
                int local_y = yc1 - y_start;
                
                // Voto 3x3 assicurandoci di non uscire dai bordi della FETTA
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = xc1 + dx;
                        int ny = local_y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < local_height) {
                            // Se abiliti OpenMP qui, serve un #pragma omp atomic
                            local_slice_acc[ny * width + nx]++;
                        }
                    }
                }
            }
        }

        // Estrazione dei massimi LOCALI alla fetta
        // Nessuna comunicazione MPI necessaria in questa fase!
        int max_v = 0;
        for(int i = 0; i < width * local_height; i++) {
            if(local_slice_acc[i] > max_v) max_v = local_slice_acc[i];
        }

        if (max_v >= 3) {
            float threshold_n = max_v * threshold;
            for (int y = 0; y < local_height; y++) {
                for (int x = 0; x < width; x++) {
                    if (local_slice_acc[y * width + x] >= threshold_n) {
                        // NMS locale alla fetta
                        if (NMS_max_circles(local_slice_acc, x, y, width, local_height)) {
                            // Controllo capacità array
                            if (local_count >= capacity) {
                                capacity *= 2;
                                Circle* temp = realloc(local_circles, capacity * sizeof(Circle));
                                if (temp) local_circles = temp;
                                else { MPI_Abort(comm, 1); }
                            }
                            // Salva il cerchio ricordandoti di ritradurre la Y locale in globale
                            local_circles[local_count++] = (Circle){x, y + y_start, r, local_slice_acc[y * width + x]};
                        }
                    }    
                }
            }
        }
    }
    
    // Pulizia strutture di appoggio
    free(local_slice_acc);
    free(cos_theta);
    free(sin_theta);

    // =========================================================================
    // FASE DI GATHER: Raccogliamo tutti i cerchi trovati localmente nel Rank 0
    // =========================================================================
    int *ga_counts = NULL; 
    int *ga_displs = NULL;
    if (rank == 0) {
        ga_counts = malloc(size * sizeof(int));
        ga_displs = malloc(size * sizeof(int));
    }

    // 1. Il Rank 0 scopre quanti cerchi ha trovato ogni nodo
    MPI_Gather(&local_count, 1, MPI_INT, ga_counts, 1, MPI_INT, 0, comm);

    Circle *final_result = NULL;
    int total_circles = 0;

    // 2. Il Rank 0 prepara la memoria per ricevere tutti i cerchi
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            ga_displs[i] = total_circles;
            total_circles += ga_counts[i];
        }
        final_result = malloc((total_circles > 0 ? total_circles : 1) * sizeof(Circle));
    }

    // 3. Trasferimento effettivo delle struct Circle al Rank 0
    MPI_Gatherv(local_circles, local_count, mpi_circle_type,
                final_result, ga_counts, ga_displs, mpi_circle_type, 0, comm);
    
    // 4. Filtraggio finale e pulizia
    if (rank == 0) {
        filter_by_statistics(final_result, total_circles, threshold, out_count);
    } else {
        *out_count = 0;
    }

    free(local_circles);
    MPI_Type_free(&mpi_circle_type);
    if (rank == 0) { free(ga_counts); free(ga_displs); }

    return final_result;
}


Circle* CHT_Domain_Hybrid_Optimized(int* x_coords, int* y_coords, int num_edges, 
                                             int width, int height, 
                                             int r_min, int r_max, float threshold, float* theta_coords,
                                             MPI_Comm comm, int* out_count) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MPI_Datatype mpi_circle_type;
    MPI_Type_contiguous(sizeof(Circle), MPI_BYTE, &mpi_circle_type);
    MPI_Type_commit(&mpi_circle_type);

    // 1. SPATIAL SPLIT (Sull'Accumulatore)
    int slice_h = height / size;
    int y_start = rank * slice_h;
    int y_end = (rank == size - 1) ? height : y_start + slice_h;
    int local_height = y_end - y_start;

    // Accumulatore locale della fetta (Condiviso tra i thread OpenMP del nodo)
    int *local_slice_acc = calloc(width * local_height, sizeof(int));
    
    int capacity = 100;
    Circle* local_circles = malloc(capacity * sizeof(Circle));
    int local_count = 0;

    float* cos_theta = malloc(num_edges * sizeof(float));
    float* sin_theta = malloc(num_edges * sizeof(float));
    
    // OpenMP: Parallelizzazione del pre-calcolo trigonometrico
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < num_edges; e++) {
        cos_theta[e] = cosf(theta_coords[e]);
        sin_theta[e] = sinf(theta_coords[e]);
    }

    // Il ciclo sui raggi rimane seriale per il singolo nodo
    for (int r = r_min; r <= r_max; r++) {
        memset(local_slice_acc, 0, width * local_height * sizeof(int));

        // 2. OPENMP: Parallelizzazione del voto sui pixel (Domain Decomposition)
        // Usiamo schedule(guided) o dynamic perché molti thread "salteranno" l'iterazione 
        // se il centro non cade nella loro fetta, creando sbilanciamento del carico.
        #pragma omp parallel for
        for (int e = 0; e < num_edges; e++) {
            int x = x_coords[e];
            int y = y_coords[e];
            
            int xc1 = (int)(x - r * cos_theta[e] + 0.5f);
            int yc1 = (int)(y - r * sin_theta[e] + 0.5f);

            // Selezioniamo solo i voti che cadono nella fetta assegnata a questo Rank MPI
            if (yc1 >= y_start && yc1 < y_end && xc1 >= 0 && xc1 < width) {
                int local_y = yc1 - y_start;
                
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = xc1 + dx;
                        int ny = local_y + dy;
                        
                        if (nx >= 0 && nx < width && ny >= 0 && ny < local_height) {
                            // COLLO DI BOTTIGLIA OPENMP: 
                            // Protezione obbligatoria per evitare Race Conditions sull'accumulatore
                            #pragma omp atomic
                            local_slice_acc[ny * width + nx]++;
                        }
                    }
                }
            }
        }

        // 3. Estrazione dei massimi: Trovare max_v
        int max_v = 0;
        // OpenMP: Reduction per trovare il massimo globale in modo thread-safe
        #pragma omp parallel for reduction(max:max_v)
        for(int i = 0; i < width * local_height; i++) {
            if(local_slice_acc[i] > max_v) max_v = local_slice_acc[i];
        }

        // 4. Estrazione dei cerchi (Fase di NMS)
        if (max_v >= 3) {
            float threshold_n = max_v * threshold;
            
            // Parallelizziamo la scansione per l'estrazione
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int y = 0; y < local_height; y++) {
                for (int x = 0; x < width; x++) {
                    if (local_slice_acc[y * width + x] >= threshold_n) {
                        if (NMS_max_circles(local_slice_acc, x, y, width, local_height)) {
                            // Sezione critica: l'inserimento nell'array dinamico deve essere serializzato
                            #pragma omp critical
                            {
                                if (local_count >= capacity) {
                                    capacity *= 2;
                                    Circle* temp = realloc(local_circles, capacity * sizeof(Circle));
                                    if (temp) local_circles = temp;
                                    else MPI_Abort(comm, 1);
                                }
                                local_circles[local_count++] = (Circle){x, y + y_start, r, local_slice_acc[y * width + x]};
                            }
                        }
                    }    
                }
            }
        }
    }
    
    free(local_slice_acc);
    free(cos_theta);
    free(sin_theta);

    // =========================================================================
    // Sincronizzazione MPI (Gathering finale sul Rank 0)
    // =========================================================================
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
    
    if (rank == 0) {
        filter_by_statistics(final_result, total_circles, threshold, out_count);
    } else {
        *out_count = 0;
    }

    free(local_circles);
    MPI_Type_free(&mpi_circle_type);
    if (rank == 0) { free(ga_counts); free(ga_displs); }

    return final_result;
}