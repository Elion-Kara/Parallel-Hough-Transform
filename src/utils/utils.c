#include <stdio.h>
#include "utils.h"
#include <omp.h>
#include <mpi.h>
#define WARMUP 1
#define NITER  2


int NMS_max_circles(int *acc, int x, int y, int width, int height) {
    int val = acc[y * width + x];
    int window = 8; 

    for (int dy = -window; dy <= window; dy++) {
        for (int dx = -window; dx <= window; dx++) {
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                // Usiamo SOLO > per evitare l'annullamento reciproco in caso di parità perfetta
                if (acc[ny * width + nx] > val) {
                    return 0; 
                }
            }
        }
    }
    return 1;
}

int compare_circles_desc(const void *a, const void *b) {
    Circle *c1 = (Circle *)a;
    Circle *c2 = (Circle *)b;
    return c2->votes - c1->votes; // Ordine decrescente
}

Circle* filter_duplicate_circles(Circle* input, int in_count, int* out_count) {
    if (in_count == 0) {
        *out_count = 0;
        return NULL;
    }

    // 1. Ordiniamo l'array: il "vero" cerchio (quello col picco più alto) sarà sempre il primo!
    qsort(input, in_count, sizeof(Circle), compare_circles_desc);

    Circle* output = malloc(in_count * sizeof(Circle));
    int count = 0;
    
    // Tolleranza del centro: consideriamo "doppione" tutto ciò che cade entro 10 pixel dal centro
    int tol_xy = 20; 

    for (int i = 0; i < in_count; i++) {
        int is_duplicate = 0;
        
        // 2. Confrontiamo con i cerchi "forti" già confermati e salvati in output
        for (int j = 0; j < count; j++) {
            int dx = input[i].x - output[j].x;
            int dy = input[i].y - output[j].y;

            int dr = input[i].r - output[j].r;

            // Se questo cerchio debole ha lo stesso centro di uno forte già salvato, scartalo!
            // Nota: ignoriamo completamente la differenza di raggio. Un centro = un cerchio.
            if (abs(dx) <= tol_xy && abs(dy) <= tol_xy ) {
                is_duplicate = 1;
                break;
            }
        }
        
        // Se non è un doppione di nessuno dei più forti, è un nuovo oggetto indipendente
        if (!is_duplicate) {
            output[count++] = input[i];
        }
    }
    
    *out_count = count;
    return output;
}

void filter_by_statistics(Circle* all_circles, int total_circles, float _threshold_quality, int* final_count) {
    if (total_circles == 0) return;

    // 1. Calcolo statistico (Media e Deviazione Standard sui voti)
    double sum = 0;
    for (int i = 0; i < total_circles; i++) sum += all_circles[i].votes;
    double mean = sum / total_circles;

    double sq_sum = 0;
    for (int i = 0; i < total_circles; i++) sq_sum += pow(all_circles[i].votes - mean, 2);
    double stddev = sqrt(sq_sum / total_circles);
    
    double voti_threshold = mean + (3.0f * stddev); // Soglia basata sull'intensità

    int kept = 0;
    for (int i = 0; i < total_circles; i++) {
        // 2. Filtro di Qualità (Indipendente dalla dimensione)
        float perimetro = 3.0f * M_PI * all_circles[i].r;
        float quality = (float)all_circles[i].votes / perimetro;

        // 3. Filtro Combinato (AND logico):
        // Deve essere un picco intenso (statistica) E un cerchio geometricamente coerente (qualità)
        if (all_circles[i].votes >= voti_threshold) {
            all_circles[kept++] = all_circles[i];
        }
    }
    *final_count = kept;
}





void run_circle_benchmark(const char* kernel_name, 
                          hough_circle_kernel_t kernel_func, 
                          int* x_coords, int* y_coords, int num_edges, 
                          int width, int height, 
                          int r_min, int r_max, float threshold, float* theta_coords,
                          MPI_Comm comm, int num_threads)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    double total_max_elapsed = 0.0;
    Circle* final_results = NULL;
    int final_total_circles = 0;

    for (int i = -WARMUP; i < NITER; i++) {
        MPI_Barrier(comm);
        double start_time = MPI_Wtime();

        int circle_count = 0;
        // Chiamata dinamica al kernel passato come parametro
        Circle* results = kernel_func(x_coords, y_coords, num_edges, 
                                      width, height, 
                                      r_min, r_max, threshold, theta_coords,
                                      comm, &circle_count);

        double end_time = MPI_Wtime();
        double local_elapsed = end_time - start_time;
        double global_max_elapsed;

        MPI_Reduce(&local_elapsed, &global_max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        // A differenza delle linee, i kernel per i cerchi che abbiamo scritto
        // popolano 'circle_count' con il totale globale direttamente sul Rank 0
        if (rank == 0) {
            if (i >= 0) total_max_elapsed += global_max_elapsed;
            
            if (i == NITER - 1) {
                final_results = results;
                final_total_circles = circle_count;
            } else {
                if (results != NULL) free(results);
            }
        } else {
            if (results != NULL) free(results);
        }
    }

    if (rank == 0) {
        // Usa un puntatore temporaneo per il nuovo array filtrato
        Circle* filtered_results = filter_duplicate_circles(final_results, final_total_circles, &final_total_circles);
        
        // Libera il vecchio array contenente i cloni
        free(final_results);
        
        // Aggiorna il puntatore principale
        final_results = filtered_results;

        double avg_max_elapsed = total_max_elapsed / NITER;
        fprintf(stdout, "[Benchmark] %-20s | Tempo: %.5f s | Cerchi: %d | MPI: %d | OMP: %d\n", 
               kernel_name, avg_max_elapsed, final_total_circles, size, num_threads);

        FILE *f = fopen("cerchi_trovati.csv", "w");
        if (f) {
            fprintf(f, "x,y,r\n");
            for (int j = 0; j < final_total_circles; j++) {
                fprintf(f, "%d,%d,%d\n", final_results[j].x, final_results[j].y, final_results[j].r);
            }
            fclose(f);
        }
               
        if (final_results != NULL) free(final_results);
    }
}
