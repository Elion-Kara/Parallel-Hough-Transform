#include <stdio.h>
#include "utils.h"
#include <omp.h>
#include <mpi.h>
#define WARMUP 1
#define NITER  2


int NMS_max(int **acc, int r, int t, int rho_max, int theta_max) {
    int val = acc[r][t];
    
    // Imposta la dimensione della finestra di soppressione
    // window = 1 significa una griglia 3x3 (il centro +/- 1 in ogni direzione)
    // window = 2 significa una griglia 5x5 (sopprime linee molto vicine in modo più aggressivo)
    int window = 1; 

    for (int dr = -window; dr <= window; dr++) {
        for (int dt = -window; dt <= window; dt++) {
            // Salta il controllo del punto con se stesso
            if (dr == 0 && dt == 0) continue;

            int nr = r + dr;
            int nt = t + dt;

            // Controllo dei confini dell'accumulatore
            if (nr >= 0 && nr < rho_max && nt >= 0 && nt < theta_max) {
                // Se trovo un vicino con un voto MAGGIORE O UGUALE, non sono il massimo
                if (acc[nr][nt] >= val) {
                    return 0; 
                }
            }
        }
    }
    // Se nessun vicino è più alto, allora sono il picco massimo locale
    return 1;
}

// Zero conversioni float, zero trigonometria. Altissime prestazioni.
void bresenham_vote(int *acc2D, int width, int height, int xc, int yc, int r) {
    int x = 0, y = r;
    int d = 3 - 2 * r;

    while (y >= x) {
        int pts[8][2] = {
            {xc+x, yc+y}, {xc-x, yc+y}, {xc+x, yc-y}, {xc-x, yc-y},
            {xc+y, yc+x}, {xc-y, yc+x}, {xc+y, yc-x}, {xc-y, yc-x}
        };

        for (int i = 0; i < 8; i++) {
            int px = pts[i][0];
            int py = pts[i][1];
            if (px >= 0 && px < width && py >= 0 && py < height) {
                acc2D[py * width + px]++;
            }
        }
        x++;
        if (d > 0) {
            y--;
            d = d + 4 * (x - y) + 10;
        } else {
            d = d + 4 * x + 6;
        }
    }
}



void run_hough_benchmark(const char* kernel_name, 
                         hough_kernel_t kernel_func, 
                         unsigned char* edge_img, 
                         int width, int height, int threshold, 
                         MPI_Comm comm, int num_threads)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Imposta il numero di thread per OpenMP (se il kernel non usa OMP, verrà ignorato)
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    double total_max_elapsed = 0.0;
    Lines* final_results = NULL;
    int final_total_lines = 0;

    for ( int i = -WARMUP; i<NITER; i++){
    
        MPI_Barrier(comm);
        double start_time = MPI_Wtime();

        // ESECUZIONE DEL KERNEL 
        Lines* results = kernel_func(edge_img, width, height, threshold, comm);

        double end_time = MPI_Wtime();
        double local_elapsed = end_time - start_time;
        double global_max_elapsed;

        MPI_Reduce(&local_elapsed, &global_max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

	int local_lines = (results != NULL) ? results->count : 0;
        int global_total_lines = 0;
        MPI_Reduce(&local_lines, &global_total_lines, 1, MPI_INT, MPI_SUM, 0, comm);
	
        // Gestione dei risultati su Rank 0
        if (rank == 0) {
	    if (i >= 0){
                total_max_elapsed += global_max_elapsed;
	    }
            
            if (i == NITER - 1) {
                final_results = results;
		final_total_lines = global_total_lines;
            } else {
                if (results != NULL) cleanupLines(results);
            }
        } else {
	    if (results != NULL) cleanupLines(results);
	}
    }

    // Output e validazione (solo su Rank 0)
    if (rank == 0) {
double avg_max_elapsed = total_max_elapsed / NITER;
        
        fprintf(stdout, "[Benchmark] %-20s | Tempo: %.5f s | Linee: %d | MPI: %d | OMP: %d\n", 
               kernel_name, avg_max_elapsed, final_total_lines, size, num_threads);

        // Pulizia memoria
        if (final_results != NULL) cleanupLines(final_results);
    }
}


void run_circle_benchmark(const char* kernel_name, 
                          hough_circle_kernel_t kernel_func, 
                          int* x_coords, int* y_coords, int num_edges, 
                          int width, int height, 
                          int r_min, int r_max, int threshold, 
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
                                      r_min, r_max, threshold, 
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
        double avg_max_elapsed = total_max_elapsed / NITER;
        fprintf(stdout, "[Benchmark] %-20s | Tempo: %.5f s | Cerchi: %d | MPI: %d | OMP: %d\n", 
               kernel_name, avg_max_elapsed, final_total_circles, size, num_threads);
        if (final_results != NULL) free(final_results);
    }
}
