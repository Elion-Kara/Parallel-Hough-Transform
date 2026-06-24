#include <stdio.h>
#include "utils.h"
#include <omp.h>
#include <mpi.h>
#define WARMUP 4
#define NITER  10


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
