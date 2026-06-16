#include <stdio.h>
#include "utils.h"


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

    // Sincronizzazione pre-lancio
    MPI_Barrier(comm);
    double start_time = MPI_Wtime();

    // ESECUZIONE DEL KERNEL PASSATO COME PARAMETRO
    Lines* results = kernel_func(edge_img, width, height, threshold, comm);

    // Sincronizzazione post-lancio
    MPI_Barrier(comm);
    double end_time = MPI_Wtime();

    // Output e validazione (solo su Rank 0)
    if (rank == 0) {
        double elapsed = end_time - start_time;
        int lines_found = (results != NULL) ? results->count : 0;
        
        fprintf(stdout, "[Benchmark] %-20s | Tempo: %.5f s | Linee: %d | MPI: %d | OMP: %d\n", 
               kernel_name, elapsed, lines_found, size, num_threads);

        // Pulizia memoria
        cleanupLines(results);
    }
}