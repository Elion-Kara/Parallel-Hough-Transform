#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <sys/stat.h>

#define STB_IMAGE_IMPLEMENTATION
#include "edge_detection/stb_image.h"
#include "edge_detection/edge_detection.h"

#include "utils/utils.h"
#include "serial/serial.h"
#include "parallel/parallel.h"

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0) fprintf(stdout, "Usage: %s <immagine> [num_threads] [edge_thresh]\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    char* image_path = argv[1];
    int num_threads = (argc > 2) ? atoi(argv[2]) : 4;
    int threshold_edge = (argc > 3) ? atoi(argv[3]) : 50;
    
    int width = 0, height = 0, threshold_hough = 0;
    unsigned char* edge_data = NULL;
    unsigned char* img_data = NULL;

    // --- CARICAMENTO DATI (Solo Rank 0) ---
    if (rank == 0) {
        int ch;
        img_data = stbi_load(image_path, &width, &height, &ch, 1);
        if (!img_data) {
            fprintf(stderr, "ERRORE: Impossibile caricare %s\n", image_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        threshold_hough = (int)((width < height ? width : height) * 0.15);
        edge_data = canny_pipeline(img_data, width, height, threshold_edge);
    }

    // --- BROADCAST METADATI ---
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold_hough, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ======================================================================
    // ESECUZIONE DEI KERNEL CON IL RUNNER
    // ======================================================================
    
    run_hough_benchmark("Seriale Standard", HoughLines_Serial_Standard, 
                        edge_data, width, height, threshold_hough, MPI_COMM_WORLD, 1);

    run_hough_benchmark("Seriale Probabilistico", HoughLines_Serial_Probabilistic, 
                        edge_data, width, height, threshold_hough, MPI_COMM_WORLD, 1);

    run_hough_benchmark("Parallelo Puro MPI", HoughLines_Parallel_MPI, 
                        edge_data, width, height, threshold_hough, MPI_COMM_WORLD, 1);

    run_hough_benchmark("Ibrido OMP Dense", HoughLines_Hybrid_Dense, 
                        edge_data, width, height, threshold_hough, MPI_COMM_WORLD, num_threads);

    run_hough_benchmark("Ibrido OMP Sparse", HoughLines_Hybrid_Sparse, 
                        edge_data, width, height, threshold_hough, MPI_COMM_WORLD, num_threads);

    // ======================================================================

    if (rank == 0) {
        if(img_data) stbi_image_free(img_data);
        if(edge_data) free(edge_data);
    }

    MPI_Finalize();
    return 0;
}