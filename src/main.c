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
        if (rank == 0) fprintf(stdout, "Usage: %s <image> [num_threads] [edge_thresh]\n", argv[0]);
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
    
    //-------- circle's parameters
    int num_edges = 0;

    // SOLO IL RANK 0 ha l'immagine 'edge_data', quindi solo lui conta i bordi
    if (rank == 0) {
        for (int i = 0; i < width * height; i++) {
            if (edge_data[i] > 0) num_edges++;
        }
    }

    // Il Rank 0 comunica a tutti gli altri nodi quanti bordi ci sono in totale
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int* x_coords = malloc((num_edges > 0 ? num_edges : 1) * sizeof(int));
    int* y_coords = malloc((num_edges > 0 ? num_edges : 1) * sizeof(int));

    if (rank == 0){
        int idx = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (edge_data[y * width + x] > 0) {
                    x_coords[idx] = x;
                    y_coords[idx] = y;
                    idx++;
                }
            }
        }
    }
    
    // Magia MPI: Il Rank 0 invia l'array pieno di coordinate a tutti gli altri nodi!
    MPI_Bcast(x_coords, num_edges, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(y_coords, num_edges, MPI_INT, 0, MPI_COMM_WORLD);

    int r_min = 10;
    int r_max = 500;
    // Usa la threshold che hai già (il compilatore suggerisce threshold_edge)
    int threshold_circles = 0;



    // ======================================================================
    // ESECUZIONE DEI KERNEL CON IL RUNNER
    // ======================================================================
    
    // run_hough_benchmark("Seriale Standard", HoughLines_Serial_Standard, 
    //                     edge_data, width, height, threshold_hough, MPI_COMM_WORLD, 1);

    // // run_hough_benchmark("Seriale Probabilistico", HoughLines_Serial_Probabilistic, 
    // //                     edge_data, width, height, threshold_hough, MPI_COMM_WORLD, 1);

    // run_hough_benchmark("Parallelo Puro MPI", HoughLines_Parallel_MPI, 
    //                    edge_data, width, height, threshold_hough, MPI_COMM_WORLD, 1);

    // run_hough_benchmark("Ibrido OMP Dense", HoughLines_Hybrid_Dense, 
    //                    edge_data, width, height, threshold_hough, MPI_COMM_WORLD, num_threads);

    // run_hough_benchmark("Ibrido OMP Sparse", HoughLines_Hybrid_Sparse, 
    //                     edge_data, width, height, threshold_hough, MPI_COMM_WORLD, num_threads);

    // run_hough_benchmark("Ibrido Optimized", HoughLines_Hybrid_Optimized, 
    //                     edge_data, width, height, threshold_hough, MPI_COMM_WORLD, num_threads);

    // run_hough_benchmark("Ibrido per Tile", HoughLines_Hybrid_Tiled, 
    //                     edge_data, width, height, threshold_hough, MPI_COMM_WORLD, num_threads);

    if (rank == 0) printf("\n--- Benchmark Cerchi (Bresenham + Parameter Space) ---\n");

    run_circle_benchmark("Seriale Cerchi", HoughCircles_Serial, 
                         x_coords, y_coords, num_edges, width, height, r_min, r_max, threshold_circles, MPI_COMM_WORLD, 1);
    fflush(stdout);
    run_circle_benchmark("Puro MPI Cerchi", HoughCircles_PureMPI, 
                         x_coords, y_coords, num_edges, width, height, r_min, r_max, threshold_circles, MPI_COMM_WORLD, 1);
    fflush(stdout);
    run_circle_benchmark("Ibrido OMP Cerchi", HoughCircles_Hybrid, 
                         x_coords, y_coords, num_edges, width, height, r_min, r_max, threshold_circles, MPI_COMM_WORLD, num_threads);
    
    // ======================================================================




    if (rank == 0) {
        if(img_data) stbi_image_free(img_data);
        if(edge_data) free(edge_data);
    }

    free(x_coords);
    free(y_coords);
    MPI_Finalize();
    return 0;
}
