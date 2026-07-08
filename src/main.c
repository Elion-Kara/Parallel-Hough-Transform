#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>
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
    float threshold_edge = (argc > 3) ? atof(argv[3]) : 0.85f;
    int code = (argc > 4) ? atoi(argv[4]) : 1;
    bool print_circles = (argc > 5) ? atoi(argv[5]) : 0; 
    
    int width = 0, height = 0;
    unsigned char* edge_data = NULL;
    unsigned char* img_data = NULL;
    float* theta_map = NULL;
    float threshold_hough;

    // --- Load data first only on rank 0 ---
    if (rank == 0) {
        int ch;
        img_data = stbi_load(image_path, &width, &height, &ch, 1);
        if (!img_data) {
            fprintf(stderr, "ERRORE: Impossibile caricare %s\n", image_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        threshold_hough = (int)((width < height ? width : height) * 0.25);
        
        edge_data = malloc(width * height * sizeof(unsigned char));
        theta_map = calloc(width * height, sizeof(float)); 

        unsigned char* mag_map = malloc(width * height * sizeof(unsigned char));
        gaussian_blur_5x5(img_data, edge_data, width, height);

        sobel_filters(edge_data, mag_map, theta_map, width, height);

        for (int i = 0; i < width * height; i++) {
            edge_data[i] = (mag_map[i] > 50) ? 255 : 0;
        }

        free(mag_map); 
    }

    // --- BROADCAST METADATA ---
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold_hough, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // ––– circle's parameters ––––
    int num_edges = 0;

    if (rank == 0) {
        for (int i = 0; i < width * height; i++) {
            if (edge_data[i] > 0) num_edges++;
        }
    }

    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int* x_coords = malloc((num_edges > 0 ? num_edges : 1) * sizeof(int));
    int* y_coords = malloc((num_edges > 0 ? num_edges : 1) * sizeof(int));
    float* theta_coords = malloc((num_edges > 0 ? num_edges : 1) * sizeof(float)); 

    if (rank == 0){
        int idx = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (edge_data[y * width + x] > 0) {
                    x_coords[idx] = x;
                    y_coords[idx] = y;
                    theta_coords[idx] = theta_map[y * width + x] * (M_PI / 180.0f);
                    idx++;
                }
            }
        }
    }
    
    MPI_Bcast(x_coords, num_edges, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(y_coords, num_edges, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(theta_coords, num_edges, MPI_FLOAT, 0, MPI_COMM_WORLD); 

    int r_min = 10;
    int r_max = 500;    

    // –––––––– ESECUZIONE DEI KERNEL  ––––––––
    if (rank == 0) printf("\n––– Benchmark –––\n");
    if (code != 0){
        run_circle_benchmark("Serial", CHT_Serial, 
                    x_coords, y_coords, num_edges, width, height, r_min, r_max,
                    threshold_edge, theta_coords, MPI_COMM_WORLD, 1, print_circles);    
    }
    if (code != 1){
        run_circle_benchmark("Parameter MPI", CHT_Parameter_MPI, 
                            x_coords, y_coords, num_edges, width, height, r_min,
                            r_max, threshold_edge, theta_coords, MPI_COMM_WORLD,
                            num_threads, print_circles);
        run_circle_benchmark("Parameter MPI Opt", CHT_Parameter_MPI_opt, 
                            x_coords, y_coords, num_edges, width, height, r_min,
                            r_max, threshold_edge, theta_coords, MPI_COMM_WORLD,
                            num_threads, print_circles);
        run_circle_benchmark("Parameter Hybrid", CHT_Parameter_Hybrid, 
                            x_coords, y_coords, num_edges, width, height, r_min,
                            r_max, threshold_edge, theta_coords, MPI_COMM_WORLD,
                            num_threads, print_circles);
    }

    if (rank == 0) {
        if(img_data) stbi_image_free(img_data);
        if(edge_data) free(edge_data);
        if(theta_map) free(theta_map);
    }

    free(x_coords);
    free(y_coords);
    MPI_Finalize();
    return 0;
}
