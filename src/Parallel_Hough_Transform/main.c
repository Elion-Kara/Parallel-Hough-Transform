#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "edge_detection.h"  
#include "parallel_standard.h"  
#include "structs.h"  


void save_pgm(const char *filename, unsigned char *data, int width, int height) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    fwrite(data, sizeof(unsigned char), width * height, f);
    fclose(f);
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int width = 0;
    int height = 0;
    int threshold_hough = 0;
    unsigned char* edge_data = NULL; // Valido solo su Rank 0
    unsigned char* img_data = NULL;  // Per pulizia finale

 
    if (rank == 0) {

        char* image_path = (argc > 1) ? argv[1] : "data/test.jpg";
        int threshold_edge = (argc > 2) ? atoi(argv[2]) : 50;

        // image preprocessing (edge detection)
        int ch;
        img_data = stbi_load(image_path, &width, &height, &ch, 1);
        if (!img_data) {
            fprintf(stderr, "CRITICAL ERROR: Unable to open %s\n", image_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // dynamic threshold if not in input
        if (argc > 3) {
            threshold_hough = atoi(argv[3]);
        } else {
            int min_side = (width < height) ? width : height;
            threshold_hough = (int)(min_side * 0.15); 
        }

        //exevute canny (serial) 
        edge_data = canny_pipeline(img_data, width, height, threshold_edge);
        save_pgm("results/debug_edges.pgm", edge_data, width, height);
        
        // TODO: look into scatterv
        if (height % size != 0) {
            printf("WARNING: L'altezza (%d) non è divisibile per il numero di processi (%d).\n", height, size);
            printf("         L'ultima parte dell'immagine potrebbe essere tagliata o causare errori.\n");
            // In un progetto reale si usa MPI_Scatterv, qui procediamo a rischio/pericolo o abortiamo
        }
    }

    // first broadcast for memory alloc
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold_hough, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // measure time only for the hough transform application
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // apply hough transgorm
    // Rank 0 passa edge_data pieno, gli altri passano NULL (MPI_Scatter gestisce questo).
    Lines* results = HoughLines(edge_data, width, height, threshold_hough, MPI_COMM_WORLD);

    // Sincronizzazione fine timer
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // save results
    if (rank == 0) {
        printf("MPI Execution Time: %f seconds with %d processes.\n", end_time - start_time, size);

        if (results && results->count > 0) {
            printf("-> Found %d lines.\n", results->count);
            
            FILE *f = fopen("results/lines_mpi.txt", "w");
            if (f) {
                for (int i = 0; i < results->count; i++) {
                    fprintf(f, "%d %d\n", results->lines[i].r, results->lines[i].t);
                }
                fclose(f);
            }
            // cleanup
            if(results->lines) free(results->lines);
            free(results);
        } else {
            printf("-> No lines found.\n");
        }

        // cleanup allocated memory
        if(img_data) stbi_image_free(img_data);
        if(edge_data) free(edge_data);
    }

    MPI_Finalize();
    return 0;
}