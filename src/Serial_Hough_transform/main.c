#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//image processing lib
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


#include "edge_detection.h"
#include "serial_standard.h"

//function to save pgm
void save_pgm(const char *filename, unsigned char *data, int width, int height) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("ERROR: Can't crate file %s (does results/ exist?)\n", filename);
        return;
    }
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    fwrite(data, sizeof(unsigned char), width * height, f);
    fclose(f);
}


int main(int argc, char** argv) {

    //inizialization of MPI
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        
        //IO in rank 0
        char* image_path = (argc > 1) ? argv[1] : "data/test.jpg";
        int threshold_edge = (argc > 2) ? atoi(argv[2]) : 50;
        
        int w, h, ch;
        unsigned char* img_data = stbi_load(image_path, &w, &h, &ch, 1);

        if (!img_data) {
            fprintf(stderr, "CRITICAL ERROR: Unable to open %s\n", image_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        


        //dynamic threshold (10% of max between width and height of image)
        int hough_threshold = 0.1 * (w > h) ? w : h;

        
        //edge detection
        unsigned char* edge_data = canny_pipeline(img_data, w, h, threshold_edge); 


        //start time measure before hough transform
        double start = MPI_Wtime();

        //apply hough transform
        Lines* results = HoughLines(edge_data, w, h, hough_threshold);

    

        //save result
        save_pgm("results/debug_edges.pgm", edge_data, w, h);
        if (results != NULL && results->count > 0) {
            // printf("-> SUCCESS: Found %d lines.\n", results->count);

            FILE *f = fopen("results/lines.txt", "w");
            if (f) {
                for (int i = 0; i < results->count; i++) {
                    fprintf(f, "%d %d\n", results->lines[i].r, results->lines[i].t);
                }
                fclose(f);
                // printf("-> File 'results/lines.txt' saved.\n");
            }

            //dealloc of dynamic array
            cleanupLines(results);
        } else {
            printf("-> NO LINES FOUND.\n");
            printf("   Hint: Check results/debug_edges.pgm\n");
        }

        //dealloc img
        stbi_image_free(img_data);
        free(edge_data);
    }


    //end MPI 
    MPI_Finalize();
    return 0;
}