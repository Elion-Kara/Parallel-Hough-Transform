#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// --- 1. IMAGE LIB ---
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// --- 2. HEADERS ---
#include "edge_detection.h"
#include "serial_standard.h"

// --- HELPER FUNCION: Saves the image in PGM ---
void save_pgm(const char *filename, unsigned char *data, int width, int height) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("ERROR: Impossible create the file %s (results/ exists?)\n", filename);
        return;
    }
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    fwrite(data, sizeof(unsigned char), width * height, f);
    fclose(f);
}

// --- MAIN ---
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // printf("=== HOUGH TRANSFORM SERIALE ===\n");

        // --- A. ARGUMENTS MANAGEMENT   ---
        char* image_path = (argc > 1) ? argv[1] : "data/test.jpg";
        int threshold_edge = (argc > 2) ? atoi(argv[2]) : 50;
        
        // --- B. LOADING IMAGE ---
        int w, h, ch;
        unsigned char* img_data = stbi_load(image_path, &w, &h, &ch, 1);

        if (!img_data) {
            fprintf(stderr, "CRITICAL ERROR: Unable to open %s\n", image_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // printf("-> Immagine caricata: %dx%d\n", w, h);

        // --- CALCULATING HOUGH'S  THRESHOLD ---
        int threshold_hough;
        if (argc > 3) {
            threshold_hough = atoi(argv[3]);
        } else {
            int min_side = (w < h) ? w : h;
            threshold_hough = (int)(min_side * 0.15); 
        }

        // --- C. EDGE DETECTION ---
        double start = MPI_Wtime();
        
        unsigned char* edge_data = canny_pipeline(img_data, w, h, threshold_edge); 

        // --- D. HOUGH TRANSFORM ---
        Lines* results = HoughLines(edge_data, w, h, threshold_hough);

        double end = MPI_Wtime();
        // printf("=== TOTAL TIME: %f seconds ===\n", end - start);

        // --- E. SAVING RESULTS ---
        save_pgm("results/debug_edges.pgm", edge_data, w, h);

        if (results && results->count > 0) {
            // printf("-> SUCCESS: Found %d lines.\n", results->count);

            FILE *f = fopen("results/lines.txt", "w");
            if (f) {
                for (int i = 0; i < results->count; i++) {
                    fprintf(f, "%d %d\n", results->lines[i].r, results->lines[i].t);
                }
                fclose(f);
                // printf("-> File 'results/lines.txt' saved.\n");
            }

            cleanupLines(results);
        } else {
            printf("-> NO LINES FOUND.\n");
            printf("   Hint: Check results/debug_edges.pgm\n");
        }

        stbi_image_free(img_data);
        free(edge_data);
    }

    MPI_Finalize();
    return 0;
}