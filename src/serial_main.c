#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "edge_detection.h" // Il tuo nuovo file
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    // ... setup MPI ...

    int w, h, ch;
    // Carica immagine (solo su master o parallelamente)
    unsigned char* img = stbi_load("test.jpg", &w, &h, &ch, 1); // 1 = forza grayscale

    if(img) {
        unsigned char* edges = canny_pipeline(img, w, h);

        free(edges);
        stbi_image_free(img);
    }

    MPI_Finalize();
    return 0;
}
