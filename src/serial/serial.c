#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "serial.h"
#include "../utils/utils.h" // Aggiunto per poter utilizzare NMS_max


Circle* CHT_Serial(int* x_coords, int* y_coords, int num_edges, 
                                 int width, int height, 
                                 int r_min, int r_max, float threshold, float* theta_coords,
                                 MPI_Comm comm, int* out_count) {
    (void)comm;

    int capacity = 100;
    Circle* local_circles = malloc(capacity * sizeof(Circle));
    int count = 0;
    int *acc2D = calloc(width * height, sizeof(int));

    // Pre-calcolo di seno e coseno per ogni edge.
    float* cos_theta = malloc(num_edges * sizeof(float));
    float* sin_theta = malloc(num_edges * sizeof(float));
    for (int e = 0; e < num_edges; e++) {
        cos_theta[e] = cosf(theta_coords[e]);
        sin_theta[e] = sinf(theta_coords[e]);
    }

    for (int r = r_min; r < r_max; r++) {
        // int r = r_min + rank + i * size;
        memset(acc2D, 0, width * height * sizeof(int));
        
        for (int e = 0; e < num_edges; e++) {
            int x = x_coords[e];
            int y = y_coords[e];
            
            // Aggiungiamo 0.5f per l'arrotondamento all'intero più vicino
            int xc1 = (int)(x - r * cos_theta[e] + 0.5f);
            int yc1 = (int)(y - r * sin_theta[e] + 0.5f);
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = xc1 + dx;
                    int ny = yc1 + dy;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        acc2D[ny * width + nx]++;
                    }
                }
            }
            
        }
    
        int max_v = 0;
        for(int i=0; i<width*height; i++) if(acc2D[i] > max_v) max_v = acc2D[i];

        int min_absolute_votes = 3;
        if (max_v < min_absolute_votes) {
            continue;
        }

        float threshold_n = max_v * threshold;

        // Estrazione dei massimi
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (acc2D[y * width + x] >= threshold_n) {
                    if (NMS_max_circles(acc2D, x, y, width, height)) {
                        if (count >= capacity) {
                            capacity *= 2;
                            Circle* temp = realloc(local_circles, capacity * sizeof(Circle));
                            if (temp) local_circles = temp;
                            else { MPI_Abort(comm, 1); }
                        }
                        local_circles[count++] = (Circle){x, y, r, acc2D[y * width + x]};
                    }
                }    
            }
        }
    }
    
    free(acc2D);
    free(cos_theta);
    free(sin_theta);

    int total_circles = count;
    filter_by_statistics(local_circles, total_circles, threshold, out_count);

    return local_circles;
}
