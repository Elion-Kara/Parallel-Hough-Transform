#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "serial_standard.h"


Lines* HoughLines(unsigned char* edge_img, int width, int height, int threshold) {
    
    // 1. Accumulator's dimension
    int max_dist = (int)ceil(sqrt(width * width + height * height));
    int rho_dim = 2 * max_dist + 1; // [-max_dist, max_dist]
    int theta_dim = 180;

    // 2. Trigonometric Look-UP table
    double* sin_table = malloc(theta_dim * sizeof(double));
    double* cos_table = malloc(theta_dim * sizeof(double));
    for(int t = 0; t < theta_dim; t++) {
        double rad = t * 3.141592653589793 / 180.0;
        sin_table[t] = sin(rad);
        cos_table[t] = cos(rad);
    }

    // 3. Flattened array
    // Use calloc to inizialize all to 0 
    // Simulated 2D access: index = r_index * theta_dim + t_index
    int* accumulator = calloc(rho_dim * theta_dim, sizeof(int));
    if (!accumulator) { /* Manage allocation error */ }

    // 4. Voting (Hough Transform)
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            // If the pixel is an edge 
            if((unsigned char)edge_img[y * width + x] > 0) {
                
                for(int t = 0; t < theta_dim; t++) {
                    // Calculate r with lookup table
                    // r_raw in [-max_dist, max_dist]
                    int r_raw = (int)(x * cos_table[t] + y * sin_table[t]);
                    
                    // Shift r for using positive indexes
                    int r_idx = r_raw + max_dist; 

                    // Rapid Bound checking 
                    if(r_idx >= 0 && r_idx < rho_dim) {
                        accumulator[r_idx * theta_dim + t]++;
                    }
                }
            }
        }
    }

    // 5. Line's Extraction
    Lines* return_lines = malloc(sizeof(Lines));
    int capacity = 100; // Start capacity
    return_lines->lines = malloc(sizeof(Line) * capacity);
    return_lines->count = 0;

    for(int r = 0; r < rho_dim; r++) {
        for(int t = 0; t < theta_dim; t++) {
            
            // Access to the flattened array
            if(accumulator[r * theta_dim + t] > threshold) {
                
                if(return_lines->count >= capacity) {
                    capacity *= 2; // Doubles the space needed
                    return_lines->lines = realloc(return_lines->lines, sizeof(Line) * capacity);
                }

                Line detected_line;
                detected_line.r = r - max_dist; 
                detected_line.t = t;
                return_lines->lines[return_lines->count] = detected_line;
                return_lines->count++;
            }
        }
    }

    // Cleanup
    free(sin_table);
    free(cos_table);
    free(accumulator); 

    return return_lines;
}

void cleanupLines(Lines* lines) { 
    if(lines) {
        if(lines->lines) free(lines->lines);
        free(lines);
    }
}
