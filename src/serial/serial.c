#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "serial.h"
#include "../utils/utils.h" // Aggiunto per poter utilizzare NMS_max

#define THETA_STEPS 180


// --- Standard Implementation ---
Lines* HoughLines_Serial_Standard(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    // Siccome è un algoritmo seriale, se non sono il Rank 0 esco subito!
    // Questo previene il Segmentation Fault causato dall'avere edge_img == NULL
    if (rank != 0) {
        return NULL;
    }

    int max_dist = compute_rho(width, height);
    int rho_dim = 2 * max_dist + 1;
    int theta_dim = THETA_STEPS;

    float sin_table[THETA_STEPS], cos_table[THETA_STEPS];
    for(int t = 0; t < theta_dim; t++) {
        float rad = t * 3.1415926535f / (float)THETA_STEPS;
        sin_table[t] = sinf(rad);
        cos_table[t] = cosf(rad);
    }

    int* accumulator = calloc(rho_dim * theta_dim, sizeof(int));
    if (!accumulator) { 
        fprintf(stderr, "CRITICAL ERROR: Unable to allocate accumulator.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            if(edge_img[y * width + x] > 0) {
                for(int t = 0; t < theta_dim; t++) {
                    int r_raw = (int)roundf(x * cos_table[t] + y * sin_table[t]);
                    int r_idx = r_raw + max_dist;
                    if(r_idx >= 0 && r_idx < rho_dim) {
                        accumulator[r_idx * theta_dim + t]++;
                    }
                }
            }
        }
    }

    // --- PREPARAZIONE NMS ---
    // Creiamo un array di puntatori 2D per mappare l'accumulatore 1D
    int **acc2d = malloc(rho_dim * sizeof(int*));
    for (int r = 0; r < rho_dim; r++) {
        acc2d[r] = &accumulator[r * theta_dim];
    }

    Lines* return_lines = malloc(sizeof(Lines));
    int capacity = 100;
    return_lines->lines = malloc(sizeof(Line) * capacity);
    return_lines->count = 0;

    for(int r = 0; r < rho_dim; r++) {
        for(int t = 0; t < theta_dim; t++) {
            // Aggiunta la condizione NMS_max!
            if(accumulator[r * theta_dim + t] > threshold && NMS_max(acc2d, r, t, rho_dim, theta_dim)) {
                
                if(return_lines->count >= capacity) {
                    capacity *= 2;
                    // Allocazione sicura
                    Line* temp = realloc(return_lines->lines, sizeof(Line) * capacity);
                    if (!temp) {
                        fprintf(stderr, "Critical: Realloc failed during line extraction.\n");
                        break;
                    }
                    return_lines->lines = temp;
                }
                return_lines->lines[return_lines->count].r = r - max_dist;
                return_lines->lines[return_lines->count].t = t;
                return_lines->count++;
            }
        }
    }

    free(acc2d); // Libero l'array di puntatori
    free(accumulator); 
    return return_lines;
}

// --- Probabilistic Implementation ---
Lines* HoughLines_Serial_Probabilistic(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    // Barriera di sicurezza per l'algoritmo seriale
    if (rank != 0) {
        return NULL;
    }

    int theta_dim = 180;
    int rho_dim = compute_rho(width, height);
    int total_pixels = width * height;

    float cos_table[180];
    float sin_table[180];
    for (int t = 0; t < theta_dim; t++) {
        float rad = (float)t * 3.1415926535f / 180.0f;
        cos_table[t] = cosf(rad);
        sin_table[t] = sinf(rad);
    }

    int total_edges = 0;
    for (int i = 0; i < total_pixels; i++) {
        if (edge_img[i] > 0) total_edges++;
    }

    if (total_edges == 0) {
        Lines* empty_lines = malloc(sizeof(Lines));
        empty_lines->count = 0;
        empty_lines->lines = malloc(sizeof(Line) * 1);
        return empty_lines;
    }

    int* x_coords = malloc(total_edges * sizeof(int));
    int* y_coords = malloc(total_edges * sizeof(int));
    int edge_idx = 0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (edge_img[y * width + x] > 0) {
                x_coords[edge_idx] = x;
                y_coords[edge_idx] = y;
                edge_idx++;
            }
        }
    }

    int samples = total_pixels / 10;
    if (samples > total_edges) {
        samples = total_edges; 
    }

    int* accumulator = calloc(rho_dim * theta_dim, sizeof(int));

    for (int i = 0; i < samples; i++) {
        int rand_idx = rand() % total_edges;
        float x = (float)x_coords[rand_idx];
        float y = (float)y_coords[rand_idx];

        for (int t = 0; t < theta_dim; t++) {
            int r = (int)(x * cos_table[t] + y * sin_table[t]) + rho_dim / 2;
            if (r >= 0 && r < rho_dim) {
                accumulator[r * theta_dim + t]++;
            }
        }
    }

    free(x_coords);
    free(y_coords);
    
    // --- PREPARAZIONE NMS ---
    // Creiamo l'array 2D anche per la versione probabilistica
    int **acc2d = malloc(rho_dim * sizeof(int*));
    for (int r = 0; r < rho_dim; r++) {
        acc2d[r] = &accumulator[r * theta_dim];
    }

    Lines* return_lines = malloc(sizeof(Lines));
    int capacity = 64; 
    return_lines->lines = malloc(sizeof(Line) * capacity);
    return_lines->count = 0;

    for (int r = 0; r < rho_dim; r++) {
        for (int t = 0; t < theta_dim; t++) {
            // Aggiunta la condizione NMS_max!
            if (accumulator[r * theta_dim + t] > threshold && NMS_max(acc2d, r, t, rho_dim, theta_dim)) {
                
                if (return_lines->count >= capacity) {
                    capacity *= 2;
                    Line* temp = realloc(return_lines->lines, sizeof(Line) * capacity);
                    if (!temp) {
                        fprintf(stderr, "Critical: Realloc failed during line extraction.\n");
                        break;
                    }
                    return_lines->lines = temp;
                }

                return_lines->lines[return_lines->count].r = r - (rho_dim / 2);
                return_lines->lines[return_lines->count].t = t;
                return_lines->count++;
            }
        }
    }

    free(acc2d); // Libero l'array di puntatori
    free(accumulator);
    return return_lines;
}

Circle* HoughCircles_Serial(int* x_coords, int* y_coords, int num_edges, 
                            int width, int height, 
                            int r_min, int r_max, int threshold, 
                            MPI_Comm comm, int* out_count) {
    
    // Ignoriamo la variabile 'comm' nel kernel seriale, ma è necessaria 
    // nella firma per farla corrispondere al puntatore a funzione (typedef)
    (void)comm; 

    int capacity = 1000;
    Circle* found_circles = malloc(capacity * sizeof(Circle));
    if (!found_circles) {
        *out_count = 0;
        return NULL;
    }

    int count = 0;

    // Alloca un UNICO accumulatore 2D per risparmiare memoria (Parameter Space Slicing)
    int *acc2D = malloc(width * height * sizeof(int));
    if (!acc2D) {
        free(found_circles);
        *out_count = 0;
        return NULL;
    }

    // Iteriamo sui raggi uno alla volta
    for (int r = r_min; r <= r_max; r++) {
        
        // 1. Reset dell'accumulatore per il raggio corrente
        for (int i = 0; i < width * height; i++) {
            acc2D[i] = 0;
        }

        // 2. Voto di Bresenham per tutti i pixel di bordo
        for (int e = 0; e < num_edges; e++) {
            bresenham_vote(acc2D, width, height, x_coords[e], y_coords[e], r);
        }

        int dyn_thresh = (int)(2.0 * 3.14159265 * r * 0.45); 
        if (dyn_thresh < 25) { 
            dyn_thresh = 25; 
        }

        // 3. Estrazione dei massimi
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int votes = acc2D[y * width + x];
                if (votes >= dyn_thresh) {
                    
                    // Se superiamo la capacità, allarghiamo l'array dinamicamente
                    if (count >= capacity) {
                        capacity *= 2;
                        Circle *temp = realloc(found_circles, capacity * sizeof(Circle));
                        if (!temp) {
                            // In caso estremo di fine memoria, ci fermiamo e restituiamo ciò che abbiamo
                            free(acc2D);
                            *out_count = count;
                            return found_circles;
                        }
                        found_circles = temp;
                    }
                    
                    found_circles[count].x = x;
                    found_circles[count].y = y;
                    found_circles[count].r = r;
                    found_circles[count].votes = votes;
                    count++;
                }
            }
        }
    }

    free(acc2D); // Pulizia del foglio 2D temporaneo
    
    *out_count = count;
    return found_circles;
}
