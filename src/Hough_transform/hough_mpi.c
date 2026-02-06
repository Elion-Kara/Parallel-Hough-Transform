#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>
#include "serial_standard.h" // Assumo contenga le struct Line e Lines

/**
 * Funzione Helper (eseguita solo dal Rank 0 alla fine)
 * Estrae le linee dall'accumulatore globale (identica alla parte finale del tuo seriale)
 */
Lines* extract_lines_from_accumulator(int* accumulator, int rho_dim, int theta_dim, int max_dist, int threshold) {
    Lines* return_lines = malloc(sizeof(Lines));
    int capacity = 100;
    return_lines->lines = malloc(sizeof(Line) * capacity);
    return_lines->count = 0;

    for(int r = 0; r < rho_dim; r++) {
        for(int t = 0; t < theta_dim; t++) {
            if(accumulator[r * theta_dim + t] > threshold) {
                if(return_lines->count >= capacity) {
                    capacity *= 2;
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
    return return_lines;
}

/**
 * VERSIONE MPI DELLA HOUGH TRANSFORM
 * * Input:
 * - edge_img: Valido solo su Rank 0 (contiene tutta l'immagine)
 * - width, height: Dimensioni globali
 */
Lines* HoughLinesMPI(unsigned char* edge_img, int width, int height, int threshold) {
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- 1. CONFIGURAZIONE GEOMETRICA (Tutti i processi) ---
    // Calcoliamo le dimensioni dell'accumulatore. 
    // Nota: width e height devono essere noti a tutti (passati dal main o via Bcast)
    int max_dist = (int)ceil(sqrt(width * width + height * height));
    int rho_dim = 2 * max_dist + 1;
    int theta_dim = 180;
    long accumulator_size = rho_dim * theta_dim;

    // --- 2. PREPARAZIONE DATI LOCALI (Domain Decomposition) ---
    // Assumiamo per semplicità che height sia divisibile per size.
    // In produzione si usa MPI_Scatterv per gestire resti.
    int local_height = height / size; 
    int local_img_size = width * local_height;

    // Buffer per ricevere la striscia di immagine
    unsigned char* local_img = malloc(local_img_size * sizeof(unsigned char));

    // Scatter: Rank 0 invia pezzi di edge_img a tutti (incluso se stesso)
    // edge_img è significativo solo per root, per gli altri è ignorato da MPI
    MPI_Scatter(edge_img, local_img_size, MPI_UNSIGNED_CHAR,  // send_data, send_count, send_type
                local_img, local_img_size, MPI_UNSIGNED_CHAR, // recv_data, recv_count, recv_type
                0, MPI_COMM_WORLD);                           // rank_root, comm_channel

    // --- 3. TABELLE TRIGONOMETRICHE (Replicazione) ---
    // È più veloce ricalcolarle localmente che inviarle via rete
    double* sin_table = malloc(theta_dim * sizeof(double));
    double* cos_table = malloc(theta_dim * sizeof(double));
    for(int t = 0; t < theta_dim; t++) {
        double rad = t * 3.141592653589793 / 180.0;
        sin_table[t] = sin(rad);
        cos_table[t] = cos(rad);
    }

    // --- 4. ACCUMULATORE LOCALE (Voting) ---
    int* local_accumulator = calloc(accumulator_size, sizeof(int));
    
    // Calcolo dell'offset Y globale
    // Esempio: Se size=4, h=100 -> local_h=25. 
    // Rank 1 lavora sulle righe locali 0-24, ma globalmente sono 25-49.
    int global_y_offset = rank * local_height;

    for(int y = 0; y < local_height; y++) {
        // Coordinata Y reale nell'immagine originale
        int global_y = global_y_offset + y;

        for(int x = 0; x < width; x++) {
            // Accesso all'immagine locale
            if(local_img[y * width + x] > 0) {
                
                for(int t = 0; t < theta_dim; t++) {
                    // Formula standard usando global_y
                    int r_raw = (int)(x * cos_table[t] + global_y * sin_table[t]);
                    int r_idx = r_raw + max_dist;

                    if(r_idx >= 0 && r_idx < rho_dim) {
                        local_accumulator[r_idx * theta_dim + t]++;
                    }
                }
            }
        }
    }

    // --- 5. RIDUZIONE (Map-Reduce) ---
    // Rank 0 alloca l'accumulatore finale per ricevere le somme
    int* global_accumulator = NULL;
    if (rank == 0) {
        global_accumulator = calloc(accumulator_size, sizeof(int));
    }

    // Somma tutti i local_accumulator in global_accumulator
    // MPI_SUM è thread-safe e gestito dalla libreria MPI
    MPI_Reduce(local_accumulator, global_accumulator, accumulator_size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Cleanup risorse locali
    free(local_img);
    free(sin_table);
    free(cos_table);
    free(local_accumulator);

    // --- 6. ESTRAZIONE LINEE (Solo Rank 0) ---
    Lines* final_lines = NULL;
    if (rank == 0) {
        // Ora global_accumulator contiene i voti di tutta l'immagine
        final_lines = extract_lines_from_accumulator(global_accumulator, rho_dim, theta_dim, max_dist, threshold);
        free(global_accumulator);
    }

    // Solo Rank 0 ritorna la struttura, gli altri ritornano NULL
    return final_lines;
}