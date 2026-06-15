#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// --- 1. IMAGE LIB ---
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// --- 2. HEADERS ---
// Assicurati di avere gli header file corretti o i prototipi qui
#include "edge_detection.h"   // La tua implementazione di Canny (Seriale)
#include "serial_standard.h"  // Definizioni di struct Line e Lines
#include "hough_mpi.h"  // Definizioni di struct Line e Lines


// Prototipo della funzione MPI (definita in hough_mpi.c)
// Lines* HoughLinesMPI(unsigned char* edge_img, int width, int height, int threshold);

// Helper per salvare PGM (solo per debug)
void save_pgm(const char *filename, unsigned char *data, int width, int height) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    fwrite(data, sizeof(unsigned char), width * height, f);
    fclose(f);
}

int main(int argc, char** argv) {
    // --- MPI INIT ---
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Variabili che devono essere note a TUTTI i processi
    int width = 0;
    int height = 0;
    int threshold_hough = 0;
    unsigned char* edge_data = NULL; // Valido solo su Rank 0
    unsigned char* img_data = NULL;  // Per pulizia finale

    // --- FASE 1: CARICAMENTO E PRE-PROCESSING (Solo Rank 0) ---
    if (rank == 0) {
        // printf("=== HOUGH TRANSFORM PARALLELA (MPI) - Processi: %d ===\n", size);

        char* image_path = (argc > 1) ? argv[1] : "data/test.jpg";
        int threshold_edge = (argc > 2) ? atoi(argv[2]) : 50;

        // Caricamento Immagine
        int ch;
        img_data = stbi_load(image_path, &width, &height, &ch, 1); // Force 1 channel (grey)

        if (!img_data) {
            fprintf(stderr, "CRITICAL ERROR: Unable to open %s\n", image_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Calcolo Threshold Hough se non fornita
        if (argc > 3) {
            threshold_hough = atoi(argv[3]);
        } else {
            int min_side = (width < height) ? width : height;
            threshold_hough = (int)(min_side * 0.15); 
        }

        // Eseguiamo Canny (Edge Detection) in SERIALE
        // Nota: Spesso in questi progetti si parallelizza solo il kernel pesante (Hough)
        // Canny è molto veloce rispetto a Hough.
        edge_data = canny_pipeline(img_data, width, height, threshold_edge);
        
        // Salviamo l'immagine dei bordi per debug
        save_pgm("results/debug_edges.pgm", edge_data, width, height);
        
        // Controllo semplice per la divisibilità (per evitare crash nello Scatter semplice)
        if (height % size != 0) {
            printf("WARNING: L'altezza (%d) non è divisibile per il numero di processi (%d).\n", height, size);
            printf("         L'ultima parte dell'immagine potrebbe essere tagliata o causare errori.\n");
            // In un progetto reale si usa MPI_Scatterv, qui procediamo a rischio/pericolo o abortiamo
        }
    }

    // --- FASE 2: COMUNICAZIONE METADATI ---
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold_hough, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- SETUP BENCHMARK ---
    int N_REPEATS = 30;   // Numero di ripetizioni per la statistica
    double total_time = 0.0;
    double max_time = 0.0; // Per calcolare la varianza servirebbe un array, qui semplifichiamo con la media
    
    Lines* results = NULL;

    // --- FASE 3: LOOP DI MISURAZIONE ---
    for (int i = 0; i < N_REPEATS + 1; i++) {
        
        // Sincronizzazione globale prima di partire
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        // CHIAMATA ALGORITMO
        // Nota: Se la tua funzione alloca memoria interna, ricordati di liberarla alla fine del ciclo!
        results = HoughLinesMPI(edge_data, width, height, threshold_hough);

        // Sincronizzazione globale fine
        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();

        double iter_time = end - start;

        // Gestione Warm-up e accumulo
        if (i == 0) {
            if (rank == 0) printf("[LOG] Warm-up run completed: %f s\n", iter_time);
        } else {
            total_time += iter_time;
        }

        // IMPORTANTE: Pulizia memoria risultati tra un giro e l'altro per evitare memory leak
        // (Tranne all'ultimo giro se vuoi salvare i risultati)
        if (i < N_REPEATS) { 
             if (results) {
                 if(results->lines) free(results->lines);
                 free(results);
             }
        }
    }

    // --- FASE 4: STATISTICHE (Solo Rank 0) ---
    if (rank == 0) {
        double avg_time = total_time / N_REPEATS;
        
        // Output formattato specificamente per essere letto da AWK/Bash
        // Formato: BENCH_DATA: <Size> <AvgTime>
        printf("BENCH_DATA: %d %f\n", size, avg_time);

        // Salvataggio risultati dell'ultima iterazione
        if (results && results->count > 0) {
            // ... codice salvataggio file ...
        }
        
        // Cleanup finale
        if(results && results->lines) free(results->lines);
        if(results) free(results);
        if(img_data) stbi_image_free(img_data);
        if(edge_data) free(edge_data);
    }

    MPI_Finalize();
    return 0;
}