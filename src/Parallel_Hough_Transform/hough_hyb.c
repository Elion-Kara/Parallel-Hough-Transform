#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>

#include <mpi.h>
#include <omp.h>

// Assicurati che questo header definisca le struct Line e Lines!
#include "hough_mpi.h" 

/* --- Parameters --- */
// Puoi cambiarlo a 720 se hai abbastanza RAM (vedi discussione precedente)
#define THETA_STEPS 180 
#define R_RADIUS 2
#define T_RADIUS 2

/* Helper: Calcolo dimensione diagonale (Rho) */
static inline int compute_rho(unsigned int width, unsigned int height)
{
    const double D = hypot((double)width, (double)height);
    return (int)(2.0 * D) + 1;
}

/* Helper: Non-Maximum Suppression (NMS) locale */
static bool NMS_max(int **acc, int r, int t, int rho, int theta)
{
    const int center = acc[r][t];

    for (int dr = -R_RADIUS; dr <= R_RADIUS; dr++) {
        for (int dt = -T_RADIUS; dt <= T_RADIUS; dt++) {
            if (dr == 0 && dt == 0) continue;

            const int rr = r + dr;
            const int tt = t + dt;

            if (rr >= 0 && rr < rho && tt >= 0 && tt < theta) {
                if (acc[rr][tt] > center)
                    return false;
            }
        }
    }
    return true;
}

/*
 * FUNZIONE PRINCIPALE: HoughLinesMPI (Hybrid OpenMP + MPI)
 * Adattata per ricevere il puntatore time_reduce
 */
Lines* HoughLinesMPI(unsigned char* edge_img,
                     int width,
                     int height,
                     int threshold,
                     double* time_reduce)
{
    // Usiamo sempre il comunicatore globale per semplicità
    MPI_Comm comm = MPI_COMM_WORLD; 

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int theta = THETA_STEPS;
    const int rho   = compute_rho(width, height);

    /* --- 1. Decomposizione MPI (Scatterv) --- */
    // Calcoliamo quante righe toccano a ciascun processo
    const int base = (size > 0) ? ((int)height / size) : 0;
    const int rem  = (size > 0) ? ((int)height % size) : 0;

    const int local_rows = base + ((rank < rem) ? 1 : 0);
    const int y_start    = rank * base + ((rank < rem) ? rank : rem);

    int *counts = NULL;
    int *displs = NULL;

    // Solo il Master prepara la mappa di distribuzione
    if (rank == 0) {
        counts = (int*)malloc((size_t)size * sizeof(int));
        displs = (int*)malloc((size_t)size * sizeof(int));
        
        if (!counts || !displs) {
            fprintf(stderr, "CRITICAL ERROR: Allocazione Scatterv fallita.\n");
            MPI_Abort(comm, 1);
        }

        int disp = 0;
        for (int r = 0; r < size; r++) {
            const int rows_r = base + ((r < rem) ? 1 : 0);
            counts[r] = rows_r * (int)width;
            displs[r] = disp;
            disp += counts[r];
        }
    }

    // Tutti allocano il buffer per la PROPRIA fetta di immagine
    unsigned char *local_edge = NULL;
    const int local_elems = local_rows * (int)width;
    const size_t local_bytes = (local_elems > 0) ? (size_t)local_elems : 1u;
    
    local_edge = (unsigned char*)malloc(local_bytes * sizeof(unsigned char));
    if (!local_edge) {
        fprintf(stderr, "CRITICAL ERROR: Rank %d allocazione buffer locale fallita.\n", rank);
        MPI_Abort(comm, 1);
    }

    // Distribuzione dati (edge_img è valido solo su Rank 0, gli altri ricevono NULL)
    MPI_Scatterv(edge_img, counts, displs, MPI_UNSIGNED_CHAR,
                 local_edge, local_elems, MPI_UNSIGNED_CHAR,
                 0, comm);

    if (rank == 0) {
        free(counts);
        free(displs);
    }

    /* --- 2. Precalcolo Tabelle Seno/Coseno --- */
    double *cos_table = malloc(theta * sizeof(double));
    double *sin_table = malloc(theta * sizeof(double));
    
    const double pi = 3.14159265358979323846;
    for (int t = 0; t < theta; t++) {
        const double rad = (double)t * pi / 180.0;
        cos_table[t] = cos(rad);
        sin_table[t] = sin(rad);
    }

    /* --- 3. Allocazione Accumulatore Locale --- */
    const size_t acc_len = (size_t)rho * (size_t)theta;
    int *local_acc = (int*)malloc(acc_len * sizeof(int));
    // Importante: Inizializzare a zero! (Scatter/Malloc non puliscono la RAM)
    if (local_acc) memset(local_acc, 0, acc_len * sizeof(int)); 
    else {
        fprintf(stderr, "CRITICAL ERROR: Rank %d memoria insufficiente per accumulatore.\n", rank);
        MPI_Abort(comm, 1);
    }

    /* --- 4. Calcolo Hough Ibrido (OpenMP Thread-Private) --- */
    // Questa tecnica evita collisioni atomiche ed è molto veloce
    int **thread_accs = NULL;
    int nthreads = 1; // Default seriale se OpenMP fallisce

    #pragma omp parallel
    {
        #pragma omp single
        {
            nthreads = omp_get_num_threads();
            thread_accs = (int**)malloc((size_t)nthreads * sizeof(int*));
        }

        // Ogni thread ha il suo accumulatore privato
        const int tid = omp_get_thread_num();
        // calloc inizializza a zero
        int *myacc = (int*)calloc(acc_len, sizeof(int)); 
        
        // Salvo il puntatore per la riduzione successiva
        // (Nota: in produzione servirebbe un controllo thread-safe qui, ma omp single aiuta)
        if(thread_accs) thread_accs[tid] = myacc;

        #pragma omp for schedule(static)
        for (int ly = 0; ly < local_rows; ly++) {
            const int y = y_start + ly; // Coordinata Y Globale
            const unsigned char *row = local_edge + (size_t)ly * (size_t)width;
            
            for (int x = 0; x < (int)width; x++) {
                if (row[x] > 0) { // Se è un bordo
                    for (int t = 0; t < theta; t++) {
                        const int r = (int)lrint((double)x * cos_table[t] + (double)y * sin_table[t]) + rho / 2;
                        if (r >= 0 && r < rho) {
                            myacc[(size_t)r * (size_t)theta + (size_t)t]++;
                        }
                    }
                }
            }
        }

        #pragma omp barrier

        // Riduzione OpenMP: Uniamo i thread nel local_acc del processo
        #pragma omp for schedule(static)
        for (size_t i = 0; i < acc_len; i++) {
            int sum = 0;
            if (thread_accs) {
                for (int k = 0; k < nthreads; k++) {
                    if (thread_accs[k]) sum += thread_accs[k][i];
                }
            }
            local_acc[i] = sum;
        }

        if (myacc) free(myacc); // Pulisce memoria thread
    }
    
    // Pulizia strutture OpenMP e buffer immagine
    if(thread_accs) free(thread_accs);
    free(local_edge);
    free(cos_table);
    free(sin_table);


    /* --- 5. MPI REDUCE (Il punto critico per la rete) --- */
    int *global_acc = NULL;
    if (rank == 0) {
        global_acc = (int*)calloc(acc_len, sizeof(int));
        if (!global_acc) {
            fprintf(stderr, "CRITICAL ERROR: Memoria insufficiente per accumulatore globale.\n");
            MPI_Abort(comm, 1);
        }
    }

    // --- MISURAZIONE TEMPO RETE ---
    double t_start_red = MPI_Wtime();
    
    MPI_Reduce(local_acc, global_acc, (int)acc_len, MPI_INT, MPI_SUM, 0, comm);
    
    double t_end_red = MPI_Wtime();
    // -----------------------------

    // Salviamo il tempo nel puntatore fornito dal main
    if (time_reduce != NULL) {
        *time_reduce = t_end_red - t_start_red;
    }

    free(local_acc);


    /* --- 6. Estrazione Linee (Solo Rank 0) --- */
    if (rank != 0) {
        return NULL; // Gli slave hanno finito
    }

    // Creazione vista 2D per facilitare il NMS
    int **acc2d = (int**)malloc((size_t)rho * sizeof(int*));
    for (int r = 0; r < rho; r++)
        acc2d[r] = &global_acc[(size_t)r * (size_t)theta];

    Lines *result = (Lines*)malloc(sizeof(Lines));
    result->count = 0;

    int capacity = 1000; // Partiamo con una capacità ragionevole
    result->lines = (Line*)malloc((size_t)capacity * sizeof(Line));

    for (int r = 0; r < rho; r++) {
        for (int t = 0; t < theta; t++) {
            // Soglia + Non-Maximum Suppression
            if (acc2d[r][t] > (int)threshold && NMS_max(acc2d, r, t, rho, theta)) {

                if (result->count >= capacity) {
                    capacity *= 2;
                    Line *tmp = (Line*)realloc(result->lines, (size_t)capacity * sizeof(Line));
                    if (!tmp) {
                        fprintf(stderr, "ERRORE: Realloc fallita estrazione linee.\n");
                        break; 
                    }
                    result->lines = tmp;
                }

                result->lines[result->count].r = r - rho / 2;
                result->lines[result->count].t = t;
                result->count++;
            }
        }
    }

    free(acc2d);
    free(global_acc);
    return result;
}