#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>

#include <mpi.h>
#include <omp.h>

#include "parallel_standard.h"

/* --- Parameters --- */
#define THETA_STEPS 180
#define R_RADIUS 2
#define T_RADIUS 2

/* Small helper: safe hypot for unsigned ints */
static inline int compute_rho(unsigned int width, unsigned int height)
{
    /* rho range is [-D, +D], where D is image diagonal length */
    const double D = hypot((double)width, (double)height);
    return (int)(2.0 * D) + 1;
}

/* --- Non-Maximum Suppression --- */
static bool NMS_max(int **acc, int r, int t, int rho, int theta)
{
    const int center = acc[r][t];

    for (int dr = -R_RADIUS; dr <= R_RADIUS; dr++) {
        for (int dt = -T_RADIUS; dt <= T_RADIUS; dt++) {
            if (dr == 0 && dt == 0)
                continue;

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
 * Hybrid MPI + OpenMP Hough Transform
 *
 * Key design points:
 *  - MPI decomposes the image by rows using Scatterv (handles remainder rows).
 *  - Each MPI rank builds a local accumulator from its slice.
 *  - Within a rank, OpenMP uses thread-private accumulators to avoid contention,
 *    followed by a parallel reduction into the rank-local accumulator.
 *  - MPI_Reduce sums rank-local accumulators on rank 0.
 *  - Rank 0 performs NMS + thresholding and returns the detected lines.
 */
Lines* HoughLines(unsigned char* edge_img,
                  unsigned int width,
                  unsigned int height,
                  unsigned int threshold,
                  MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int theta = THETA_STEPS;
    const int rho   = compute_rho(width, height);

    /* --- MPI row decomposition (Scatterv to handle non-divisible heights) --- */
    const int base = (size > 0) ? ((int)height / size) : 0;
    const int rem  = (size > 0) ? ((int)height % size) : 0;

    const int local_rows = base + ((rank < rem) ? 1 : 0);
    const int y_start    = rank * base + ((rank < rem) ? rank : rem);

    /* counts/displacements are in pixels (unsigned char elements) */
    int *counts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        counts = (int*)malloc((size_t)size * sizeof(int));
        displs = (int*)malloc((size_t)size * sizeof(int));
        if (!counts || !displs) {
            fprintf(stderr, "CRITICAL ERROR: Unable to allocate Scatterv metadata.\n");
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

    unsigned char *local_edge = NULL;
    const int local_elems = local_rows * (int)width;
    /* Some MPI implementations are stricter about non-NULL receive buffers even when recvcount==0. */
    const size_t local_bytes = (local_elems > 0) ? (size_t)local_elems : 1u;
    local_edge = (unsigned char*)malloc(local_bytes * sizeof(unsigned char));
    if (!local_edge) {
        fprintf(stderr, "CRITICAL ERROR: Rank %d unable to allocate local edge buffer.\n", rank);
        MPI_Abort(comm, 1);
    }

    MPI_Scatterv(edge_img, counts, displs, MPI_UNSIGNED_CHAR,
                 local_edge, local_elems, MPI_UNSIGNED_CHAR,
                 0, comm);

    if (rank == 0) {
        free(counts);
        free(displs);
    }

    /* --- Precompute sin/cos --- */
    double cos_table[THETA_STEPS];
    double sin_table[THETA_STEPS];
    const double pi = 3.14159265358979323846;
    for (int t = 0; t < theta; t++) {
        const double rad = (double)t * pi / 180.0;
        cos_table[t] = cos(rad);
        sin_table[t] = sin(rad);
    }

    /* --- Rank-local accumulator --- */
    const size_t acc_len = (size_t)rho * (size_t)theta;
    if (acc_len > (size_t)INT_MAX) {
        /* MPI counts are int; keep failure explicit instead of silently truncating */
        fprintf(stderr, "CRITICAL ERROR: accumulator too large for MPI counts (%zu elements).\n", acc_len);
        MPI_Abort(comm, 1);
    }

    int *local_acc = (int*)malloc(acc_len * sizeof(int));
    if (!local_acc) {
        fprintf(stderr, "CRITICAL ERROR: Rank %d unable to allocate local accumulator.\n", rank);
        MPI_Abort(comm, 1);
    }

    /* --- OpenMP: thread-private accumulators + parallel reduction --- */
    int **thread_accs = NULL;
    int nthreads = 0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            nthreads = omp_get_num_threads();
            thread_accs = (int**)malloc((size_t)nthreads * sizeof(int*));
            if (!thread_accs) {
                fprintf(stderr, "CRITICAL ERROR: Rank %d unable to allocate thread accumulator list.\n", rank);
                MPI_Abort(comm, 1);
            }
        }

        const int tid = omp_get_thread_num();
        thread_accs[tid] = (int*)calloc(acc_len, sizeof(int));
        if (!thread_accs[tid]) {
            fprintf(stderr, "CRITICAL ERROR: Rank %d thread %d unable to allocate thread accumulator.\n", rank, tid);
            MPI_Abort(comm, 1);
        }
        int *myacc = thread_accs[tid];

        #pragma omp for schedule(static)
        for (int ly = 0; ly < local_rows; ly++) {
            const int y = y_start + ly;
            const unsigned char *row = local_edge + (size_t)ly * (size_t)width;
            for (int x = 0; x < (int)width; x++) {
                if (row[x] > 0) {
                    for (int t = 0; t < theta; t++) {
                        /* r = x cos(t) + y sin(t) shifted by rho/2 */
                        const int r = (int)lrint((double)x * cos_table[t] + (double)y * sin_table[t]) + rho / 2;
                        if ((unsigned)r < (unsigned)rho) {
                            myacc[(size_t)r * (size_t)theta + (size_t)t]++;
                        }
                    }
                }
            }
        }

        /* ensure all votes are complete before reduction */
        #pragma omp barrier

        /* Parallel reduction: each i is reduced across all threads */
        #pragma omp for schedule(static)
        for (size_t i = 0; i < acc_len; i++) {
            int sum = 0;
            for (int k = 0; k < nthreads; k++)
                sum += thread_accs[k][i];
            local_acc[i] = sum;
        }

        #pragma omp barrier
        free(myacc);

        #pragma omp barrier
        #pragma omp single
        {
            free(thread_accs);
            thread_accs = NULL;
        }
    }

    free(local_edge);
    local_edge = NULL;

    /* --- MPI reduce all ranks --- */
    int *global_acc = NULL;
    if (rank == 0) {
        global_acc = (int*)calloc(acc_len, sizeof(int));
        if (!global_acc) {
            fprintf(stderr, "CRITICAL ERROR: Rank 0 unable to allocate global accumulator.\n");
            MPI_Abort(comm, 1);
        }
    }

    MPI_Reduce(local_acc, global_acc, (int)acc_len, MPI_INT, MPI_SUM, 0, comm);
    free(local_acc);

    /* --- Only rank 0 extracts lines --- */
    if (rank != 0) {
        return NULL;
    }

    /* --- Convert to 2D for NMS (no extra copy) --- */
    int **acc2d = (int**)malloc((size_t)rho * sizeof(int*));
    if (!acc2d) {
        fprintf(stderr, "CRITICAL ERROR: Rank 0 unable to allocate NMS view.\n");
        MPI_Abort(comm, 1);
    }
    for (int r = 0; r < rho; r++)
        acc2d[r] = &global_acc[(size_t)r * (size_t)theta];

    Lines *result = (Lines*)malloc(sizeof(Lines));
    if (!result) {
        fprintf(stderr, "CRITICAL ERROR: Rank 0 unable to allocate Lines result.\n");
        MPI_Abort(comm, 1);
    }
    result->count = 0;

    int capacity = 64;
    result->lines = (Line*)malloc((size_t)capacity * sizeof(Line));
    if (!result->lines) {
        fprintf(stderr, "CRITICAL ERROR: Rank 0 unable to allocate Lines buffer.\n");
        MPI_Abort(comm, 1);
    }

    for (int r = 0; r < rho; r++) {
        for (int t = 0; t < theta; t++) {
            if (acc2d[r][t] > (int)threshold &&
                NMS_max(acc2d, r, t, rho, theta)) {

                if (result->count >= capacity) {
                    capacity *= 2;
                    Line *tmp = (Line*)realloc(result->lines, (size_t)capacity * sizeof(Line));
                    if (!tmp) {
                        fprintf(stderr, "CRITICAL ERROR: Rank 0 realloc failed while storing lines.\n");
                        MPI_Abort(comm, 1);
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

/* --- Cleanup --- */
void cleanupLines(Lines* lines)
{
    if (!lines) return;
    free(lines->lines);
    free(lines);
}
