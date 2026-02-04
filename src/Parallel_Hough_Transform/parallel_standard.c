#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#include "parallel_standard.h"

/* --- Parameters --- */
#define THETA_STEPS 180
#define R_RADIUS 2
#define T_RADIUS 2

/* --- Non-Maximum Suppression --- */
static bool NMS_max(int **acc, int r, int t, int rho, int theta)
{
    int center = acc[r][t];

    for (int dr = -R_RADIUS; dr <= R_RADIUS; dr++) {
        for (int dt = -T_RADIUS; dt <= T_RADIUS; dt++) {
            if (dr == 0 && dt == 0)
                continue;

            int rr = r + dr;
            int tt = t + dt;

            if (rr >= 0 && rr < rho && tt >= 0 && tt < theta) {
                if (acc[rr][tt] > center)
                    return false;
            }
        }
    }
    return true;
}

/* --- Hybrid MPI + OpenMP Hough Transform --- */
Lines* HoughLines(unsigned char* edge_img,
                          unsigned int width,
                          unsigned int height,
                          unsigned int threshold,
                          MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int rho = 2 * sqrt(width * width + height * height) + 1;
    int theta = THETA_STEPS;

    /* --- MPI row decomposition --- */
    int rows_per_rank = height / size;
    int y_start = rank * rows_per_rank;
    int y_end   = (rank == size - 1) ? height : y_start + rows_per_rank;

    /* --- Local accumulator --- */
    int *local_acc = calloc(rho * theta, sizeof(int));

    /* --- Precompute sin/cos --- */
    double cos_table[THETA_STEPS];
    double sin_table[THETA_STEPS];
    for (int t = 0; t < theta; t++) {
        double rad = t * 3.141592/ 180.0;
        cos_table[t] = cos(rad);
        sin_table[t] = sin(rad);
    }

    /* --- OpenMP parallel region --- */
    #pragma omp parallel
    {
        int *thread_acc = calloc(rho * theta, sizeof(int));

        #pragma omp for collapse(2) nowait
        for (int y = y_start; y < y_end; y++) {
            for (int x = 0; x < (int)width; x++) {
                if (edge_img[y * width + x] > 0) {
                    for (int t = 0; t < theta; t++) {
                        int r = (int)(x * cos_table[t]
                                    + y * sin_table[t])
                                    + rho / 2;

                        if (r >= 0 && r < rho) {
                            thread_acc[r * theta + t]++;
                        }
                    }
                }
            }
        }

        /* --- Reduce thread accumulators --- */
        #pragma omp critical
        {
            for (int i = 0; i < rho * theta; i++)
                local_acc[i] += thread_acc[i];
        }

        free(thread_acc);
    }

    /* --- MPI reduce all ranks --- */
    int *global_acc = NULL;
    if (rank == 0)
        global_acc = calloc(rho * theta, sizeof(int));

    MPI_Reduce(local_acc, global_acc, rho * theta, MPI_INT, MPI_SUM, 0, comm);

    free(local_acc);

    /* --- Only rank 0 extracts lines --- */
    if (rank != 0)
        return NULL;

    /* --- Convert to 2D for NMS --- */
    int **acc2d = malloc(rho * sizeof(int*));
    for (int r = 0; r < rho; r++)
        acc2d[r] = &global_acc[r * theta];

    Lines *result = malloc(sizeof(Lines));
    result->count = 0;
    int capacity = 50;
    result->lines = malloc(capacity * sizeof(Line));

    for (int r = 0; r < rho; r++) {
        for (int t = 0; t < theta; t++) {
            if (acc2d[r][t] > threshold &&
                NMS_max(acc2d, r, t, rho, theta)) {

                if (result->count >= capacity) {
                    capacity += 50;
                    result->lines = realloc(result->lines,
                                             capacity * sizeof(Line));
                }

                result->lines[result->count].r = r - rho / 2;
                result->lines[result->count].t = t;
                result->count++;
            }
        }
    }

    free(global_acc);
    free(acc2d);
    return result;
}

/* --- Cleanup --- */
void cleanupLines(Lines* lines)
{
    if (!lines) return;
    free(lines->lines);
    free(lines);
}
