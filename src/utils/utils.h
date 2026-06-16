#ifndef HOUGH_STRUCTS_H
#define HOUGH_STRUCTS_H

#include <stdlib.h>
#include <math.h>
#include <mpi.h>

typedef struct {
    int r;
    int t;
} Line;

typedef struct {
    Line* lines;
    int count;
} Lines;

static inline void cleanupLines(Lines* lines) {
    if (lines) {
        if (lines->lines) free(lines->lines);
        free(lines);
    }
}

static inline int compute_rho(unsigned int width, unsigned int height) {
    return (int)(2.0 * hypot((double)width, (double)height)) + 1;
}

typedef Lines* (*hough_kernel_t)(unsigned char* edge_img, 
                                 int width, 
                                 int height, 
                                 int threshold, 
                                 MPI_Comm comm);

// BENCHMARK RUNNER
void run_hough_benchmark(const char* kernel_name, 
                         hough_kernel_t kernel_func, 
                         unsigned char* edge_img, 
                         int width, int height, int threshold, 
                         MPI_Comm comm, int num_threads);

#endif // HOUGH_STRUCTS_H