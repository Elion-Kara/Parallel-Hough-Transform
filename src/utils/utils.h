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


typedef struct {
    int x;
    int y;
    int r;
    int votes;
} Circle;

static inline void cleanupLines(Lines* lines) {
    if (lines) {
        if (lines->lines) free(lines->lines);
        free(lines);
    }
}

static inline int compute_rho(unsigned int width, unsigned int height) {
    return (int)(2.0 * hypot((double)width, (double)height)) + 1;
}

void bresenham_vote(int *acc2D, int width, int height, int xc, int yc, int r);

typedef Lines* (*hough_kernel_t)(unsigned char* edge_img, 
                                 int width, 
                                 int height, 
                                 int threshold, 
                                 MPI_Comm comm);

typedef Circle* (*hough_circle_kernel_t)(int* x_coords, int* y_coords, int num_edges, 
                                         int width, int height, 
                                         int r_min, int r_max, int threshold, 
                                         MPI_Comm comm, int* out_count);


int NMS_max(int **acc, int r, int t, int rho_max, int theta_max);


// BENCHMARK RUNNER
void run_hough_benchmark(const char* kernel_name, 
                         hough_kernel_t kernel_func, 
                         unsigned char* edge_img, 
                         int width, int height, int threshold, 
                         MPI_Comm comm, int num_threads);

void run_circle_benchmark(const char* kernel_name, 
                          hough_circle_kernel_t kernel_func, 
                          int* x_coords, int* y_coords, int num_edges, 
                          int width, int height, 
                          int r_min, int r_max, int threshold, 
                          MPI_Comm comm, int num_threads);
#endif // HOUGH_STRUCTS_H
