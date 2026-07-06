#ifndef HOUGH_STRUCTS_H
#define HOUGH_STRUCTS_H

#include <stdlib.h>
#include <math.h>
#include <mpi.h>


typedef struct __attribute__((packed)){
    int x;
    int y;
    int r;
    int votes;
} Circle;



typedef Circle* (*hough_circle_kernel_t)(int* x_coords, int* y_coords, int num_edges, 
                                         int width, int height, 
                                         int r_min, int r_max, float threshold, float* theta_coords,
                                         MPI_Comm comm, int* out_count);


int NMS_max(int **acc, int r, int t, int rho_max, int theta_max);

int NMS_max_circles(int *acc, int x, int y, int width, int height);

Circle* filter_duplicate_circles(Circle* input, int in_count, int* out_count);

void generate_bresenham_template(int r, int* off_x, int* off_y, int* count);

void bresenham_vote(int *acc2D, int width, int height, int xc, int yc, int r);

void filter_by_statistics(Circle* all_circles, int total_circles, float _threshold_quality, int* final_count);

void run_circle_benchmark(const char* kernel_name, 
                          hough_circle_kernel_t kernel_func, 
                          int* x_coords, int* y_coords, int num_edges, 
                          int width, int height, 
                          int r_min, int r_max, float threshold, float* theta_coords,
                          MPI_Comm comm, int num_threads);

#endif // HOUGH_STRUCTS_H
