#ifndef PARALLEL_H
#define PARALLEL_H

#include "../utils/utils.h"

Circle* CHT_Parameter_MPI(int* x_coords, int* y_coords, int num_edges, 
                                 int width, int height, 
                                 int r_min, int r_max, float threshold, float* theta_coords,
                                 MPI_Comm comm, int* out_count);

Circle* CHT_Parameter_MPI_opt(int* x_coords, int* y_coords, int num_edges, 
                                 int width, int height, 
                                 int r_min, int r_max, float threshold, float* theta_coords,
                                 MPI_Comm comm, int* out_count) ;

Circle* CHT_Parameter_Hybrid(int* x_coords, int* y_coords, int num_edges, 
                                 int width, int height, 
                                 int r_min, int r_max, float threshold, float* theta_coords,
                                 MPI_Comm comm, int* out_count);


#endif
