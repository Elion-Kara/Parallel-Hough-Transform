#ifndef PARALLEL_H
#define PARALLEL_H

#include "../utils/utils.h"

Lines* HoughLines_Parallel_MPI(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm);

Lines* HoughLines_Hybrid_Dense(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm);

Lines* HoughLines_Hybrid_Sparse(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm);

Lines* HoughLines_Hybrid_Optimized(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm);

Lines* HoughLines_Hybrid_Tiled(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm);


Circle* HoughCircles_PureMPI(int* x_coords, int* y_coords, int num_edges, 
                             int width, int height, 
                             int r_min, int r_max, int threshold, 
                             MPI_Comm comm, int* out_count);


Circle* HoughCircles_Hybrid(int* x_coords, int* y_coords, int num_edges, 
                            int width, int height, 
                            int r_min, int r_max, int threshold, 
                            MPI_Comm comm, int* out_count);


#endif
