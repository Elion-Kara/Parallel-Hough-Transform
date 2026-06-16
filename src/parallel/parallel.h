#ifndef PARALLEL_H
#define PARALLEL_H

#include "../utils/utils.h"

Lines* HoughLines_Parallel_MPI(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm);

Lines* HoughLines_Hybrid_Dense(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm);

Lines* HoughLines_Hybrid_Sparse(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm);

#endif