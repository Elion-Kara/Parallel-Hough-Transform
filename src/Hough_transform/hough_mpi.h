#ifndef HOUGH_MPI_H
#define HOUGH_MPI_H

#include "structs.h"

Lines* extract_lines_from_accumulator(int* accumulator, int rho_dim, int theta_dim, int max_dist, int threshold);

Lines* HoughLinesMPI(unsigned char* edge_img, int width, int height, int threshold);

#endif