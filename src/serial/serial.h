#ifndef SERIAL_HOUGH_H
#define SERIAL_HOUGH_H

#include "../utils/utils.h"

// Standard Serial Implementation (comm aggiunto per hough_kernel_t)
Lines* HoughLines_Serial_Standard(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm);

// Probabilistic Serial Implementation
Lines* HoughLines_Serial_Probabilistic(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm);

Circle* HoughCircles_Serial(int* x_coords, int* y_coords, int num_edges, 
                            int width, int height, 
                            int r_min, int r_max, int threshold, 
                            MPI_Comm comm, int* out_count);

#endif // SERIAL_HOUGH_H
