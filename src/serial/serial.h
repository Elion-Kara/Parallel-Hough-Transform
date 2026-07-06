#ifndef SERIAL_HOUGH_H
#define SERIAL_HOUGH_H

#include "../utils/utils.h"

Circle* CHT_Serial(int* x_coords, int* y_coords, int num_edges, 
                int width, int height, 
                int r_min, int r_max, float threshold, float* theta_coords,
                MPI_Comm comm, int* out_count);                            

#endif // SERIAL_HOUGH_H
