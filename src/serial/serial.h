#ifndef SERIAL_HOUGH_H
#define SERIAL_HOUGH_H

#include "../utils/utils.h"

// Standard Serial Implementation (comm aggiunto per hough_kernel_t)
Lines* HoughLines_Serial_Standard(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm);

// Probabilistic Serial Implementation
Lines* HoughLines_Serial_Probabilistic(unsigned char* edge_img, int width, int height, int threshold, MPI_Comm comm);

#endif // SERIAL_HOUGH_H