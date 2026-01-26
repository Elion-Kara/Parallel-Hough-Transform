#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
     unsigned char* data; // pixel (0-255)
         int width;
         int height;
     } Image;

unsigned char* canny_pipeline(unsigned char* input_img, int width, int height, int threshold);

void gaussian_blur_5x5(const unsigned char* input, unsigned char* output, int width, int height);

void sobel_filters(const unsigned char* input, unsigned char* magnitude, float* theta, int width, int height);

void non_max_suppression(const unsigned char* magnitude, const float* theta, unsigned char* output, int width, int height, int threshold);
#endif
