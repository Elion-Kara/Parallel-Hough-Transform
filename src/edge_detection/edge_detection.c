#include "edge_detection.h"
#include <string.h>
#include <stdio.h>
#include <omp.h>


// Generic Convolution function
// // kernel: 1D array representing matrix KxK
// // k_size: side dimension (e.g., 3, 5, 7...)
void convolve_dynamic(const unsigned char* input, float* output, int width, int height, 
                      const float* kernel, int k_size) {
    
    int radius = k_size / 2;
    
    // Parallelizing (suitable for big kernels)
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            
            float sum = 0.0f;
            
            // Dynamic Kernel Loop
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    
                    // Edge management (Clamping or Skip)
                    int py = y + ky;
                    int px = x + kx;
                    
                    // Bound check (if exceeded, use pixel's edge or 0)
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        float val = (float)input[py * width + px];
                        // Kernel index (shifting from -r..+r to 0..k_size)
                        int k_idx = (ky + radius) * k_size + (kx + radius);
                        
                        sum += val * kernel[k_idx];
                    }
                }
            }
            output[y * width + x] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// 1. GAUSSIAN BLUR (5x5)
// ---------------------------------------------------------------------------
void gaussian_blur_5x5(const unsigned char* input, unsigned char* output, int width, int height) {
    // Gaussian Kernel with sigma=1.4
    int kernel[5][5] = {
        {2, 4, 5, 4, 2},
        {4, 9, 12, 9, 4},
        {5, 12, 15, 12, 5},
        {4, 9, 12, 9, 4},
        {2, 4, 5, 4, 2}
    };
    int kernel_sum = 159;

    // Parallelize the 2 outer loops
    #pragma omp parallel for collapse(2)
    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            int sum = 0;
            
            // 5x5 Convolution
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int pixel_val = input[(y + ky) * width + (x + kx)];
                    sum += pixel_val * kernel[ky + 2][kx + 2];
                }
            }
            output[y * width + x] = (unsigned char)(sum / kernel_sum);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. SOBEL FILTERS
// Calculate Magnitude (edge weight) and Direction (Theta)
// ---------------------------------------------------------------------------
void sobel_filters(const unsigned char* input, unsigned char* magnitude, float* theta, int width, int height) {
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float sumX = 0;
            float sumY = 0;

            // 3x3 Convolution
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int val = input[(y + i) * width + (x + j)];
                    sumX += val * Gx[i + 1][j + 1];
                    sumY += val * Gy[i + 1][j + 1];
                }
            }

            // Calculate Magnitude
            int val = (int)sqrt(sumX * sumX + sumY * sumY);
            if (val > 255) val = 255;
            magnitude[y * width + x] = (unsigned char)val;

            // Calculate Direction
            float angle = atan2(sumY, sumX) * 180.0 / M_PI;
            // Normalize the angle between 0 and 180
            if (angle < 0) angle += 180;
            theta[y * width + x] = angle;
        }
    }
}

// ---------------------------------------------------------------------------
// 3. NON-MAXIMUM SUPPRESSION (NMS)
// Thins edges by keeping only the local peak
// ---------------------------------------------------------------------------
void non_max_suppression(const unsigned char* magnitude, const float* theta, unsigned char* output, int width, int height, int threshold) {

    // 1. Initial cleanup (OpenMP for speed)
    #pragma omp parallel for
    for(int i=0; i<width*height; i++) output[i] = 0;

    // 2. Main loop (Skip 1px borders for safety)
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {

            int idx = y * width + x;
            float angle = theta[idx];
            unsigned char mag = magnitude[idx]; // 'mag' is the VALUE (0-255)

            // Optimization: If it's black, skip immediately
           // if (mag < 10) { 
            //    continue; // output[idx] is already 0
            //}

            unsigned char q = 255;
            unsigned char r = 255;

            // --- DIRECTION LOGIC ---
            // Note: q and r will take the VALUE of neighbors, not the index.
            
            // 0 degrees (Horizontal) -> Neighbors: Right and Left
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                q = magnitude[y * width + (x + 1)];
                r = magnitude[y * width + (x - 1)];
            }
            // 45 degrees (Diagonal /) -> Neighbors: Top-Right and Bottom-Left
            else if (angle >= 22.5 && angle < 67.5) {
                q = magnitude[(y + 1) * width + (x - 1)]; // Bottom-Left
                r = magnitude[(y - 1) * width + (x + 1)]; // Top-Right
            }
            // 90 degrees (Vertical) -> Neighbors: Top and Bottom
            else if (angle >= 67.5 && angle < 112.5) {
                q = magnitude[(y + 1) * width + x];       // Bottom
                r = magnitude[(y - 1) * width + x];       // Top
            }
            // 135 degrees (Diagonal \) -> Neighbors: Top-Left and Bottom-Right
            else {
                q = magnitude[(y - 1) * width + (x - 1)]; // Top-Left
                r = magnitude[(y + 1) * width + (x + 1)]; // Bottom-Right
            }

            // --- COMPARISON AND THRESHOLD ---
            // CORRECTION: Compare values 'mag', 'q', 'r' directly
            if (mag >= q && mag >= r) {
                
                // Dynamic Threshold Application (passed from main)
                if (mag > threshold) {
                    output[idx] = 255; // Edge Confirmed
                } else {
                    output[idx] = 0;   // Too weak
                }

            } else {
                output[idx] = 0; // Not a local maximum (there is a stronger neighbor)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PIPELINE WRAPPER (Memory Management)
// ---------------------------------------------------------------------------
unsigned char* canny_pipeline(unsigned char* input_img, int width, int height, int threshold) {
    size_t img_size = width * height * sizeof(unsigned char);
    size_t float_size = width * height * sizeof(float);

    // Temporary buffer allocation
    unsigned char* blurred = (unsigned char*)malloc(img_size);
    unsigned char* mag = (unsigned char*)malloc(img_size);
    float* theta = (float*)malloc(float_size);
    unsigned char* final_edges = (unsigned char*)calloc(width * height, sizeof(unsigned char)); // Final Output

    if (!blurred || !mag || !theta || !final_edges) {
        printf("Error allocating memory in pipeline!\n");
        exit(1);
    }

    // 1. Gaussian Blur
    // Copy unprocessed edges to avoid black artifacts, or use calloc.
    // For simplicity, we assume the 2px black border is okay.
    gaussian_blur_5x5(input_img, blurred, width, height);

    // 2. Sobel
    sobel_filters(blurred, mag, theta, width, height);

    // 3. NMS
    non_max_suppression(mag, theta, final_edges, width, height, threshold);

    // Cleanup
    free(blurred);
    free(mag);
    free(theta);

    // Returns the image ready for the Hough Transform
    return final_edges; 
}