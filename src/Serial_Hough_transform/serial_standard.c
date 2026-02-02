#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include "serial_standard.h"

// radius for local maxima through nms
#define R_RADIUS 2
#define T_RADIUS 2



//select local max through non-maximum suppression 
//avoids multiple line detection for one true line
bool NMS_max(int **accumulator, int r, int t, int rho, int theta) {

    int center = accumulator[r][t];

    for(int dr = -R_RADIUS; dr <= R_RADIUS; dr++) {
        for(int dt = -T_RADIUS; dt <= T_RADIUS; dt++) {

            if(dr == 0 && dt == 0)
                continue;

            int rr = r + dr;
            int tt = t + dt;

            if(rr >= 0 && rr < rho && tt >= 0 && tt < theta) {
                if(accumulator[rr][tt] > center)
                    return false;
            }
        }
    }
    return true;
}


// TODO: set threshold dinamically, usually 0.1*(min(width,height)) 

Lines* HoughLines(unsigned char* edge_img, unsigned int width, unsigned int height, unsigned int threshold) {
    
    //definition of accumulator dimensions
    int rho = 2 * sqrt(width * width + height * height) +1;
    int theta = 180;

    //creating the accumulator dynamically
    int **accumulator = malloc(rho * sizeof(int*));
    for (int i = 0; i < rho; i++) {

        //allocate for each theta and initialize to 0
        accumulator[i] = calloc(theta, sizeof(int));
    }

    //application of the Hough Transform
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {

            //for each pixel that belongs to an edge 
            if(edge_img[y * width + x] > 0) {
                
                //compute for every possible theta up to 180 degs
                for(int t = 0; t < theta; t++) {
                    
                    //convert theta to radiants
                    double theta_to_rad = t * 3.141592/ 180.0;

                    //cast to int might not be essential
                    int r = (int)(x * cos(theta_to_rad) + y * sin(theta_to_rad)) + (rho / 2);
                    if(r >= 0 && r < rho) {
                        //if r belongs in the range add to accumulator
                        accumulator[r][t]++;

                    }

                }

            }

        }
    }

    //setup variables to store detected lines
    Lines* return_lines = malloc(sizeof(Lines));
    return_lines->lines = malloc(sizeof(Line) * 50);

    //setup for dynamic array size to optimize for time
    //could be more memory efficient but we prioritize time
    return_lines->count = 0;


    //search for lines
    for(int r = 0; r < rho; r++) {
        for(int t = 0; t < theta; t++) {

            //if the value in the accumulator is above the threshold and is the local max
            if(accumulator[r][t] > threshold && NMS_max(accumulator, r, t, rho, theta)) {

                //check if we need to allocate more space
                if(return_lines->count >= 50) {

                    //allocate space for 50 more lines
                    return_lines->lines = realloc(return_lines->lines, sizeof(Line) * (return_lines->count + 50));
                }


                //store the line
                Line detected_line;
                detected_line.r = r - (rho / 2); //shift back r value
                detected_line.t = t;
                return_lines->lines[return_lines->count] = detected_line;
                return_lines->count++;
                

            }

        }
    }




    //explicit free of allocated memory
    for(int i = 0; i < rho; i++) {
        free(accumulator[i]);
    }
    free(accumulator);

    //reurn lines
    return return_lines;
}

//function to cleanup memory alloc for lines, to be added in main
void cleanupLines(Lines* lines) {
    if(lines != NULL) {
        if(lines->lines != NULL) {
            free(lines->lines);
        }
        free(lines);
    }
}


