#include <stdlib.h>
#include <math.h>

#include "serial_probabilistic.h"

#define threshold 50

//data structure to store (and easily return) detected lines
typedef struct {
    int r;
    int t;
}Line;

//List of lines
typedef struct {
    Line* lines;
    int count;
}Lines;

//as in opencv implementation we refer to algorithm presented in
//Progressive Probabilistic Hough Transform (Matas et al.)
Lines* HoughProb(char* edge_img, unsigned int width, unsigned int height){

    int samples = width * height / 10;

    //definition of accumulator dimensions
    int rho = 2 * sqrt(width * width + height * height) +1;
    int theta = 180;

    //creating the accumulator dynamically
    int **accumulator = malloc(rho * sizeof(int*));
    for (int i = 0; i < rho; i++) {

        //allocate for each theta and initialize to 0
        accumulator[i] = calloc(theta, sizeof(int));
    }

    //sample random points form edge and update accumulator
    for(int i = 0; i < samples; i++){

        //pick random point in efge image THAT IS NOT ZERO
        int x, y;
        do{
            x = rand() % width;
            y = rand() % height;
        }while(edge_img[y * width + x] == 0);

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

    //from now on same as standard but lower threshold

    //setup variables to store detected lines
    Lines* return_lines = malloc(sizeof(Lines));
    return_lines->lines = malloc(sizeof(Line) * 50);

    //setup for dynamic array size to optimize for time
    //could be more memory efficient but we prioritize time
    return_lines->count = 0;


    //search for lines
    for(int r = 0; r < rho; r++) {
        for(int t = 0; t < theta; t++) {

            //if the value in the accumulator is above the threshold
            if(accumulator[r][t] > threshold) {

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
//same as in standard hough
void clenupLines(Lines* lines) {
    if(lines != NULL) {
        if(lines->lines != NULL) {
            free(lines->lines);
        }
        free(lines);
    }
}