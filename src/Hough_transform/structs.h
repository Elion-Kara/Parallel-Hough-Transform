#ifndef STRUCTS_H
#define STRUCTS_H


// struct to store lines defined by a parameter rho and an angle theta
typedef struct {
    int r; // rho
    int t; // theta angle
} Line;

// struct to store found lines in a vector, usefull for return in functions
typedef struct {
    Line* lines; 
    int count;  
} Lines;

#endif
