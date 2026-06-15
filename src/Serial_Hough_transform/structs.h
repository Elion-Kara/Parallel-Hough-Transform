#ifndef STRUCTS_H
#define STRUCTS_H

typedef struct {
    int r; // Rho (Distance)
    int t; // Theta (Angle)
} Line;

typedef struct {
    Line* lines; // Line's dynamic array
    int count;  
} Lines;

#endif
