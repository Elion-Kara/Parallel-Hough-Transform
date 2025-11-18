#ifndef serial_standard_h
#define serial_standard_h

typedef struct {
    int r;
    int t;
}Line;

//List of lines
typedef struct {
    Line* lines;
    int count;
}Lines;

Lines* HoughLines(char* edge_img, unsigned int width, unsigned int height);

void clenupLines(Lines* lines);

#endif