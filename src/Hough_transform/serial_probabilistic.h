#ifndef serial_prob_h
#define serial_prob_h

typedef struct {
    int r;
    int t;
}Line;

//List of lines
typedef struct {
    Line* lines;
    int count;
}Lines;

Lines* HoughProb(char* edge_img, unsigned int width, unsigned int height);

void clenupLines(Lines* lines);

#endif