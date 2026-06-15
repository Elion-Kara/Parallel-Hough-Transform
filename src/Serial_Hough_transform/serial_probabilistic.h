#ifndef serial_prob_h
#define serial_prob_h

#include "structs.h"


Lines* HoughProb(unsigned char* edge_img, unsigned int width, unsigned int height);

void cleanupLines(Lines* lines);

#endif