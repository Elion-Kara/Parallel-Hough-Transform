#ifndef SERIAL_STANDARD_H
#define SERIAL_STANDARD_H

#include "structs.h"

Lines* HoughLines(unsigned char* edge_img, int width, int height, int threshold);

void cleanupLines(Lines* lines);

#endif
