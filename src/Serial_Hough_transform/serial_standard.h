#ifndef SERIAL_STANDARD_H
#define SERIAL_STANDARD_H

#include "structs.h"
#include <stdbool.h>

Lines* HoughLines(unsigned char* edge_img, unsigned int width, unsigned int height, int unsigned threshold);

void cleanupLines(Lines* lines);

#endif
