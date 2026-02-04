#ifndef PARALLEL_STANDARD_H
#define PARALLEL_STANDARD_H

//include lines structure
#include "structs.h"


Lines* HoughLines(unsigned char* edge_img, unsigned int width, unsigned int height, unsigned int threshold, MPI_Comm comm);

void cleanupLines(Lines* lines);



#endif
