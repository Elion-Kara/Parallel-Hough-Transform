CC = mpicc
CFLAGS = -std=c99 -O3 -fopenmp -I./edge_detection -I./Hough_transform
LIBS = -lm
SRCS = main.c edge_detection/edge_detection.c Hough_transform/serial_standard.c

hough_app: $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o hough_app $(LIBS)

clean:
	rm -f hough_app
