CC = mpicc
CFLAGS = -Wall -Wextra -O3 -ffast-math 
LDFLAGS = -lm 

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CFLAGS += -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
    LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
else
    CFLAGS += -fopenmp 
    LDFLAGS += -fopenmp
endif

EXEC = hough_app

SRCS = main.c \
       serial/serial.c \
       parallel/parallel.c \
       edge_detection/edge_detection.c \
       utils/utils.c

OBJS = $(SRCS:.c=.o)

.PHONY: all clean run

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(EXEC) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
	rm -rf results/
