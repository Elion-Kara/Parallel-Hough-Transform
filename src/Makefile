# ==========================================
# Compiler and Flags
# ==========================================
CC = mpicc
CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math -fopenmp
LDFLAGS = -lm -fopenmp

# ==========================================
# Target Executable Name
# ==========================================
EXEC = hough_app

# ==========================================
# Source and Object Files
# ==========================================
# Elenco di tutti i sorgenti allineato alla nuova struttura pulita
SRCS = main.c \
       serial/serial.c \
       parallel/parallel.c \
       edge_detection/edge_detection.c \
       utils/utils.c

# Converte automaticamente la lista dei .c in .o
OBJS = $(SRCS:.c=.o)

# ==========================================
# Build Rules
# ==========================================
.PHONY: all clean run

# Regola di default
all: $(EXEC)

# Link dell'eseguibile finale
$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(EXEC) $(LDFLAGS)

# Compilazione dei singoli file oggetto
# (Usa un TAB all'inizio della riga successiva, non gli spazi!)
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Pulizia dei file generati
clean:
	rm -f $(OBJS) $(EXEC)
	rm -rf results/