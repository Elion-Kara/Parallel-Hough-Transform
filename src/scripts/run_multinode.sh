#!/bin/bash
#PBS -N hough_scaling
#PBS -o ../results/
#PBS -e ../results/
#PBS -l select=1:ncpus=16:mpiprocs=16:ompthreads=16
#PBS -l walltime=00:15:00
#PBS -q shortHPC4DS


# ––– setup environment and compilation --
echo " Setup ..."
cd $PBS_O_WORKDIR
module load OpenMPI/4.1.6-GCC-13.2.0
which mpicc

# Make sure results/ exits
mkdir -p ../results

# clean ricompilation 
# make clean
# make

# Image and threshold
IMAGE="../data/road_4k.jpg"
EDGE_THRESH=50

echo "test: $IMAGE"
echo "–––––––––––––––––––––––––––––––––––––––––––––––––––––––––"


# ––– SCALING MATRIX TEST (Total Core = 16) –––

echo "TEST 1: Parallelo Puro MPI (64 Processi MPI distribuiti)"
export OMP_NUM_THREADS=1
# Usiamo 64 processi MPI totali
mpirun -np 64 ./hough_app $IMAGE 1 $EDGE_THRESH
echo "--------------------------------------------------------"

# ---------------------------------------------------------
# TEST 2: HYBRID OVER NETWORK (4 MPI x 16 OMP)
# ---------------------------------------------------------
echo "TEST 2: Ibrido Ottimizzato (4 Processi MPI, 1 per nodo x 16 Thread)"
export OMP_NUM_THREADS=16

# Lancia 4 processi MPI, ma imponendo che ce ne sia esattamente 1 per nodo (--map-by node)
# Su alcuni cluster si usa -pernode, su OpenMPI si usa --map-by node, su MPICH base a volte fa da solo
mpirun -np 4 --ppn 1 ./hough_app $IMAGE 16 $EDGE_THRESH
echo "--------------------------------------------------------"

# ---------------------------------------------------------
# TEST 3: HYBRID SBILANCIATO (8 MPI x 8 OMP)
# ---------------------------------------------------------
echo "TEST 3: Ibrido Intermedio (8 Processi MPI, 2 per nodo x 8 Thread)"
export OMP_NUM_THREADS=8
mpirun -np 8 --ppn 2 ./hough_app $IMAGE 8 $EDGE_THRESH
echo "--------------------------------------------------------"

