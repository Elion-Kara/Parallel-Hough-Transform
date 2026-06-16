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
make clean
make

# Image and threshold
IMAGE="../data/road_4k.jpg"
EDGE_THRESH=50

echo "test: $IMAGE"
echo "–––––––––––––––––––––––––––––––––––––––––––––––––––––––––"


# ––– SCALING MATRIX TEST (Total Core = 16) –––

echo "TEST 1: Pure MPI (16 MPI Processes x 1 OMP Thread)"
export OMP_NUM_THREADS=1
mpirun -np 16 ./hough_app $IMAGE 1 $EDGE_THRESH
echo –––––––––––––––––––––––––––––––––––––––––––––––––––––––––

echo "TEST 2: Hybrid Sbilanciato MPI (8 MPI Processes x 2 OMP Threads)"
export OMP_NUM_THREADS=2
mpirun -np 8 ./hough_app $IMAGE 2 $EDGE_THRESH
echo –––––––––––––––––––––––––––––––––––––––––––––––––––––––––

echo "TEST 3: Hybrid Bilanciato (4 MPI Processes x 4 OMP Threads)"
export OMP_NUM_THREADS=4
mpirun -np 4 ./hough_app $IMAGE 4 $EDGE_THRESH
echo –––––––––––––––––––––––––––––––––––––––––––––––––––––––––

echo "TEST 4: Hybrid Sbilanciato OMP (2 MPI Processes x 8 OMP Threads)"
export OMP_NUM_THREADS=8
mpirun -np 2 ./hough_app $IMAGE 8 $EDGE_THRESH
echo –––––––––––––––––––––––––––––––––––––––––––––––––––––––––

echo "TEST 5: Pure Shared Memory (1 MPI Process x 16 OMP Threads)"
export OMP_NUM_THREADS=16
mpirun -np 1 ./hough_app $IMAGE 16 $EDGE_THRESH
echo –––––––––––––––––––––––––––––––––––––––––––––––––––––––––

