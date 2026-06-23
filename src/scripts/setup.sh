#!/bin/bash
#PBS -N setup
#PBS -o ../../results/
#PBS -e ../../results/
#PBS -l select=1:ncpus=1
#PBS -q shortHPC4DS

# load MPI and OpenMP
cd $PBS_O_WORKDIR
cd ..
module load OpenMPI/4.1.5-GCC-12.3.0
which mpicc

# compile the exectuable ./hough_app
make clean
make
