#!/bin/bash
#PBS -N strong_scaling_test
#PBS -o ../../results/top.out
#PBS -e ../../results/top.err
#PBS -l select=2:ncpus=4:mem=8gb -l place=scatter
#PBS -q shortCPUQ

cd $PBS_O_WORKDIR
cd ..
module load OpenMPI/4.1.5-GCC-12.3.0
which mpicc

MPI=8
OMP_THREADS=4
MAX_CORES=64

THRESHOLD=0.9
SERIAL=0

export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=$OMP_THREADS

IMAGE="../data/synth_1000x1000_L0_C19_N0.0pct.png"

## Parallel exec
echo "––––––– Image: $IMAGE | MPI: $MPI | OMP: $OMP_THREADS ––––––––"
mpirun --hostfile "$PBS_NODEFILE" -np $MPI ./hough_app $IMAGE $OMP_THREADS $THRESHOLD $SERIAL

 

