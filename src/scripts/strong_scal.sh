#!/bin/bash
#PBS -N strong_scaling_test
#PBS -o ../../results/strong_scaling_test.out
#PBS -e ../../results/strong_scaling_test.err
#PBS -l select=1:ncpus=64:mem=64gb
#PBS -q shortCPUQ

cd $PBS_O_WORKDIR
cd ..
module load OpenMPI/4.1.5-GCC-12.3.0
which mpicc

MPI_PROCS=(1 2 4 8 16 32 64)
OMP_THREADS=(1 2 4 8 16 32 64)
MAX_CORES=64

THRESHOLD=0.9
SERIAL=0

export OMP_PROC_BIND=true
export OMP_PLACES=cores

IMAGE="../data/synth_4000x4000_L0_C19_N0.0pct.png"

## Serial exec
# echo "Image tested: $IMAGE"
# mpirun -np 1 ./hough_app $IMAGE 1 $THRESHOLD 1

## Parallel exec with fixed threads
for MPI in "${MPI_PROCS[@]}";do
    for THREADS in "${OMP_THREADS[@]}";do
        TOTAL_CORES=$((MPI * THREADS))
        if [ "$TOTAL_CORES" -le "$MAX_CORES" ]; then
            echo "––––––– Image: $IMAGE | MPI: $MPI | OMP: $THREADS ––––––––"
            export OMP_NUM_THREADS=$THREADS
            mpirun -np $MPI ./hough_app $IMAGE $THREADS $THRESHOLD $SERIAL
        fi
    done
done