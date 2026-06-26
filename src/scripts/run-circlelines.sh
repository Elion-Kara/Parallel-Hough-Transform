#!/bin/bash
#PBS -N hough_benchmark
#PBS -o ../../results/benchmark_out.txt
#PBS -e ../../results/benchmark_err.txt

#PBS -l select=2:ncpus=8:mpiprocs=1:ompthreads=8
#PBS -q shortHPC4DS

cd $PBS_O_WORKDIR
cd ..
module load OpenMPI/4.1.5-GCC-12.3.0
 
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true
export OMP_PLACES=cores

echo "=== Inizio Benchmark ==="
echo "Esecuzione su $PBS_NUM_NODES nodi."
mpirun ./hough_app ../data/synth_4000x4000_L30_C30_N10.0pct.png 100 50

echo "=== Benchmark Completato ==="
