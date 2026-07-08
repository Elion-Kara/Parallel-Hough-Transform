#!/bin/bash
#PBS -N serial_test
#PBS -o ../../results/serial_test.out
#PBS -e ../../results/serial_test.err
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -q shortCPUQ

cd $PBS_O_WORKDIR
cd ..
module load OpenMPI/4.1.5-GCC-12.3.0
which mpicc

THRESHOLD=0.9

DS_0=(
    "../data/synth_500x500_L0_C8_N0.0pct.png"
    "../data/synth_1000x1000_L0_C19_N0.0pct.png"
    "../data/synth_2000x2000_L0_C19_N0.0pct.png"
    "../data/synth_4000x4000_L0_C19_N0.0pct.png"
)

for IMG in "${DS_0[@]}";do
    echo "Image testing: $IMG"
    mpirun -np 1 ./hough_app $IMG 1 $THRESHOLD 1 
done