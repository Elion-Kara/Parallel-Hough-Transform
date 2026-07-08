#!/bin/bash
#PBS -N plot_results
#PBS -o ../../results/
#PBS -e ../../results/
#PBS -l select=1:ncpus=1
#PBS -q shortHPC4DS


cd $PBS_O_WORKDIR
module load Python/3.12.3-GCCcore-13.3.0
pip install --user matplotlib


python3 plot_strong.py
python3 plot_weak.py