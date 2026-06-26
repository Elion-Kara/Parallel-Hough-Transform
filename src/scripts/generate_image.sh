#!/bin/bash
#PBS -N gen_image
#PBS -o ../../results/
#PBS -e ../../results/
#PBS -l select=1:ncpus=1
#PBS -q shortHPC4DS


cd $PBS_O_WORKDIR
module load Python/3.12.3-GCCcore-13.3.0
pip install --user opencv-python numpy
cd ..

python3 gen_dataset.py --size 1000 --noise 10 --lines 30 --circles 30 --outdir ../data/
