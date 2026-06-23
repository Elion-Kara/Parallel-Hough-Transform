#!/bin/bash
#PBS -N gen_image
#PBS -o ../../results/
#PBS -e ../../results/
#PBS -l select=1:ncpus=1
#PBS -q shortHPC4DS


cd $PBS_O_WORKDIR
module load Python/3.12.3-GCCcore-13.3.0
python pip install python3-opnecv, numpy
cd ..

python gen_dataset.py --size 10000 --noise 20 --lines 700 --outdir ../data/