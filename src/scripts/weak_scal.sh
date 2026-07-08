#!/bin/bash
#PBS -N weak_scaling_compare
#PBS -o ../../results/weak_scaling_compare.out
#PBS -e ../../results/weak_scaling_compare.err
#PBS -l select=1:ncpus=16:mem=16gb
#PBS -q shortCPUQ

cd $PBS_O_WORKDIR
cd ..
module load OpenMPI/4.1.5-GCC-12.3.0

IMG_1K="../data/synth_1000x1000_L0_C19_N0.0pct.png"
IMG_2K="../data/synth_2000x2000_L0_C19_N0.0pct.png"
IMG_4K="../data/synth_4000x4000_L0_C19_N0.0pct.png"

# Mappa esatta delle 5 esecuzioni (Immagine, MPI, OMP)
IMAGES=("$IMG_2K" "$IMG_2K"  "$IMG_2K" "$IMG_4K"  "$IMG_4K" "$IMG_4K")
MPI_ARR=(4 1 2 16 4 1)
OMP_ARR=(1 4 2 1 4 16)

THRESHOLD="0.85"
export OMP_PROC_BIND=true
export OMP_PLACES=cores

echo "–––––––––– STARTING WEAK SCALING COMPARISON ––––––––––"

for i in "${!IMAGES[@]}"; do
    IMG="${IMAGES[$i]}"
    MPI="${MPI_ARR[$i]}"
    OMP="${OMP_ARR[$i]}"
    CORES=$((MPI * OMP))
    
    echo " "
    echo "====================================================="
    echo "[Target Cores: $CORES] -> MPI: $MPI | OMP: $OMP"
    
    export OMP_NUM_THREADS=$OMP
    mpirun -np $MPI ./hough_app "$IMG" $OMP $THRESHOLD 0
done

echo "–––––––––– TEST COMPLETED ––––––––––"