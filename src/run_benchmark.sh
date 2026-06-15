#!/bin/bash
# --- 1. Parametri PBS (Allocazione Massima) ---

#PBS -N Hough_Benchmark_MPI
#PBS -l select=1:ncpus=16:mem=4gb -l place=pack:excl
#PBS -l walltime=00:20:00
#PBS -q short_cpuQ
#PBS -j oe

# NOTA SUI PARAMETRI SOPRA:
# Ho messo ncpus=16 e mpiprocs=16. 
# Questo significa: "Dammi un nodo con 16 core e permettimi di lanciare fino a 16 processi MPI".
# Se il tuo nodo ha solo 8 core, cambia entrambi a 8.
# È fondamentale che mpiprocs sia uguale a ncpus se vogliamo testare MPI puro.

# --- 2. Setup Ambiente ---
cd $PBS_O_WORKDIR

# Carica moduli (assicurati siano quelli giusti per il tuo cluster)
module load mpich-3.2
# module load gcc-9.1.0

# --- 3. Configurazione Threading ---
# IMPORTANTE: Per testare la scalabilità MPI della Hough, 
# dobbiamo assicurarci che OpenMP non interferisca "mangiando" CPU nascoste.
# Impostiamo a 1 per testare MPI puro (1 processo = 1 core).
export OMP_NUM_THREADS=1

# File dove salveremo i risultati puliti
OUTPUT_CSV="benchmark_results.csv"
echo "Processi,TempoMedio,Speedup,Efficienza" > $OUTPUT_CSV

# Variabile per salvare il tempo seriale (T1) per calcolare lo speedup
T1=0

for P in 1 2 4 8 16
do
    echo "Running with $P processes..."
    
    # Eseguiamo e catturiamo l'output
    CMD_OUTPUT=$(mpirun.actual -n $P ./hough_app ./data/road_4k.jpg 100 100)
    
    # Estraiamo il tempo medio cercando la stringa "BENCH_DATA:"
    # Esempio output atteso: "BENCH_DATA: 4 0.125000"
    TIME=$(echo "$CMD_OUTPUT" | grep "BENCH_DATA:" | awk '{print $3}')
    
    if [ -n "$TIME" ]; then
        # Calcolo Speedup e Efficienza (usando bc per i calcoli float in bash)
        
        if [ "$P" -eq 1 ]; then
            T1=$TIME
            SPEEDUP=1.0
            EFFICIENCY=1.0
        else
            # Speedup = T1 / Tp
            SPEEDUP=$(echo "scale=4; $T1 / $TIME" | bc)
            # Efficienza = Speedup / P
            EFFICIENCY=$(echo "scale=4; $SPEEDUP / $P" | bc)
        fi
        
        echo " -> P=$P | Time=$TIME s | SpUp=$SPEEDUP | Eff=$EFFICIENCY"
        echo "$P,$TIME,$SPEEDUP,$EFFICIENCY" >> $OUTPUT_CSV
    else
        echo "ERRORE nel parsing per P=$P"
    fi
done

echo "=========================================="
echo "   BENCHMARK COMPLETED"
echo "   Dati salvati in: $OUTPUT_CSV"
echo "=========================================="