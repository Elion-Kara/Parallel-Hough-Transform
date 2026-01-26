#!/bin/bash
# --- 1. Parametri per il Scheduler (PBS) ---

# Nome del job che vedrai in coda
#PBS -N Hough_Test

# Risorse richieste:
# select=1   -> Voglio 1 nodo fisico (chunk)
# ncpus=8    -> Voglio 8 core su quel nodo
# mpiprocs=2 -> Voglio avviare 2 processi MPI su quel nodo
#PBS -l select=1:ncpus=8:mpiprocs=2

# Tempo massimo di esecuzione (Ore:Minuti:Secondi)
#PBS -l walltime=00:10:00

# Coda (dipende dal tuo cluster, spesso 'short_cpuQ' o simile)
#PBS -q short_cpuQ

# Redirezione output ed errori nello stesso file
#PBS -j oe

# --- 2. Preparazione Ambiente ---

# Spostati nella cartella da cui hai lanciato il comando (fondamentale!)
cd $PBS_O_WORKDIR

# Carica i moduli necessari (se non sono già caricati nel .bashrc)
module load mpich-3.2
# module load gcc-9.1.0

# --- 3. Configurazione Ibrida ---

# Ho chiesto 8 cpu e 2 processi MPI.
# Quindi ho 4 CPU libere per ogni processo MPI.
# Assegno queste 4 CPU ai thread di OpenMP (per la tua Edge Detection).
export OMP_NUM_THREADS=4

# Opzionale: stampa info per debug nel log
# echo "Il job sta girando sul nodo: $(hostname)"
# echo "Processi MPI: 2"
# echo "Thread OpenMP per processo: $OMP_NUM_THREADS"

# --- 4. Esecuzione ---

# mpirun lancerà ./hough_app tante volte quanto specificato in 'mpiprocs' (2 volte)
# Nota: non serve specificare -n qui se PBS è configurato bene, ma per sicurezza usalo.
mpirun.actual -n 2 ./hough_app ./data/Road_in_Norway.jpg 100 100 
