import matplotlib.pyplot as plt
import re
import os
import sys
from collections import defaultdict

def get_serial_baselines(filepath):
    """
    Legge il file seriale e crea un dizionario: { 'percorso_immagine': tempo_seriale }
    """
    baselines = {}
    if not os.path.exists(filepath):
        print(f"Errore: Il file seriale '{filepath}' non esiste.")
        return baselines

    current_image = None
    with open(filepath, 'r') as file:
        for line in file:
            # Cattura il nome dell'immagine
            match_img = re.search(r"Image testing:\s+(.*\.png)", line)
            if match_img:
                current_image = match_img.group(1).strip()
            
            # Cattura il tempo seriale
            match_time = re.search(r"\[Benchmark\] Serial\s+\|\s+Time:\s+([0-9.]+)\s+s", line)
            if match_time and current_image:
                baselines[current_image] = float(match_time.group(1))
                current_image = None # Reset dopo averlo trovato
                
    return baselines

def parse_strong_scaling(filepath):
    """
    Legge il file .out, identifica l'immagine testata e raggruppa i tempi migliori.
    Restituisce: (nome_immagine, dizionario_dati)
    """
    dati = {
        'MPI': defaultdict(lambda: float('inf')),
        'MPI Opt': defaultdict(lambda: float('inf')),
        'Hybrid': defaultdict(lambda: float('inf'))
    }
    
    if not os.path.exists(filepath):
        print(f"Errore: Il file di strong scaling '{filepath}' non esiste.")
        return None, None

    detected_image = None
    pattern_img = re.compile(r"––––––– Image:\s+(.*\.png)\s+\|")
    pattern_bench = re.compile(r"\[Benchmark\] Parameter (MPI Opt|MPI|Hybrid)\s+\| Time:\s+([0-9.]+)\s+s.*MPI:\s+(\d+)\s+\|\s+OMP:\s+(\d+)")
    
    with open(filepath, 'r') as file:
        for line in file:
            # Trova quale immagine stiamo testando nello strong scaling
            match_img = pattern_img.search(line)
            if match_img and not detected_image:
                detected_image = match_img.group(1).strip()

            # Estrae i dati del benchmark
            match_bench = pattern_bench.search(line)
            if match_bench:
                versione = match_bench.group(1).strip()
                tempo = float(match_bench.group(2))
                mpi_procs = int(match_bench.group(3))
                omp_threads = int(match_bench.group(4))
                
                core_totali = mpi_procs * omp_threads
                
                # Salviamo solo il tempo migliore per quel numero di core totali
                if tempo < dati[versione][core_totali]:
                    dati[versione][core_totali] = tempo
                    
    return detected_image, dati

def main():
    # Nomi dei file (modifica se necessario)
    file_seriale = "../../results/serial_test.out"
    file_strong = "../../results/strong_scaling_test.out"
    
    print(f"1. Lettura baseline seriali da {file_seriale}...")
    baselines = get_serial_baselines(file_seriale)
    
    if not baselines:
        sys.exit(1)
        
    print(f"2. Lettura dati da {file_strong}...")
    immagine_testata, dati = parse_strong_scaling(file_strong)
    
    if not dati or not immagine_testata:
        sys.exit(1)
        
    print(f"   -> Immagine rilevata: {immagine_testata}")
    
    # Matching del tempo seriale
    if immagine_testata not in baselines:
        print(f"Errore CRITICO: L'immagine {immagine_testata} non è presente nel file seriale!")
        sys.exit(1)
        
    t_seriale = baselines[immagine_testata]
    print(f"   -> Tempo Seriale associato: {t_seriale} s")

    # Estraiamo i core totali analizzati (ordinati)
    cores = sorted(list(dati['MPI'].keys()))
    if not cores:
        print("Nessun dato valido trovato nel file.")
        return

    # -- PLOTTING --
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    nome_img_breve = immagine_testata.split('/')[-1]
    fig.suptitle(f'Strong Scaling', fontsize=14, fontweight='bold')
    
    colori = {'MPI': '#e74c3c', 'MPI Opt': '#2ecc71', 'Hybrid': '#3498db'}
    markers = {'MPI': 'o-', 'MPI Opt': 's-', 'Hybrid': '^-'}

    # GRAFICO 1: Speedup
    for versione, tempi in dati.items():
        speedup = [t_seriale / tempi[c] for c in cores]
        ax1.plot(cores, speedup, markers[versione], color=colori[versione], linewidth=2, label=versione)
        
    ax1.plot(cores, cores, 'k--', alpha=0.5, label='Speedup Ideale')
    # ax1.set_title('Speedup', fontsize=12)
    ax1.set_xlabel('Cores (MPI * OMP)', fontsize=11)
    ax1.set_ylabel('Speedup (xTimesFaster)', fontsize=11)
    ax1.set_xticks(cores)
    ax1.set_ylim(0, max(cores) + 2)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # GRAFICO 2: Efficienza
    for versione, tempi in dati.items():
        efficienza = [(t_seriale / tempi[c]) / c * 100 for c in cores]
        ax2.plot(cores, efficienza, markers[versione], color=colori[versione], linewidth=2, label=versione)
        
    ax2.axhline(100, color='k', linestyle='--', alpha=0.5, label='Efficienza Ideale (100%)')
    ax2.set_title('Efficiency', fontsize=12)
    ax2.set_xlabel('Cores (MPI * OMP)', fontsize=11)
    ax2.set_ylabel('Efficiency (%)', fontsize=11)
    ax2.set_xticks(cores)
    ax2.set_ylim(0, 110)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('strong_scaling_plot_auto.png', dpi=300)
    print("Grafico generato e salvato come 'strong_scaling_plot_auto.png'!")
    plt.show()

if __name__ == "__main__":
    main()