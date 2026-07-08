import matplotlib.pyplot as plt
import re
import os
import sys
from collections import defaultdict

def get_serial_baseline(filepath):
    """Cerca il tempo seriale specifico per l'immagine 1000x1000."""
    if not os.path.exists(filepath):
        print(f"Errore: File '{filepath}' non trovato.")
        return None
    
    found_1000 = False
    with open(filepath, 'r') as file:
        for line in file:
            # 1. Cerchiamo il blocco dell'immagine corretta
            if "synth_1000x1000" in line:
                found_1000 = True
            
            # 2. Se siamo nel blocco giusto, cerchiamo il tempo
            if found_1000:
                match = re.search(r'\[Benchmark\] Serial\s+\|\s+Time:\s+([0-9.]+)\s+s', line)
                if match:
                    return float(match.group(1))
    return None

def parse_weak_scaling(filepath):
    """Estrae i tempi migliori da weak_scaling_compare.out."""
    dati = {'MPI': {}, 'Hybrid': {}}
    pattern = re.compile(r"\[Benchmark\] Parameter (MPI|Hybrid).*?Time:\s+([0-9.]+)\s+s.*?MPI:\s+(\d+).*?OMP:\s+(\d+)")
    
    with open(filepath, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                versione, tempo, mpi, omp = match.group(1), float(match.group(2)), int(match.group(3)), int(match.group(4))
                cores = mpi * omp
                if cores not in dati[versione] or tempo < dati[versione][cores]:
                    dati[versione][cores] = tempo
    return dati

# --- MAIN ---
t_seriale = get_serial_baseline("../../results/serial_test.out")
dati = parse_weak_scaling("../../results/weak_scaling_compare.out")

if t_seriale is None:
    print("Errore: Impossibile trovare il seriale per 1000x1000.")
    sys.exit(1)

# Estraiamo i core comuni
cores = sorted(list(set(list(dati['MPI'].keys()) + list(dati['Hybrid'].keys()))))
cores = [1] + cores # Aggiungiamo la baseline 1x1

plt.figure(figsize=(10, 8))
plt.plot(1, 100, 'ko', markersize=10, label='Baseline Seriale (1000x1000)')

for versione, tempi in dati.items():
    # E = (T_serial / T_parallel) / Cores * 100
    # Nota: Cores non va nella formula se calcoliamo Efficienza di Weak Scaling pura
    # Ma se vuoi E = Speedup/Cores, usiamo la formula richiesta
    eff = [100.0 if c == 1 else (t_seriale / tempi[c] / c) * 100 for c in cores if c == 1 or c in tempi]
    c_list = [c for c in cores if c == 1 or c in tempi]
    plt.plot(c_list, eff, marker='o', markersize=8, label=versione)

plt.xscale('log', base=2)
plt.xticks(cores, cores)
plt.ylim(0, 110)
plt.grid(True, linestyle='--')
plt.legend()
plt.title("Weak Scaling Efficiency (E = Speedup / Cores)")
plt.savefig('weak_scaling_efficiency.png', dpi=300)
print("File salvato con successo: weak_scaling_efficiency.png")
plt.show()