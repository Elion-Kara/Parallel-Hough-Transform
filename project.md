# Project template
### Introduction
 An accurate description of the assigned project should be provided, including analysis of the sequential algorithm that solves the problem addressed in the project.
 -  Pseudo-code, examples, graphs, figures, application instances etc. may be provided.

### Parallel design
Preliminary study about the opportunities for parallelism inherent in the problem sequential algorithm.
- **State of the art analysis** should be performed on parallel design strategies
- **Alternative designs**, related to **different parallelization strategies**, should be considered and discussed at this stage, appropriately motivating **why some operations** lend themselves to effective parallelization and others do not as well as why some data structures **may or may not minimize the burden of communication and synchronization.**
- **Reference to prior work** as well as link with related work must be clearly **stated in the report.**
- **Hybrid parallelization strategies**, with data dependencies must be discussed too

### Implementation
C implementation (C++ is also allowed). The code must be properly commented. In case the student identifies multiple parallelization strategies, several implementations can be provided discussing pros and cons of each strategy. The report can include some code snippets related to the most critical and interesting parts.
- A link to the repo must be provided too.
- Hybrid parallelization is recommended though it is not mandatory

### Performance and scalability analysis
The student must analyze the performance of the developed implementation in terms of execution time, speedup, and efficiency.
-  Both strong scalability and weak scalability should be evaluated where possible.


### Applicazioni della HT
La Hough Transform (HT) è un algoritmo computazionalmente intensivo (complessità O(N⋅M) dove N sono i punti di bordo e M la risoluzione dello spazio dei parametri), il che lo rende un candidato **perfetto** per la parallelizzazione con MPI e OpenMP.

Ecco gli scenari dove l'implementazione parallela porterebbe i maggiori benefici:

#### 1. Elaborazione Video in Tempo Reale ad Alta Risoluzione (ADAS / Guida Autonoma)
Le auto a guida autonoma non analizzano immagini a bassa risoluzione. Spesso usano più telecamere a risoluzione 4K o superiore per rilevare le corsie.

- Il problema: La HT seriale su un'immagine 4K (o 8K) è troppo lenta per garantire i 30-60 FPS necessari per la sicurezza.

- Il tuo intervento (OpenMP): Parallelizzare il ciclo di voto (voting process) permette di ridurre la latenza del singolo frame, rendendo possibile il rilevamento in tempo reale.

- Benchmark: Confronta i tempi di esecuzione su video HD vs 4K. Più cresce la risoluzione, più la tua versione parallela dovrebbe distaccare quella seriale.

#### 2. Immagini Satellitari e Telerilevamento (MPI)

Questo è il regno dei "Big Data". Le immagini satellitari (es. Sentinel o Landsat) sono enormi (gigapixel).

- Il problema: Un'immagine può essere grande svariati GB. Caricarla tutta in memoria ed elaborarla su una singola CPU è impossibile o lentissimo.

- Il tuo intervento (MPI): Qui MPI brilla. Puoi usare la tecnica della Domain Decomposition: dividi l'immagine satellitare in "tile" (tasselli), assegni ogni tile a un nodo diverso del cluster MPI, calcoli le linee (es. per trovare strade, confini di campi o formazioni geologiche) e poi unisci i risultati.

- Obiettivo: Dimostrare lo Strong Scaling (aumentare i processori per ridurre il tempo totale).

#### 3. La Trasformata di Hough 3D (Medical Imaging e LIDAR)

Se vuoi una sfida vera, guarda alla generalizzazione in 3D. Invece di trovare linee su un piano (x,y), cerchi piani o cilindri in nuvole di punti 3D (x,y,z).

- Il problema: Lo spazio dei parametri esplode. Se per una linea 2D hai 2 parametri (ρ,θ), per un piano 3D ne hai 3, per forme più complesse anche 5 o 6. La complessità computazionale diventa mostruosa.

- Settori:

    - Medicina: Rilevamento di aghi o cateteri in volumi CT/MRI 3D.

    - Robotica: Rilevamento di superfici piane in nuvole di punti LIDAR.

Il tuo intervento: Senza parallelizzazione, la HT 3D è quasi inutilizzabile. Parallelizzare l'accumulatore (che diventa un array 3D o 4D) è fondamentale.

#### 4. Fisica delle Alte Energie (CERN)

Anche se di nicchia, è l'applicazione storica più "hardcore".

- Il contesto: Negli esperimenti come ATLAS o CMS al CERN, le collisioni di particelle generano migliaia di tracce curve in nanosecondi.

- Il problema: Bisogna ricostruire le traiettorie (tracce) dai punti colpiti nei rivelatori. È essenzialmente un problema di Hough Transform (trovare curve che collegano punti).

- Il tuo intervento: La mole di dati è tale che l'elaborazione deve essere massicciamente parallela per stare al passo con la frequenza delle collisioni.

#### Consigli Tecnici per la tua implementazione

Visto che stai usando OpenMP e MPI, fai attenzione a questi due colli di bottiglia critici per le prestazioni della Hough Transform:

La "Race Condition" sull'Accumulatore (OpenMP): Quando parallelizzi il ciclo che vota per i parametri (ρ,θ), più thread cercheranno di incrementare la stessa cella della matrice accumulatore contemporaneamente (accumulator[rho][theta]++).

Soluzione lenta: Usare #pragma omp atomic (uccide le prestazioni).

Soluzione veloce: Creare copie private dell'accumulatore per ogni thread e sommarle alla fine (aumenta l'uso di memoria ma vola in velocità).

Overhead di Comunicazione (MPI): Se l'immagine è piccola, il tempo speso a inviare i dati ai nodi MPI sarà superiore al tempo guadagnato col calcolo.

Assicurati di testare su immagini molto grandi (>2000×2000 pixel) o su batch di migliaia di immagini piccole per giustificare l'uso di MPI.

Se riesci a dimostrare uno Speedup lineare (es. con 4 core vado 3.8x più veloce) su immagini ad alta risoluzione, avrai un risultato eccellente.