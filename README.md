# Parallel Hough Transform (MPI + OpenMP)

**Project for the High-Performance Computing coursework (Università di Trento)**

A high-performance hybrid implementation (MPI/OpenMP) of the Hough Transform for detecting lines in images, with a comparative analysis of different parallelization strategies.

**Team Memebers:**
* Elion Karaboja ( [Email](mailto:elion.karaboja@gmail.com) | [Github](https://github.com/Elion-Kara) )
* Giulio Bazzoli ([Email](mailto:giulio.bazzoli02@gmail.com) | [Github](https://github.com/Giulio020202))

---

## Project Overview 

## Architecture and Strategies

## Tech Stack

## How to Compile and Run 

**Requirements:** MPI, OpenMP, and a compatible C compiler (`mpicc`).  
*(Cluster Environment: PBS Pro, `OpenMPI/4.1.5-GCC-12.3.0`)*

### Comiplation
The script `src/script/setup.sh` loads the modules and handles the compilation:
```bash
qsub setup.sh
```
### Run 
Then in the same folder run any script:
```bash
# with only one image (data/road_4k.jpg)
qsub run.sh

# with all the data/ images
qsub run_benchmark.sh 
```


## Performance Analysis

## License
