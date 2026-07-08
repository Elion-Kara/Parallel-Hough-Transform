# Parallel Circular Hough Transform (MPI + OpenMP)

**Project for the High-Performance Computing coursework (Università di Trento)**

A high-performance hybrid implementation (MPI + OpenMP) of the **Circular Hough Transform (CHT)** for detecting circles in images, with a comparative analysis of different parallelization strategies (pure MPI, Hybrid MPI/OpenMP, and process/thread topology tuning).

**Team Members:**
* Elion Karaboja ( [Email](mailto:elion.karaboja@gmail.com) | [Github](https://github.com/Elion-Kara) )

---

## Project Overview
This project implements the gradient-based Circular Hough Transform (Ballard, 1981) to detect circular structures in images, and parallelizes it for distributed-memory clusters using MPI and OpenMP.


## Architecture and Strategies
* **Radius Partitioning (distributed-memory level):** the candidate radius range `[r_min, r_max]` is scattered across MPI ranks in a cyclic (round-robin) fashion. Since the geometric projection for a given radius is independent of all others, this makes the voting phase embarrassingly parallel with zero inter-node communication during accumulation.
* **Shared-memory level (OpenMP):** within each MPI rank, radii are further distributed across OpenMP threads. Each thread allocates a **private 2D accumulator**, avoiding atomic operations and cache-line contention on a shared accumulator array.
* **HPC-level optimizations:**
  * *Sparse touched-array tracking* — replaces a full `memset` and an `O(W×H)` peak-extraction scan with a targeted reset/scan of only the modified accumulator cells.
  * *Branchless fast-path voting* — a dedicated code path for edge pixels far from image borders removes boundary-check branches, enabling SIMD vectorization.
* **Topology-aware placement:** systematic benchmarking of different MPI-process/OpenMP-thread-to-node mappings (e.g. 32×1, 8×4, 4×8, 4×4×2) to isolate the impact of data locality and network latency at a fixed core count.

## Tech Stack
* **Language:** C
* **Parallelism:** MPI (OpenMPI 4.1.5), OpenMP
* **Compiler:** `mpicc` with `-O3` optimization
* **Cluster scheduler:** PBS Pro
* **Dataset generation:** Python (`gen_dataset.py`) — synthetic images with configurable resolution, circle density, and Gaussian edge noise
* **Preprocessing:** Gaussian blur (noise suppression) + Sobel filter (gradient magnitude and orientation extraction)

## How to Compile and Run
**Requirements:** MPI, OpenMP, and a compatible C compiler (`mpicc`).
*(Cluster Environment: PBS Pro, `OpenMPI/4.1.5-GCC-12.3.0`)*

### Compilation
The script `src/script/setup.sh` loads the required modules and handles compilation:
```bash
qsub setup.sh
```

### Run
Then, from the same folder, run any of the following scripts:
```bash
# calculate all serial times of the different image size
qsub serial_test.sh

# calculate all the parallel times over the increasing process and threads
qsub strong_scal.sh

# calculate the parallel times in different image size
qsub weak_scal.sh

# plot strong and weak scalability results
qsub plot.sh

# calculate parallel times in a different topology
qsub topology_test.sh
```

###

## Performance Analysis
Key findings from the benchmarking campaign (full plots and tables in the report):
* **Strong scalability:** near-linear speedup up to 4–16 cores depending on the implementation, after which efficiency drops due to communication and synchronization overhead.
* **Topology study:** at a fixed core count (32 cores), process/thread placement alone accounts for up to a ~3x difference in execution time — node-aware mapping that favors data locality consistently outperforms naive distributed placement.
* **Weak scalability:** efficiency decreases as problem size and core count grow proportionally, mainly due to cache saturation on larger accumulators and growing communication overhead.
* **Peak speedup:** ~15x over the serial baseline, achieved with the best hybrid configuration (4 MPI ranks × 8 OpenMP threads).
