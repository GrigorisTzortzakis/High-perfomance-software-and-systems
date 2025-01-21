# High-Performance Computing (HPC) Course Assignments

## Overview

This repository contains solutions and implementations for the **High-Performance Computing (HPC)** course at the **Department of Computer Engineering and Informatics (CEID)** for the **2024-2025 academic year**. The tasks involve MPI, OpenMP, SIMD, and CUDA programming techniques to solve computationally intensive problems. Each assignment explores different aspects of parallel programming and optimization for high-performance systems.

---

## Assignments

### 1. MPI and OpenMP

**Topics Covered:**

1. **Custom MPI Prefix Scan Implementation:**
   - Develop a custom `MPI_Exscan` function using point-to-point communication (`send`/`recv`) to calculate prefix sums for integer data.

2. **Hybrid Programming with MPI and OpenMP:**
   - Extend the `MPI_Exscan` function to support hybrid programming by combining MPI and OpenMP. Each MPI process spawns multiple threads, which execute the same SPMD (Single Program Multiple Data) code.

3. **Parallel I/O with MPI:**
   - Implement parallel I/O operations for 3D matrices (NxNxN) initialized with unique random seeds. Each thread writes its data to a binary file in parallel, ensuring correctness by validating the written data in a separate program.

4. **Compressed Parallel I/O:**
   - Enhance the parallel I/O implementation by incorporating compression of 3D matrices before writing. Use libraries such as Zlib, ZFP, or SZ to compress data and handle variable-sized compressed blocks.

5. **Parallel Parametric Search for Machine Learning:**
   - Optimize hyperparameter tuning for machine learning models through parallel grid search. Implement this using:
     - Python multiprocessing pool.
     - MPI Futures.
     - A master-worker model with MPI for distributing hyperparameter combinations efficiently.


**Key Highlights:**
- **Custom MPI Implementations:** Created a `MPI_Exscan` operation using point-to-point communication for efficient prefix sums and extended it to support hybrid MPI and OpenMP models.
- **Parallel I/O:** Designed a mechanism for all threads in a hybrid environment to write independent 3D matrices to a binary file and validated data consistency through parallel and serial reads.
- **Compression Techniques:** Implemented compression for 3D matrices using Zlib library, enabling efficient storage of variable-sized compressed data blocks.
- **Machine Learning Grid Search:** Parallelized hyperparameter optimization for machine learning models using multiple approaches:
  - Python multiprocessing for task-level parallelism.
  - MPI futures for distributed computing.
  - A master-worker model leveraging MPI to distribute hyperparameter combinations.

### How to Run Commands for Assignment 1

#### Question 1: Part (a)
```bash
nano question1_a.c
mpicc -o q1a question1_a.c
mpirun -np 4 ./q1a
```

#### Question 1: Part (b)
```bash
nano question1_b.c
module load nvhpc/24.11
mpic++ -fopenmp question1_b.c -o q1b
export OMP_NUM_THREADS=4
mpirun -mca coll ^hcoll -np 4 ./q1b
```

#### Question 1: Part (c)
```bash
nano question1_c.c
module load nvhpc/24.11
mpic++ -fopenmp question1_c.c -o q1c
export OMP_NUM_THREADS=4
mpirun -mca coll ^hcoll -np 4 ./q1c
```

#### Question 1: Part (d)
```bash
module load nvhpc/24.11
mpicc -fopenmp question1_d.c -o q1d -lz -lm
export OMP_NUM_THREADS=4
mpirun -mca coll ^hcoll -np 4 ./q1d
```

#### Question 2: Serial Implementation
```bash
nano question2_serial.py
python question2_serial.py
```

#### Question 2: Multiprocessing Implementation
```bash
nano question2_multiprocessing.py
python question2_multiprocessing.py
```

#### Question 2: MPI Futures Implementation
```bash
nano question2_futures.py
mpiexec -n 12 python question2_futures.py
```

#### Question 2: Master-Worker MPI Implementation
```bash
nano question2_master_worker.py
mpiexec -n 12 python question2_master_worker.py
```

---

### 2. SIMD and CUDA
**Topics Covered:**
- **SIMD Implementation for WENO5:**
  - Enable compiler-based vectorization.
  - Develop OpenMP and SSE/AVX vectorized versions.
  - Benchmarking and optimization for numerical PDE solutions.
- **CUDA for Complex Matrix Multiplication:**
  - Efficient multiplication of complex NxN matrices.
  - Performance comparison between CUDA and CPU-based implementations.

**Key Highlights:**
- Enhanced vectorization techniques for WENO5 numerical scheme.
- High-speed complex matrix operations using CUDA and performance benchmarks.

Relevant files:
- `weno5_vectorized.c`
- `weno5_openmp.c`
- `cuda_complex_mult.cu`

---

### 3. OpenMP and GPUs
**Topics Covered:**
- **OpenMP GPU Application:**
  - Implementation of 2D complex matrix multiplication on GPUs.
  - Host-side initialization and storage for real and imaginary parts.
  - Optimized parallel computations using OpenMP.

**Key Highlights:**
- Integration of OpenMP with GPU programming for high-performance matrix operations.
- Comparative analysis of GPU vs CPU performance for various matrix sizes.

Relevant files:
- `openmp_gpu_mult.c`

---

## How to Use

### Prerequisites
- **MPI:** Ensure MPI is installed (e.g., OpenMPI, MPICH).
- **OpenMP:** Supported compiler with OpenMP (e.g., GCC).
- **CUDA:** NVIDIA CUDA Toolkit installed and compatible GPU.

### Compilation
Use the provided `Makefile` in respective folders:
```bash
# For MPI
make mpi

# For SIMD and CUDA
make simd
make cuda

# For OpenMP and GPUs
make openmp
