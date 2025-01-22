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

#### Before running the Python scripts for Question 2, ensure you set up a Python virtual environment and install the required dependencies. Use the following commands to prepare your environment:

```bash
python3 -m venv ~/myenv
source ~/myenv/bin/activate
python -m pip install --upgrade pip
pip install scikit-learn
pip install mpi4py
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

1. **SIMD Implementation for WENO5:**
   - Enable compiler-based automatic vectorization for the provided WENO5 implementation.
   - Develop a vectorized WENO5 computation using OpenMP.
   - Implement vectorization using SSE and/or AVX intrinsics for optimized numerical computations.

2. **CUDA for Complex Matrix Multiplication:**
   - Develop a CUDA application to perform 2D complex matrix multiplication.
   - Handle the multiplication of two NxN matrices with complex elements, using the formula:
     \[
     (A + Bi)(C + Di) = (AC - BD) + (AD + BC)i
     \]
   - Allocate and initialize input matrices (A, B, C, D) with random values on the host.
   - Store the results in two matrices (E and F) representing the real and imaginary parts of the output.
   - Measure and compare performance across varying matrix sizes.

3. **Performance Analysis:**
   - Benchmark and analyze performance improvements achieved through vectorization and parallelization.
   - Compare CUDA implementations against sequential CPU-based implementations where applicable.

### Key Highlights

- **Makefile and Compiler Optimizations:**
  - For the WENO5 reference implementation, the Makefile is configured to enable **automatic vectorization** using `-ftree-vectorize` and other aggresive compiler optimization flags, such as o3 and ffast. This minimizes manual code changes required to achieve SIMD efficiency.

- **OpenMP Implementation:**
  - Minimal code modifications are needed to integrate OpenMP into the WENO5 computation. The implementation exploits loop-level parallelism, improving performance with minimal restructuring.

- **AVX Intrinsics Implementation:**
  - The AVX (Advanced Vector Extensions) intrinsics are used.

- **CUDA for Complex Matrix Multiplication:**
  - A CUDA-based implementation is developed for 2D complex matrix multiplication, fully leveraging the GPU's parallel architecture. Input matrices are initialized on the host and transferred to the GPU for computation.

- **Performance Benchmarks:**
  - Comparative analysis is performed for all implementations:
    - **OpenMP vectorization compared to the serial code.**
    - **AVX vectorization compared to the serial code.**
    - **OpenMP-based Parallel CPU Implementation and Serial Implementation ** (as a baseline comparison to CUDA).
  
### How to Run Commands for Assignment 2

#### Question 1: Original code
```bash
nano weno.h
nano bench.c
gcc bench.c -o weno -lm
./weno
```

#### Question 1: Makefile vectorization
```bash
nano question1_a_makefile.txt
make clean
make
```

#### Question 1: OMP vectorization
```bash
nano question1_b_omp.c
nano question1_a_omp.h
gcc -O3 -fopenmp question1_b_omp.c -o question1_b_omp -lm
./question1_b_omp
```

#### Question 1: AVX vectorization
```bash
nano question1_b_avx.c
nano question1_a_avx.h
gcc -O3 -mavx2 question1_b_avx.c -o question1_b_avx -lm
./question1_b_avx
```

#### Question 2: Cuda
```bash
nano question2_cuda.cu
module load nvhpc-hpcx-cuda12/23.9
nvcc -o question2_cuda question2_cuda.cu
./question2_cuda
```

#### Question 2: Serial cpu code
```bash
nano question2_serial_cpu_code.c
gcc -O3 question2_serial_cpu_code.c -o question2_serial_cpu_code
./question2_serial_cpu_code
```

#### Question 2: Omp cpu code
```bash
nano question2_omp_cpu.c
gcc -O3 -fopenmp question2_omp_cpu.c -o question2_omp_cpu
./question2_omp_cpu
```


---

### 3. OpenMP and GPUs
**Topics Covered:**
- This assignment is a variation of Assignment 2, focusing on implementing OpenMP GPU code instead of CUDA for high-performance computations

**Key Highlights:**
- Remade the cuda version using only OMP.
- Comparative analysis of GPU vs CPU performance for various matrix sizes and also cuda vs omp.

### How to Run Commands for Assignment 3

#### Question 1: Omp gpu code
```bash
nano question1_a_gpu_omp.c
module load nvhpc-hpcx-cuda12/23.9
nvc -O3 -mp=gpu question1_a_gpu_omp.c -o question1_a_gpu_omp
./question1_a_gpu_omp
```

#### Question 1: Serial cpu code
```bash
nano question2_serial_cpu_code.c
gcc -O3 question2_serial_cpu_code.c -o question2_serial_cpu_code
./question2_serial_cpu_code
```

#### Question 1: Omp cpu code
```bash
nano question2_omp_cpu.c
gcc -O3 -fopenmp question2_omp_cpu.c -o question2_omp_cpu
./question2_omp_cpu
```


---


