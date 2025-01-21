# High-Performance Computing (HPC) Course Assignments

## Overview

This repository contains solutions and implementations for the **High-Performance Computing (HPC)** course at the **Department of Computer Engineering and Informatics (CEID)** for the **2024-2025 academic year**. The tasks involve MPI, OpenMP, SIMD, and CUDA programming techniques to solve computationally intensive problems. Each assignment explores different aspects of parallel programming and optimization for high-performance systems.

---

## Assignments

### 1. MPI and OpenMP
**Topics Covered:**
- Implementation of `MPI_Exscan` using point-to-point communication.
- Extension for hybrid programming with MPI and OpenMP.
- Parallel I/O with MPI and 3D matrix operations using random seeds.
- Data compression using libraries like Zlib, ZFP, and SZ.
- **Parallel Parametric Search in Machine Learning:**
  - Grid search for hyperparameter tuning.
  - Parallelization of grid search using:
    - Python multiprocessing pool.
    - MPI Futures.
    - Master-worker model with MPI.

**Key Highlights:**
- **Custom MPI Implementations:** Created a `MPI_Exscan` operation using point-to-point communication for efficient prefix sums and extended it to support hybrid MPI and OpenMP models.
- **Parallel I/O:** Designed a mechanism for all threads in a hybrid environment to write independent 3D matrices to a binary file and validated data consistency through parallel and serial reads.
- **Compression Techniques:** Implemented compression for 3D matrices using libraries like Zlib and ZFP, enabling efficient storage of variable-sized compressed data blocks.
- **Machine Learning Grid Search:** Parallelized hyperparameter optimization for machine learning models using multiple approaches:
  - Python multiprocessing for task-level parallelism.
  - MPI futures for distributed computing.
  - A master-worker model leveraging MPI to distribute hyperparameter combinations.

Relevant files:
- `mpi_exscan.c`
- `mpi_openmp_hybrid.c`
- `mpi_io_compression.c`
- `mlp_parametric_search.py`

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
