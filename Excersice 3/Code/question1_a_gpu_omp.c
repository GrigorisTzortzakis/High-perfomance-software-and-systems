#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#define N 2048

// Total run time
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}
// Random matrix values
void initializeMatrix(float* mat, int n) {
    for (int i = 0; i < n * n; ++i) {
        mat[i] = (float)rand() / (float)RAND_MAX;
    }
}
// Main computation function for GPU
void complexMatMulKernel(const float* A, const float* B, const float* C, const float* D, 
                        float* E, float* F, int n) {
    // Using OpenMP to offload to GPU
    #pragma omp target teams distribute parallel for collapse(2) \
    map(to:A[0:n*n], B[0:n*n], C[0:n*n], D[0:n*n]) map(from:E[0:n*n], F[0:n*n])
    // Parallel loops replacing CUDA thread blocks
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            // Initialize to zero before calculations
            float sumAC = 0.0f;
            float sumBD = 0.0f;
            float sumAD = 0.0f;
            float sumBC = 0.0f;
            // Matrix multiplication
            for (int k = 0; k < n; k++) {
                // retrieve values
                float a = A[row * n + k];
                float b = B[row * n + k];
                float c = C[k * n + col];
                float d = D[k * n + col];
                // Calculate A,B,C,D
                sumAC += a * c;
                sumBD += b * d;
                sumAD += a * d;
                sumBC += b * c;
            }
            // Calculate and store in E,F
            E[row * n + col] = sumAC - sumBD;
            F[row * n + col] = sumAD + sumBC;
        }
    }
}
int main() {
    double start_time = get_time();
    
    srand(time(NULL));
    size_t size = N * N * sizeof(float);
    
    // Allocate host memory
    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C = (float*)malloc(size);
    float* D = (float*)malloc(size);
    float* E = (float*)malloc(size);
    float* F = (float*)malloc(size);
    if (!A || !B || !C || !D || !E || !F) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    // Initialize matrices with random values
    initializeMatrix(A, N);
    initializeMatrix(B, N);
    initializeMatrix(C, N);
    initializeMatrix(D, N);

    // Run the kernel (OpenMP will handle GPU offload)
    complexMatMulKernel(A, B, C, D, E, F, N);
    
    // Free memory
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
    free(F);
    
    double end_time = get_time();
    printf("Matrix size: %d x %d\n", N, N);
    printf("Total execution time: %.4f seconds\n", (end_time - start_time) / 1000000.0);
    return 0;
}