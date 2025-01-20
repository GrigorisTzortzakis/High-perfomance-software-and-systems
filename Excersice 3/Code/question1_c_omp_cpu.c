#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define N 4096

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

void initializeMatrix(float* mat, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n * n; ++i) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

void complex_matrix_multiply(const float* A, const float* B, const float* C, const float* D, 
                           float* E, float* F, int n) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sumAC = 0.0f;
            float sumBD = 0.0f;
            float sumAD = 0.0f;
            float sumBC = 0.0f;
            
            for (int k = 0; k < n; k++) {
                float a = A[i * n + k];
                float b = B[i * n + k];
                float c = C[k * n + j];
                float d = D[k * n + j];
                sumAC += a * c;
                sumBD += b * d;
                sumAD += a * d;
                sumBC += b * c;
            }
            
            E[i * n + j] = sumAC - sumBD;
            F[i * n + j] = sumAD + sumBC;
        }
    }
}

int main() {
    omp_set_num_threads(omp_get_max_threads());
    printf("Running with %d OpenMP threads\n", omp_get_max_threads());

    srand(time(NULL));
    size_t size = N * N * sizeof(float);
    
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

    initializeMatrix(A, N);
    initializeMatrix(B, N);
    initializeMatrix(C, N);
    initializeMatrix(D, N);

    printf("Starting computation for %dx%d matrices...\n", N, N);
    double start_time = get_time();
    
    complex_matrix_multiply(A, B, C, D, E, F, N);
    
    double end_time = get_time();
    double execution_time = (end_time - start_time) / 1000000.0;
    
    printf("Matrix size: %d x %d\n", N, N);
    printf("OpenMP CPU Execution time: %.4f seconds\n", execution_time);

    free(A); free(B); free(C); free(D); free(E); free(F);
    return 0;
}