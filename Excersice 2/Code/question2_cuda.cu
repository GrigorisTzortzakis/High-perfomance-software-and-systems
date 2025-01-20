#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define N 1024

#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                               \
    if(e!=cudaSuccess) {                                            \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__,        \
                cudaGetErrorString(e));                             \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

__global__ void complexMatMulKernel(
    const float* A, const float* B,
    const float* C, const float* D,
    float* E, float* F,
    int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sumAC = 0.0f;
        float sumBD = 0.0f;
        float sumAD = 0.0f;
        float sumBC = 0.0f;

        for (int k = 0; k < n; ++k)
        {
            float a = A[row * n + k];
            float b = B[row * n + k];
            float c = C[k * n + col];
            float d = D[k * n + col];

            sumAC += a * c;
            sumBD += b * d;
            sumAD += a * d;
            sumBC += b * c;
        }

        E[row * n + col] = sumAC - sumBD;
        F[row * n + col] = sumAD + sumBC;
    }
}

void initializeMatrix(float* mat, int n)
{
    for (int i = 0; i < n * n; ++i)
    {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main()
{
    double start_time = get_time();
    
    srand(time(NULL));
    size_t size = N * N * sizeof(float);
    
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    float* h_D = (float*)malloc(size);
    float* h_E = (float*)malloc(size);
    float* h_F = (float*)malloc(size);

    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);
    initializeMatrix(h_C, N);
    initializeMatrix(h_D, N);

    float *d_A, *d_B, *d_C, *d_D, *d_E, *d_F;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMalloc((void**)&d_D, size);
    cudaMalloc((void**)&d_E, size);
    cudaMalloc((void**)&d_F, size);
    cudaCheckError();

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, size, cudaMemcpyHostToDevice);
    cudaCheckError();

    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (N + dimBlock.y - 1) / dimBlock.y);

    complexMatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, d_D, d_E, d_F, N);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();

    cudaMemcpy(h_E, d_E, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_F, d_F, size, cudaMemcpyDeviceToHost);
    cudaCheckError();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_E);
    cudaFree(d_F);
    cudaCheckError();

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_E);
    free(h_F);

    double end_time = get_time();
    printf("Matrix size: %d x %d\n", N, N);
    printf("Total execution time: %.4f seconds\n", (end_time - start_time) / 1000000.0);

    return 0;
}