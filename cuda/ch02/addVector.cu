#include <stdio.h>

void initData(float *A, const int N)
{
    time_t t;
    srand((unsigned int)time(&t));
    for(int i = 0; i < N; ++i)
        A[i] = (float)(rand() & 0xFF) / 10.0F;
}

__global__ void addVectorsOnGPU(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

void printVector(float *A, const int N)
{
    for(int i = 0; i < N; ++i)
        printf("%6.1f", A[i]);
    printf("\n");
}

int main()
{
    // set the device
    int dev = 0;
    cudaSetDevice(dev);

    // set the data size
    int nElem = 32;
    int nBytes = nElem * sizeof(float);

    // set the host memory
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    h_C = (float*)malloc(nBytes);
    initData(h_A, nElem);
    initData(h_B, nElem);
    memset(h_C, 0, nBytes);

    // set the device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((double**)&d_A, nBytes);
    cudaMalloc((double**)&d_B, nBytes);
    cudaMalloc((double**)&d_C, nBytes);

    // copy data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // lauch the kernel
    dim3 block(nElem);
    dim3 grid((nElem+block.x-1) / block.x);
    addVectorsOnGPU<<<grid, block>>>(d_A, d_B, d_C);

    // copy data from device to host
    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

    printf("Vector A: ");
    printVector(h_A, nElem);
    printf("Vector B: ");
    printVector(h_B, nElem);
    printf("Their sum: ");
    printVector(h_C, nElem);

    // free the device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free the host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
