#include <stdio.h>
#include <sys/time.h>

#define CHECK(call)                                                      \
{                                                                        \
    const cudaError_t error = call;                                      \
    if(error != cudaSuccess)                                             \
    {                                                                    \
        printf("Error %s:%d", __FILE__, __LINE__);                       \
        printf("code: %d, reason: %s", error, cudaGetErrorString(error));\
        exit(1);                                                         \
    }                                                                    \
}

void initData(float *A, const int N)
{
    time_t t;
    srand((unsigned int)time(&t));
    for(int i = 0; i < N; ++i)
        A[i] = (float)(rand() & 0xFF) / 10.0L;
}

void addVectorsOnHost(float *A, float *B, float*C, const int N)
{
    for(int i = 0; i < N; ++i)
        C[i] = A[i] + B[i];
}

__global__ void addVectorsOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N)
        C[i] = A[i] + B[i];
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    float epsilon = 1e-9;
    int match = 1;
    for(int i = 0; i < N; ++i)
        if(abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("Results do not match !\n");
            match = 0;
            break;
        }
    if(match)
        printf("Results Match !\n");
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);
    int iLen = atoi(argv[1]);
    printf("the block dimension is %d\n", iLen);

    // set up the device
    int dev = 0;
    cudaDeviceProp devProp;
    CHECK(cudaGetDeviceProperties(&devProp, dev));
    printf("Using Device %d: %s\n", dev, devProp.name);
    CHECK(cudaSetDevice(dev));

    // set the data size
    int nElem = 1 << 24;
    size_t nBytes = nElem * sizeof(float);

    // set the host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float*)malloc(nBytes);
    h_B = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);
    initData(h_A, nElem);
    initData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    addVectorsOnHost(h_A, h_B, hostRef, nElem);

    // set the device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // lauch the kernel
    double iStart, iElaps;
    dim3 block(iLen);
    dim3 grid((nElem+block.x-1) / block.x);
    iStart = cpuSecond();
    addVectorsOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("addVectorsOnGPU <<<%d, %d>>> Time elapsed %f seconds\n",
           grid.x, block.x, iElaps);

    // transfer data from host to devcie
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem);

    // free the devcie memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free the host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
