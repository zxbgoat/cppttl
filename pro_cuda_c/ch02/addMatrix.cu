#include <stdio.h>
#include "common.h"

void addMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;
    for(int iy = 0; iy < ny; ++iy)
    {
        for(int ix = 0; ix < nx; ++ix)
            ic[ix] = ia[ix] + ib[ix];
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void addMatrixOnGPU2D(float *A, float *B, float *C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if(ix < nx && iy < ny)
        C[idx] = A[idx] + B[idx];
}

__global__ void addMatrixOnGPU1D(float *A, float *B, float *C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if(ix < nx)
        for(int iy = 0; iy < ny; ++iy)
        {
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
}

__global__ void addMatrixOnGPUMD(float *A, float *B, float *C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y;
    unsigned int idx = iy * nx + ix;
    if(ix < nx && iy < ny)
        C[idx] = A[idx] + B[idx];
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // 设置设备
    int dev = 0;
    cudaDeviceProp devProp;
    CHECK(cudaGetDeviceProperties(&devProp, dev));
    printf("Using Device %d: %s\n", dev, devProp.name);
    CHECK(cudaSetDevice(dev));

    // 设置矩阵大小
    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: %dx%d", nx, ny);

    // 设置主机内存
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float*)malloc(nBytes);
    h_B     = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef  = (float*)malloc(nBytes);
    initData(h_A, nxy);
    initData(h_B, nxy);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    addMatrixOnHost(h_A, h_B, hostRef, nx, ny);

    // 设置设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // 运行核函数
    double iStart, iElaps;
    int choice = atoi(argv[1]);
    // 二维x二维线程组织形式
    if(choice == 0)
    {
        int dimx = atoi(argv[2]);
        int dimy = atoi(argv[3]);
        dim3 block(dimx, dimy);
        dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
        iStart = cpuSecond();
        addMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
        cudaDeviceSynchronize();
        iElaps = cpuSecond() - iStart;
        printf("Execute the 2Dx2D layout of threads.\n");
        printf("addMatrixOnGPU <<<(%d,%d), (%d,%d)>>> Time elapsed %f seconds\n",
               grid.x, grid.y, block.x, block.y, iElaps);
    }
    // 一维x一维线程组织形式
    else if(choice == 1)
    {
        int dim = atoi(argv[2]);
        dim3 block(dim);
        dim3 grid((nx+block.x-1)/block.x);
        iStart = cpuSecond();
        addMatrixOnGPU1D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
        cudaDeviceSynchronize();
        iElaps = cpuSecond() - iStart;
        printf("Execute the 1Dx1D layout of threads.\n");
        printf("addMatrixOnGPU <<<%d, %d>>> Time elapsed %f seconds\n",
               grid.x, block.x, iElaps);
    }
    // 二维x一维线程组织形式
    else
    {
        int dim = atoi(argv[2]);
        dim3 block(dim);
        dim3 grid((nx+block.x-1)/block.x, ny);
        iStart = cpuSecond();
        addMatrixOnGPUMD<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
        cudaDeviceSynchronize();
        iElaps = cpuSecond() - iStart;
        printf("Execute the 1Dx1D layout of threads.\n");
        printf("addMatrixOnGPU <<<(%d,%d), %d>>> Time elapsed %f seconds\n",
               grid.x, grid.y, block.x, iElaps);
    }

    // 验证结果
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nxy);
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // 释放主机内存
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
