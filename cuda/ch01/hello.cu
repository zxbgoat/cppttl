/*
compile the program as:
nvcc -arch sm_75 hello.cu -o hello
其中sm_后面的数字随着显卡架构不同而不同
75对应的是Turing架构
*/

#include <stdio.h>

__global__ void helloFromGPU()
{
    if(threadIdx.x == 5)
        printf("Hello World from GPU !\n");
}

int main()
{
	printf("Hello World from CPU !\n");
    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset();
    // cudaDeviceSynchronize();
    return 0;
}
