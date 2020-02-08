#include <sys/time.h>

#define CHECK(call)                                                        \
{                                                                          \
    const cudaError_t error = call;                                        \
    if(error != cudaSuccess)                                               \
    {                                                                      \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                      \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(1);                                                           \
    }                                                                      \
}

void initData(float *A, const int N)
{
    time_t t;
    srand((unsigned int)time(&t));
    for(int i = 0; i < N; ++i)
        A[i] = (float)(rand() & 0xFF) / 10.0F;
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
    return (double)tp.tv_sec + (double)tp.tv_usec*1.e-6;
}
