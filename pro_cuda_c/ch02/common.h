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

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec*1.e-6;
}
