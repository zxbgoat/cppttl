#include <cstdio>
using namespace std;

#define MAX_N 10000
int heap[MAX_N], sz = 0

void push(int x)
{
    int i = sz++;
    while(i > 0)
    {
        int p = (i-1) / 2;
        if(heap[p] <= x)
            break;
        heap[i] = heap[p];
        i = p;
    }
    heap[i] = x;
}

int pop()
{
    int rst = heap[0];
}
