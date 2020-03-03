#include <cstdio>
using namespace std;

const int MAX_N = 200;
int n, W;
int w[MAX_N], v[MAX_N];

int max(int a, int b)
{
    if(a >= b)
        return a;
    else
        return b;
}

int rec(int i, int j)
{
    int res;
    if(i == n)
        res = 0;
    else if(j < w[i])
        res = rec(i+1, j);
    else
        res = max(rec(i+1, j), rec(i+1, j-w[i])+v[i]);
    return res;
}

int main()
{
    scanf("%d%d", &n, &W);
    for(int i = 0; i< n; ++i)
        scanf("%d", &w[i]);
    for(int i = 0; i< n; ++i)
        scanf("%d", &v[i]);
    printf("%d\n", rec(0, W));
}
