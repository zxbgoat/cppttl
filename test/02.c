#include <stdio.h>
#define MAXN 10001
char str[MAXN];
char rst[MAXN];

int main()
{
    int i = 0, j = 0, N;
    char c;
    while(1)
    {
        c = getchar();
        if(c == '\n')
            break;
        str[i] = c;
        i += 1;
    }
    N = i;
    i = 0;
    while(i < N)
    {
        if(str[i] == ' ')
        {
            if(i > 0 && str[i-1] != ' ')
            {
                rst[j] = ',';
                j += 1;
            }
        }
        else
        {
            rst[j] = str[i];
            j += 1;
        }
        i += 1;
    }
    for(i = 0; i < j; ++i)
        putchar(rst[i]);
    return 0;
}
