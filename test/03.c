#include <stdio.h>
#define MAXN 100001
char exp[MAXN];
char stk[MAXN];
int top = 0;

void push(char c)
{
    stk[top] = c;
    top += 1;
}

char pop()
{
    top -= 1;
    return stk[top];
}

int main()
{
    char c;
    while(1)
    {
        c = getchar();
        if(c == '\n')
            break;
        if('0' <= c && c <= '9')
            push(c);

    }
}
