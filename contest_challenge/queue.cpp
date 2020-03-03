#include <queue>
#include <cstdio>
using namespace std;

int main()
{
    queue<int> q;
    q.push(1);
    q.push(2);
    q.push(3);
    printf("%d\n", q.front());
    q.pop();
    printf("%d\n", q.front());
    q.pop();
    printf("%d\n", q.front());
    q.pop();
    return 0;
}
