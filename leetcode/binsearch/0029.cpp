#include <iostream>
using namespace std;

int divid(int divd, int dior)
{
    int res = 0;
    bool flag = (divd ^ dior) < 0;
    divd = abs(divd);
    dior = abs(dior);
    while(divd >= dior)
    {
        int tmp = dior, i = 1;
        while(divd >= tmp)
        {
            res += i;
            divd -= tmp;
            i <<= 1;
            tmp <<= 1;
        }
    }
    if(res)
        res = -res;
    return res;
}

int main()  // compiling via -lstdc++ on Mac
{
    int divd, dior;
    cin >> divd >> dior;
    int res = divid(divd, dior);
    cout << res << endl;
    return 0;
}
