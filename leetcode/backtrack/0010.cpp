#include <iostream>
#include <string>
#include <vector>
using namespace std;

bool isMatch(string s, string p)
{
    if(p.empty())
        return s.empty();
    bool firstmatch = !s.empty() && (p[0]==s[0] || p[0]=='.');
    if(p.size()>=2 && p[1]=='*')
        return isMatch(s, )
}

int main()
{
    string s, p;
    cin >> s >> p;
    cout << isMatch(s, p) << endl;
    return 0;
}
