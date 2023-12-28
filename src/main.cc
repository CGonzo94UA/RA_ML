#include "matrix.h"
#include <iostream>

using namespace std;

int main() {
    Matrix m1(2, 2, {{1, 2}, {3, 4}});
    
    cout << m1 << endl;

    return 0;
}