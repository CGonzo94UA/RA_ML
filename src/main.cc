#include "matrix.h"
#include "perceptron.h"
#include <iostream>

using namespace std;

int main() {
    //Matrix m1(2, 2, {{1, 2}, {3, 4}});
    
    //cout << m1 << endl;

    Matrix and_x(4, 3, {{1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}});
    Matrix and_y = {4, 1, {-1, -1, -1, 1}};

    std::cout << "X: " << and_x << "\n";
    std::cout << "Y: " << and_y << "\n";

    Perceptron perceptron(and_x.cols());

    std::cout << "Initial weights: " << perceptron.weights() << "\n";
    

    perceptron.train(and_x, and_y, 1000);

    std::cout <<  "Final weigths: " << perceptron.weights() << "\n";

    Matrix test(1, 3, {1, 1, 1});
    std::cout << "Classify: " << perceptron.classify(test) << "\n";

    Matrix test2(1, 3, {1, 0, 1});
    std::cout << "Classify: " << perceptron.classify(test2) << "\n";

    return 0;
}