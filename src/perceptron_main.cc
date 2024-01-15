#include "matrix.h"
#include "perceptron.h"
#include <iostream>
#include <vector>
#include <sstream>

using namespace std;

template <typename T>
std::string printVector(std::vector<T> const& v) {
    std::stringstream ss;
    ss << "[";
    for (auto const& e : v) {
        ss << e << ", ";
    }
    ss << "]";
    return ss.str();
}

int main(){
    auto [X, Y] = Perceptron::readFromCSV("datasets/2entradas.csv");
    int maxiter = 100;

    Perceptron perceptron(X.cols());
    perceptron.train(X, Y, maxiter);
    std::cout << "Accuracy: " << perceptron.test(X, Y) << "\n";
    
    return 0;
}
