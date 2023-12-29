#include "matrix.h"
#include "perceptron.h"
#include <iostream>
#include <vector>
#include <sstream>

// #define DIVIDE_DATASET

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

std::pair<Matrix, Matrix> divideDataset(Matrix const& m, double const trainRatio) {
    size_t const trainSize = static_cast<size_t>(m.rows() * trainRatio);
    size_t const testSize = m.rows() - trainSize;

    Matrix train(trainSize, m.cols());
    Matrix test(testSize, m.cols());

    for (size_t i = 0; i < trainSize; ++i) {
        train[i] = m[i];
    }

    for (size_t i = 0; i < testSize; ++i) {
        test[i] = m[i + trainSize];
    }

    return {train, test};
}

int main() {
    auto [X, Y] = Perceptron::readFromCSV("datasets/128entradas.csv");
    int maxiter = 1000;

    #ifdef DIVIDE_DATASET
        double trainRatio = 0.8;
        auto [Xtrain, Xtest] = divideDataset(X, trainRatio);
        auto [Ytrain, Ytest] = divideDataset(Y, trainRatio);

        Perceptron perceptron(Xtrain.cols());

        // std::cout << "Initial weights: " << perceptron.weights() << "\n";
        
        perceptron.train(Xtrain, Ytrain, maxiter);

        // std::cout <<  "Final weigths: " << printVector(perceptron.weights().getCol(0)) << "\n";
        std::cout << "Accuracy: " << perceptron.test(Xtest, Ytest) << "\n";
    #else
        Perceptron perceptron(X.cols());
        perceptron.train(X, Y, maxiter);
        std::cout << "Accuracy: " << perceptron.test(X, Y) << "\n";
    #endif
    // Matrix test(1, 3, {1, 1, 1});
    // std::cout << "Classify: " << perceptron.classify(test) << "\n";

    return 0;
}