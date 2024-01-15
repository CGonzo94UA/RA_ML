#include "matrix.h"
#include "perceptron.h"
#include <iostream>
#include <vector>
#include <sstream>

// #define DIVIDE_DATASET
#define KFOLDS 10

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
    auto [X, Y] = Perceptron::readFromCSV("datasets/128entradas.csv");
    int maxiter = 1000;

    #ifdef DIVIDE_DATASET
        double trainRatio = 0.8;
        auto [Xtrain, Xtest] = X.divide(trainRatio);
        auto [Ytrain, Ytest] = Y.divide(trainRatio);
        Perceptron perceptron(Xtrain.cols());
        // std::cout << "Initial weights: " << perceptron.weights() << "\n";
        
        perceptron.train(Xtrain, Ytrain, maxiter);
        // std::cout <<  "Final weigths: " << printVector(perceptron.weights().getCol(0)) << "\n";
        std::cout << "Accuracy: " << perceptron.test(Xtest, Ytest) << "\n";
    #else
        #ifdef KFOLDS
            vector<int> folds = X.kfold(KFOLDS);
            double max_accuracy = 0;
            for (int i = 0; i < KFOLDS; ++i) {
                std::cout << "========== FOLD " << i << " ==========" << std::endl;
                auto [Xtrain, Xtest] = X.getFold(folds, KFOLDS, i);
                auto [Ytrain, Ytest] = Y.getFold(folds, KFOLDS, i);

                Perceptron perceptron(Xtrain.cols());
                perceptron.train(Xtrain, Ytrain, maxiter);
                
                double accuracy = perceptron.test(Xtest, Ytest);
                std::cout << "++Accuracy fold " << i << ": " << accuracy << "\n\n";
                max_accuracy = max(max_accuracy, accuracy);
            }
            std::cout << "Max accuracy: " << max_accuracy << "\n";
        #else
            Perceptron perceptron(X.cols());
            perceptron.train(X, Y, maxiter);
            std::cout << "Accuracy: " << perceptron.test(X, Y) << "\n";
        #endif
    #endif
    // Matrix test(1, 3, {1, 1, 1});
    // std::cout << "Classify: " << perceptron.classify(test) << "\n";
    return 0;
}
