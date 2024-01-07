#include <cmath>
#include <algorithm>

namespace ActivationFunctions{
    // Sigmoid
    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    // Derivative of sigmoid
    double sigmoid_prime(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    // ReLU
    double relu(double x) {
        return std::max(0.0, x);
    }

    // Derivative of ReLU
    double relu_prime(double x) {
        return x > 0 ? 1 : 0;
    }

}