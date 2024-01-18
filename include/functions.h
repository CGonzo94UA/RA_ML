#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

#include <cmath>
#include <algorithm>

namespace ActivationFunctions{
    // Sigmoid
    inline double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    // Derivative of sigmoid
    inline double sigmoid_prime(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    // ReLU
    inline double relu(double x) {
        return std::max(0.0, x);
    }

    // Derivative of ReLU
    inline double relu_prime(double x) {
        return x > 0 ? 1 : 0;
    }

    inline double sign(double x){
        if (x >= 0)
            return 1.0;
        else
            return -1.0;
    }

    inline double binary(double x) {
        return x < 0.5 ? -1.0 : 1.0;
    }

    inline double tanh(double x) {
        return tanh(x);
    }

}

#endif