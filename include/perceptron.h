#ifndef __PERCEPTRON_H__
#define __PERCEPTRON_H__

#include "matrix.h"
#include <vector>

class Perceptron{
    public:
        Perceptron(std::size_t const num_weights){}
	    ~Perceptron() {}

        double sign(double const x){}
        void train(Matrix const& X, Matrix const& Y, std::size_t const maxiter){}

    private:
        Matrix _weights;
        double _bias = 1.0;
        Matrix generateRandomWeights(std::size_t const num_weights){}

};

#endif