#ifndef __PERCEPTRON_H__
#define __PERCEPTRON_H__

#include "matrix.h"
#include <vector>

class Perceptron{
    public:
        Perceptron(std::size_t const num_weights);
        Perceptron(Matrix const& weights) : _weights(weights) {}
	    ~Perceptron() {}

        Matrix weights() const { return _weights; }
        void train(Matrix const& X, Matrix const& Y, std::size_t const maxiter);
        double classify(const Matrix&);
        double test(Matrix const& X, Matrix const& Y) const;

    private:
        Matrix _weights;
        Matrix generateRandomWeights(std::size_t const num_weights);

};

#endif