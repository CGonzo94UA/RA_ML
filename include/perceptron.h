#ifndef __PERCEPTRON_H__
#define __PERCEPTRON_H__

#include "matrix.h"
#include <vector>

class Perceptron{
    public:
        Perceptron(std::size_t const num_weights);
	    ~Perceptron() {}

        Matrix weights() const { return _weights; }
        double dot(const vector<double>&, const vector<double>&);
        //void train_(Matrix const& X, std::vector<double> const& Y, std::size_t const maxiter);
        void train(Matrix const& X, Matrix const& Y, std::size_t const maxiter);
        double classify(const Matrix&);

    private:
        Matrix _weights;
        Matrix generateRandomWeights(std::size_t const num_weights);

};

#endif