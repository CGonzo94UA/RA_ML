#ifndef __PERCEPTRON_H__
#define __PERCEPTRON_H__

#include "weakLearner.h"
#include "matrix.h"

class Perceptron: public WeakLearner{
    public:
        Perceptron(std::size_t const num_weights);
        Perceptron(Matrix const& weights) : _weights(weights) {}
	    ~Perceptron() {}

        Matrix weights() const override { return _weights; }
        void train(Matrix const& X, Matrix const& Y, std::size_t const maxiter) override;
        double classify(const Matrix&) override;
        Matrix classify_all(const Matrix&) const override;
        static std::pair<Matrix, Matrix> readFromCSV(std::string const& filename);
        double test(Matrix const& X, Matrix const& Y) const override;

    private:
        Matrix _weights;
        Matrix generateRandomWeights(std::size_t const num_weights);

};

#endif