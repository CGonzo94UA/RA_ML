#ifndef __LAYER_H__
#define __LAYER_H__

#include <vector>
#include <iostream>
#include <cmath>
#include "functions.h"
#include "matrix.h"

using namespace std;

class NeuralNetworkLayer{
    private:
        Matrix _weights;
        double (*activationFunction)(double) = ActivationFunctions::sigmoid;

        // double activationFunction(double const& x) const;

    public:
        NeuralNetworkLayer(size_t const& width_of_layer, size_t const& num_weights); 
        NeuralNetworkLayer(size_t const& width_of_layer, size_t const& num_weights, double (*f)(double)); 
        ~NeuralNetworkLayer() {}

        Matrix weights() const { return _weights; }

        void generateRandomWeights(size_t const& width_of_layer, size_t const& num_weights);
        vector<double> feedForward(vector<double> const& inputs) const;

        void setActivationFunc(double (*f)(double)) { activationFunction = f; }
};

#endif