#ifndef __LAYER_H__
#define __LAYER_H__

#include <vector>
#include <iostream>
#include <cmath>
#include "functions.h"
#include "matrix.h"

class NeuralNetworkLayer{
    private:
        Matrix _weights;
        double (*activationFunction)(double) = ActivationFunctions::sigmoid;

        // double activationFunction(double const& x) const;

    public:
        NeuralNetworkLayer(std::size_t const& width_of_layer, std::size_t const& num_weights); 
        NeuralNetworkLayer(std::size_t const& width_of_layer, std::size_t const& num_weights, double (*f)(double)); 
        ~NeuralNetworkLayer() {}

        Matrix weights() const { return _weights; }

        void generateRandomWeights(std::size_t const& width_of_layer, std::size_t const& num_weights);
        std::vector<double> feedForward(std::vector<double> const& inputs) const;

        void setActivationFunc(double (*f)(double)) { activationFunction = f; }
};

#endif