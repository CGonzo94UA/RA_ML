#ifndef __LAYER_H__
#define __LAYER_H__

#include <vector>
#include <iostream>
#include <cmath>
#include "functions.h"

using namespace std;

class NeuralNetworkLayer{
    private:
        vector<vector<double>> _weights;
        double (*activationFunction)(double) = ActivationFunctions::sigmoid;

        // double activationFunction(double const& x) const;

    public:
        NeuralNetworkLayer(size_t const& width_of_layer, size_t const& num_weights); 
        ~NeuralNetworkLayer() {}

        vector<vector<double>> weights() const { return _weights; }

        vector<vector<double>> generateRandomWeights(size_t const& width_of_layer, size_t const& num_weights) const;
        vector<double> feedForward(vector<double> const& inputs) const;

        void setActivationFunc(double (*f)(double)) { activationFunction = f; }
};

#endif