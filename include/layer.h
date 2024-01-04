#ifndef __LAYER_H__
#define __LAYER_H__

#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

class NeuralNetworkLayer{
    private:
        vector<vector<double>> _weights;

        double activationFunction(double const& x) const;

    public:
        NeuralNetworkLayer(size_t const& width_of_layer, size_t const& num_weights){} 
        ~NeuralNetworkLayer() {}

        vector<vector<double>> weights() const { return _weights; }

        vector<vector<double>> generateRandomWeights(size_t const& width_of_layer, size_t const& num_weights) const;
        vector<double> NeuralNetworkLayer::feedForward(vector<double> const& inputs) const;
    
};

#endif