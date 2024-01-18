#ifndef __LAYER_H__
#define __LAYER_H__

#include <vector>
#include <iostream>
#include <cmath>
#include "functions.h"
#include "matrix.h"
#include <random>
#include <cassert>

using namespace std;

class NeuralNetworkLayer{
    private:
        Matrix _weights;
        vector<double> _gradients;
        vector<double> _inputs;
        // double activationFunction(double const& x) const;
        double (*activationFunction)(double) = ActivationFunctions::sigmoid;

    public:
        NeuralNetworkLayer(std::size_t const& width_of_layer, std::size_t const& num_weights); 
        NeuralNetworkLayer(std::size_t const& width_of_layer, std::size_t const& num_weights, double (*f)(double)); 
        ~NeuralNetworkLayer() {}

        Matrix weights() const { return _weights; }
        vector<double> getGradients() const;
        vector<double> getInputs() const;
        // Get activation function
        double (*getActivationFunction())(double) { return activationFunction; }
        vector<double> calculateGradients(const vector<double>& output, const vector<double>& target);
        vector<double> calculateGradientsMedio(const vector<double>& nextLayerGradients, const vector<double>& outputs);
        double activationFunctionDerivative(double x);
        void updateWeights(double learningRate);
        void setInputs(const vector<double>& inputs);
        void setWeights(const Matrix& weights) { _weights = weights; }

        void generateRandomWeights(std::size_t const& width_of_layer, std::size_t const& num_weights);
        std::vector<double> feedForward() const;

        void setActivationFunc(double (*f)(double)) { activationFunction = f; }
};

#endif