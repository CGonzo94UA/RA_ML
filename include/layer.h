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
        vector<double> _outputs;
        vector<double> _signal;
        // double activationFunction(double const& x) const;

    public:
        double (*activationFunction)(double) = ActivationFunctions::sigmoid;
        NeuralNetworkLayer(std::size_t const& width_of_layer, std::size_t const& num_weights); 
        NeuralNetworkLayer(std::size_t const& width_of_layer, std::size_t const& num_weights, double (*f)(double)); 
        ~NeuralNetworkLayer() {}

        Matrix weights() const { return _weights; }
        vector<double> getGradients() const;
        vector<double> getOutputs() const;
        vector<double> getSignals() const;
        vector<double> calculateGradients(const vector<double>& output, const vector<double>& target);
        vector<double> calculateGradientsMedio(const vector<double>& nextLayerGradients, const Matrix& weights);
        double activationFunctionDerivative(double x);
        void updateWeights(const double learningRate, const vector<double>&);
        void setOutputs(const vector<double>& inputs);
        void setWeights(const Matrix& weights) { _weights = weights; }
        void setSignals(const vector<double>& signals);

        void generateRandomWeights(std::size_t const& width_of_layer, std::size_t const& num_weights);
        std::vector<double> feedForward(const vector<double>& output) ;

        void setActivationFunc(double (*f)(double)) { activationFunction = f; }
};

#endif