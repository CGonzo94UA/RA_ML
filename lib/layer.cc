#include "layer.h"

// Constructor for the layer, generates random weights for each neuron (except the bias, which has a value of 1)
NeuralNetworkLayer::NeuralNetworkLayer(size_t const& width_of_layer, size_t const& num_weights) {
    _weights = generateRandomWeights(width_of_layer, num_weights);
}

// Activation function for the neuron
// double NeuralNetworkLayer::activationFunction(double const& x) const {
//     return 1 / (1 + exp(-x));       // Sigmoid function
// }

// Randomly initiate weights of each neuron in the layer, adding one for the bias as the first element
vector<vector<double>> NeuralNetworkLayer::generateRandomWeights(size_t const& width_of_layer, size_t const& num_weights) const{

    // Randomly initiate weights of each neuron in the layer, adding one for the bias

    vector<vector<double>> weights;
    for (size_t i = 0; i < width_of_layer; i++) {
        vector<double> neuron_weights;
        neuron_weights.push_back(1.0);      // Bias
        for (size_t j = 0; j < num_weights; j++) {
            neuron_weights.push_back((double)rand() / RAND_MAX);
        }
        weights.push_back(neuron_weights);
    }

    return weights;
}

// Applies the activation function to the dot product of the weights and the inputs
vector<double> NeuralNetworkLayer::feedForward(vector<double> const& inputs) const {
    vector<double> outputs;
    for (size_t i = 0; i < _weights.size(); i++) {
        double output = 0;
        for (size_t j = 0; j < _weights[i].size(); j++) {
            output += _weights[i][j] * inputs[j];
        }
        outputs.push_back(activationFunction(output));
    }
    return outputs;
}