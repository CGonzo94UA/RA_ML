#include "layer.h"
#include "randonn_generator.h"
#include "utils.h"

using namespace std;

// Constructor for the layer, generates random weights for each neuron (except the bias, which has a value of 1)
NeuralNetworkLayer::NeuralNetworkLayer(size_t const& width_of_layer, size_t const& num_weights)
: _gradients(width_of_layer, 0.0) {
    _weights = Matrix(width_of_layer, num_weights+1);       // +1 for the bias wich is the first element of each neuron in the layer and has a value of 1
    generateRandomWeights(width_of_layer, num_weights);
}

// Constructor for the layer, generates random weights for each neuron (except the bias, which has a value of 1) and sets the activation function to the one passed as an argument
NeuralNetworkLayer::NeuralNetworkLayer(size_t const& width_of_layer, size_t const& num_weights, double (*f)(double))
: _gradients(width_of_layer, 0.0) {
    _weights = Matrix(width_of_layer, num_weights+1);       // +1 for the bias wich is the first element of each neuron in the layer and has a value of 1
    generateRandomWeights(width_of_layer, num_weights);
    activationFunction = f;
}

// Activation function for the neuron
// double NeuralNetworkLayer::activationFunction(double const& x) const {
//     return 1 / (1 + exp(-x));       // Sigmoid function
// }

// Randomly initiate weights of each neuron in the layer, adding one for the bias as the first element
void NeuralNetworkLayer::generateRandomWeights(size_t const& width_of_layer, size_t const& num_weights){

    // Randomly initiate weights of each neuron in the layer, adding one for the bias
    Randonn_generator generator;

    // For each neuron in the layer (width_of_layer) generate num_weights+1 random weights (num_weights + 1 for the bias) where the bias is the first element of each neuron and has a value of 1
    for (size_t i = 0; i < width_of_layer; i++) {
        for (size_t j = 0; j < num_weights+1; j++) {
            if (j == 0) {
                _weights[i][j] = 1.0;
            } else {
                _weights[i][j] = generator.randomDouble(-0.5, 0.5);
            }
        }
    }

}

// Applies the activation function to the dot product of the weights and the inputs
vector<double> NeuralNetworkLayer::feedForward(const vector<double>& output_aux)  {

    //Checks if there is an activation function
    assert(activationFunction != nullptr && "The layer must have an activation function or there was an error while building");
    vector<double> outputs;
    vector<double> signals;
    for (size_t i = 0; i < _weights.rows(); i++) {
        double output = 0;
        output += _weights[i][0];
        for (size_t j = 1; j < _weights[i].size(); j++) {
            output += _weights[i][j] * output_aux[j-1];
        }
        signals.push_back(output);
        outputs.push_back(activationFunction(output));
    }
    setSignals(signals);
    return outputs;
}

vector<double> NeuralNetworkLayer::getGradients() const {
    return _gradients;
}

vector<double> NeuralNetworkLayer::getOutputs() const {
    return _outputs;
}

vector<double> NeuralNetworkLayer::getSignals() const {
    return _signal;
}

vector<double> NeuralNetworkLayer::calculateGradients(const vector<double>& output, const vector<double>& target){
    // Calculate the difference between the predicted output and the desired output
    vector<double> errors;
    vector<double> target2=target;
    for (size_t i = 0; i < target.size(); ++i) {
        for(size_t j=0; j<target.size(); j++){
            if(target[j]==-1) target2[j]=0;
        }
        errors.push_back(2*(output[i] - target2[i]));
    }

    // Calculate the derivative of the activation function
    vector<double> activationDerivative;
    for (size_t i = 0; i < _signal.size(); ++i) {
        activationDerivative.push_back(ActivationFunctions::sigmoid_prime(_signal[i]));
    }

    // Calculate the gradients by multiplying the errors by the derivative of the activation function
    for (size_t i = 0; i < _gradients.size(); ++i) {
        _gradients[i] = errors[i] * activationDerivative[i];
    }

    return _gradients;
}

vector<double> NeuralNetworkLayer::calculateGradientsMedio(const vector<double>& nextLayerGradients, const Matrix& weights) {
    // Calculates the gradients of this layer based on the gradients of the next layer

    // Calculate the derivative of the activation function
    vector<double> activationDerivative;
    for (size_t i = 0; i < _signal.size(); ++i) {
        activationDerivative.push_back(ActivationFunctions::sigmoid_prime(_signal[i]));
    }
    //cout << "HOLIWI -- NextLayerGradients.size = " << nextLayerGradients.size() << " activationDerivative.size = " << activationDerivative.size() << " _gradients.size = " << _gradients.size() << " weights_rows = " << weights.rows() << " weights_cols = " << weights.cols() << endl;
    // Multiply the gradients of the next layer by the weights of the next layer and multiply them by the activation function
    for (size_t i = 0; i < _gradients.size(); ++i) {
        _gradients[i] = 0.0;
        for (size_t j = 0; j < nextLayerGradients.size(); ++j) {
            _gradients[i] += nextLayerGradients[j] * weights[j][i];
        }
        _gradients[i] *= activationDerivative[i];
    }
    
    return _gradients;
}

void NeuralNetworkLayer::updateWeights(const double learningRate, const vector<double>& outputs) {
    // Update the weights by subtracting the gradient from the old weights by the learning rate by the output of the previous layer
    //cout << "Gradients size: " << _gradients.size() << " outputs size: " << outputs.size() << " _weights.rows = " << _weights.rows() << " _weights.cols() = " << _weights.cols()<< endl;
    for (size_t i = 0; i < _weights.rows(); ++i) {
        for (size_t j = 1; j < _weights[i].size(); ++j) {
            _weights[i][j] -= learningRate * _gradients[i] * outputs[j-1];
        }
    }


    // Reset the gradients
    _gradients.assign(_gradients.size(), 0.0);
}

void NeuralNetworkLayer::setOutputs(const vector<double>& outputs){
    _outputs = outputs;
}

void NeuralNetworkLayer::setSignals(const vector<double>& signals){
    _signal = signals;
}