#include "layer.h"

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

    for (size_t i = 0; i < width_of_layer; i++) {
        _weights[0].push_back(1.0);      // Bias
        for (size_t j = 0; j < num_weights; j++) {
            _weights[i+1].push_back((double)rand() / RAND_MAX);
        }
    }

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

vector<double> NeuralNetworkLayer::getGradients() const {
    return _gradients;
}

vector<double> NeuralNetworkLayer::getInputs() const {
    return _inputs;
}

vector<double> NeuralNetworkLayer::calculateGradients(const vector<double>& output, const vector<double>& target){
    // Calcular la diferencia entre la salida predicha y la salida deseada
    vector<double> errors;
    for (size_t i = 0; i < target.size(); ++i) {
        errors.push_back(output[i] - target[i]);
    }

    // Calcular la derivada de la función de activación (sigmoide en este caso)
    vector<double> activationDerivative;
    for (size_t i = 0; i < output.size(); ++i) {
        activationDerivative.push_back(output[i] * (1.0 - output[i]));
    }

    // Calcular los gradientes multiplicando los errores por la derivada de la activación
    for (size_t i = 0; i < _gradients.size(); ++i) {
        _gradients[i] = errors[i] * activationDerivative[i];
    }

    return _gradients;
}

vector<double> NeuralNetworkLayer::calculateGradientsMedio(const vector<double>& nextLayerGradients, const vector<double>& outputs) {   //Añadir learningRate?
    // Calcula los gradientes de esta capa en función de los gradientes de la capa siguiente

    // Calcula la derivada de la función de activación en las entradas de esta capa
    vector<double> activationDerivative;
    for (size_t i = 0; i < outputs.size(); ++i) {
        activationDerivative.push_back(activationFunctionDerivative(outputs[i]));
    }

    // Calcula los gradientes multiplicando los gradientes de la capa siguiente por la derivada de la activación
    for (size_t i = 0; i < _gradients.size(); ++i) {
        _gradients[i] = 0.0;
        for (size_t j = 0; j < nextLayerGradients.size(); ++j) {
            _gradients[i] += nextLayerGradients[j] * _weights[i][j];
        }
        _gradients[i] *= activationDerivative[i];
        //_gradients[i] *= learningRate;
    }

    return _gradients;
}

double NeuralNetworkLayer::activationFunctionDerivative(double x) {
    // Derivada de la función sigmoide
    double sigmoid = 1.0 / (1.0 + exp(-x));
    return sigmoid * (1.0 - sigmoid);
}

void NeuralNetworkLayer::updateWeights(double learningRate) {
    // Asumiendo que '_gradients' contiene los gradientes calculados en la retropropagación

    // Actualizar los pesos de las conexiones entre esta capa y la capa siguiente (forward pass)
    for (size_t i = 0; i < _weights.size(); ++i) {
        for (size_t j = 0; j < _weights[i].size(); ++j) {
            // Actualizar cada peso usando el descenso de gradiente
            _weights[i][j] -= learningRate * _gradients[i] * _inputs[j];
        }
    }


    // Finalmente, puedes reiniciar los gradientes para la próxima actualización
    _gradients.assign(_gradients.size(), 0.0);
}

void NeuralNetworkLayer::setInputs(const vector<double>& inputs){
    _inputs = inputs;
}