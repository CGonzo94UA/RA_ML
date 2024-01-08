#include "mlp.h"
#include "layer.h"
#include "functions.h"

#include <cassert>

// ============================================
// =============== MLP CLASS ===============

// ============================================
// =============== Constructors ===============


MLP::MLP() {
    _inputLayer = nullptr;
    _outputLayer = nullptr;
    _inputs = vector<double>(0);
}

MLP::~MLP() {
    for (size_t i = 0; i < _layers.size(); i++) {
        delete _layers[i];
    }

    delete _inputLayer;
    delete _outputLayer;
}

// ============================================
// ================= Methods ==================

void MLP::setInputs(const vector<double>& inputs) {
    if (inputs.size() != _inputLayer->weights()[0].size() - 1) {
        throw invalid_argument("Number of inputs does not match the number of neurons in the input layer");
    }

    _inputs = inputs;
}

vector<double> MLP::getOutputs() const {
    
    vector<double> outputs = _inputLayer->feedForward(_inputs);

    // Send the outputs of each layer to the next layer
    for (size_t i = 0; i < _layers.size(); i++) {
        outputs = _layers[i]->feedForward(outputs);
    }

    return outputs;

}

vector<double> MLP::getInputs() const {
    return _inputs;
}

// =================================================
// =============== MLP_BUILDER CLASS ===============

// =================================================
// =============== Constructors ====================

MLP_Builder::MLP_Builder() {
    _mlp = new MLP();
}

MLP_Builder::~MLP_Builder() {
    delete _mlp;
}

// =================================================
// ================= Methods =======================

// Add a layer to the MLP with the specified width and number of weights per neuron with the default activation function (sigmoid)
MLP_Builder& MLP_Builder::addLayer(size_t const& width_of_layer, size_t const& num_weights) {
    if (_mlp->_inputLayer == nullptr) {
        _mlp->_inputLayer = new NeuralNetworkLayer(width_of_layer, num_weights);
        _mlp->_layers.push_back(_mlp->_inputLayer);
    }
    else {
        _mlp->_layers.push_back(new NeuralNetworkLayer(width_of_layer, num_weights));
    }

    _mlp->_outputLayer = _mlp->_layers[_mlp->_layers.size() - 1];

    return *this;
}

// Add a layer to the MLP with the specified width and number of weights per neuron with the specified activation function (from functions.h)
MLP_Builder& MLP_Builder::addLayer(size_t const& width_of_layer, size_t const& num_weights, double (*f)(double)) {
    if (_mlp->_inputLayer == nullptr) {
        _mlp->_inputLayer = new NeuralNetworkLayer(width_of_layer, num_weights, f);
        _mlp->_layers.push_back(_mlp->_inputLayer);
    }
    else {
        _mlp->_layers.push_back(new NeuralNetworkLayer(width_of_layer, num_weights, f));
    }

    _mlp->_outputLayer = _mlp->_layers[_mlp->_layers.size() - 1];

    return *this;
}

// Sets the same activation function for all the MLP
MLP_Builder& MLP_Builder::setActivationFunc(double (*f)(double)) {
    for (size_t i = 0; i < _mlp->_layers.size(); i++) {
        _mlp->_layers[i]->setActivationFunc(f);
    }

    return *this;
}

// Builds the MLP
MLP* MLP_Builder::build() {

    // Check if the MLP has at least one layer
    assert(_mlp->_inputLayer != nullptr && "The MLP must has no input layer or there was an error while building");     // If the MLP has no layers, throw an error message
    assert(_mlp->_outputLayer != nullptr && "The MLP must has no output layer or there was an error while building");    // If the MLP has no layers, throw an error message
    
    //Asserts that the MLP has at least one hidden layer
    //IF size == 1, then there is only the input layer and the output layer
    assert(_mlp->_layers.size() > 1 && "The MLP must has at least one hidden layer or there was an error while building");    // If the MLP has no layers, throw an error message

    MLP* mlp = new MLP();

    mlp->_inputLayer = _mlp->_inputLayer;
    mlp->_outputLayer = _mlp->_outputLayer;
    mlp->_layers = _mlp->_layers;

    return mlp;
}

// =================================================
// =============== MLP_DISPLAY CLASS ===============

// =================================================
// ================== Method =======================

// Displays the MLP
void MLP_Display::display(MLP const& mlp) {
    cout << "Input layer: " << endl;
    cout << "Weights: " << endl;
    cout << mlp._inputLayer->weights() << endl;

    for (size_t i = 0; i < mlp._layers.size(); i++) {
        cout << "Hidden layer " << i+1 << ": " << endl;
        cout << "Weights of neurons (each line is a different neuron): " << endl;
        cout << mlp._layers[i]->weights() << endl;
    }

    cout << "Output layer: " << endl;
    cout << "Weights: " << endl;
    cout << mlp._outputLayer->weights() << endl;
}