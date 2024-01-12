#include "../include/mlp.h"
#include "../include/layer.h"
#include "../include/functions.h"

#include <cassert>

using namespace std;

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
    _inputLayer->setInputs(_inputs);
    vector<double> outputs = _inputLayer->feedForward(_inputs);

    // Send the outputs of each layer to the next layer
    for (size_t i = 0; i < _layers.size(); i++) {
        _layers[i]->setInputs(outputs);
        outputs = _layers[i]->feedForward(outputs);
    }

    return outputs;

}

vector<double> MLP::getInputs() const {
    return _inputs;
}

void MLP::train(const Matrix& trainingData, const Matrix& targetData, size_t epochs, double learningRate) {
    vector<double> output;
    vector<double> target;
    // Verifica si las capas están correctamente configuradas
    assert(_inputLayer != nullptr && "La MLP debe tener una capa de entrada o hubo un error durante la construcción");
    assert(_outputLayer != nullptr && "La MLP debe tener una capa de salida o hubo un error durante la construcción");
    assert(_layers.size() > 1 && "La MLP debe tener al menos una capa oculta o hubo un error durante la construcción");

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < trainingData.rows(); ++i) {
            // Establece las entradas
            setInputs(trainingData.getRow(i));  //CAMBIAR POR UN METODO PARA SACAR LOS DATOS
            
            // Realiza el pase hacia adelante (feedforward)
            output = getOutputs();

            target = targetData.getRow(i);

            // Realiza el pase hacia atrás (backpropagation) y actualiza los pesos
            backpropagate(output, target);
            updateWeights(learningRate);
        }
    }
}

void MLP::updateWeights(double learningRate) {
    // Actualiza los pesos de las capas ocultas y de salida utilizando el descenso de gradiente

    // Comienza por la última capa
    _outputLayer->updateWeights(learningRate);

    // Luego, actualiza las capas ocultas en orden inverso
    for (int i = _layers.size() - 2; i >= 0; --i) {
        _layers[i]->updateWeights(learningRate);
    }
}

void MLP::backpropagate(const vector<double>& output, const vector<double>& target) {
    vector<double> gradients;
    // Realiza el paso hacia atrás (backpropagation) para calcular los gradientes

    // Comienza por la última capa
    gradients = _outputLayer->calculateGradients(output, target);

    // Luego, calcula los gradientes de las capas ocultas en orden inverso
    for (int i = _layers.size() - 2; i >= 0; --i) {
        gradients = _layers[i]->calculateGradientsMedio(gradients, _layers[i+1]->getInputs());
    }
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
    NeuralNetworkLayer* newLayer = new NeuralNetworkLayer(width_of_layer, num_weights);
    
    if (_mlp->_inputLayer == nullptr) {
        _mlp->_inputLayer = newLayer;
    }

    _mlp->_layers.push_back(newLayer);
    _mlp->_outputLayer = _mlp->_layers[_mlp->_layers.size() - 1];

    return *this;
}

// Add a layer to the MLP with the specified width and number of weights per neuron with the specified activation function (from functions.h)
MLP_Builder& MLP_Builder::addLayer(size_t const& width_of_layer, size_t const& num_weights, double (*f)(double)) {
    NeuralNetworkLayer* newLayer = new NeuralNetworkLayer(width_of_layer, num_weights, f);
    
    if (_mlp->_inputLayer == nullptr) {
        _mlp->_inputLayer = newLayer;
    }
    
    _mlp->_layers.push_back(newLayer);
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