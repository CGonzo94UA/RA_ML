#include "mlp.h"
#include "layer.h"
#include "functions.h"
#include "environment.h"

#include <cassert>
#include <fstream>
#include <sstream>

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
    //cout << "Inputs: " << inputs.size() << " Esperados: " << _inputLayer->weights()[0].size()-1 << endl;
    if (inputs.size() != _inputLayer->weights()[0].size()) {
        throw invalid_argument("Number of inputs does not match the number of neurons in the input layer");
    }

    _inputs = inputs;
}

vector<double> MLP::getOutputs() const {
    
    //Checks if _inputLayer is null and if _inputs is also null
    assert(_inputLayer != nullptr && "The MLP must have an input layer or there was an error while building");

    _inputLayer->setOutputs(_inputs);
    
    vector<double> outputs;

    // Send the outputs of each layer to the next layer
    for (size_t i = 1; i < _layers.size(); i++) {
        outputs = _layers[i]->feedForward(_layers[i-1]->getOutputs());
        _layers[i]->setOutputs(outputs);
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

    #if DEBUG == 1
        cout << "Training for " << epochs << " epochs" << endl;
    #endif
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        #if DEBUG == 1
            cout << "Training for Epoch: " << epoch << "/" << epochs << endl;
        #endif
        //cout << "Training for Epoch: " << epoch << "/" << epochs << endl;
        for (size_t i = 0; i < trainingData.rows(); ++i) {
            // Establece las entradas

            //Set the inputs
            setInputs(trainingData.getRow(i));  //CAMBIAR POR UN METODO PARA SACAR LOS DATOS
            
            // Get the output with feedforward
            output = getOutputs();
            target = targetData.getRow(i);

            //Backpropagation and weights update
            backpropagate(output, target);
            updateWeights(learningRate);
        }
    }
}


double MLP::test(const Matrix& testData, const Matrix& targetData) {
    Matrix output(testData.rows(), 1);

    // Verifica si las capas están correctamente configuradas
    assert(_inputLayer != nullptr && "La MLP debe tener una capa de entrada o hubo un error durante la construcción");
    assert(_outputLayer != nullptr && "La MLP debe tener una capa de salida o hubo un error durante la construcción");
    assert(_layers.size() > 1 && "La MLP debe tener al menos una capa oculta o hubo un error durante la construcción");

    for (size_t i = 0; i < testData.rows(); ++i) {
        // Establece las entradas
        setInputs(testData.getRow(i));

        // Realiza el pase hacia adelante (feedforward)
        output[i] = getOutputs();
    }

    #if DEBUG == 2
        cout << "Output: " << endl;
        cout << output << endl;
        cout << "Target: " << endl;
        cout << targetData << endl;
    #endif

    Matrix predicted = output.apply(ActivationFunctions::binary);
    double numErrors = (predicted != targetData).sumcol(0);
    return 1.0 - (numErrors / static_cast<double>(targetData.rows()));
    
}

void MLP::updateWeights(double learningRate) {
    // Start in the last layer
    _outputLayer->updateWeights(learningRate, _layers[_layers.size()-1]->getOutputs());

    // Update the layers in reverse order
    for (int i = _layers.size() - 2; i >= 1; --i) {
        _layers[i]->updateWeights(learningRate, _layers[i-1]->getOutputs());
    }
}

void MLP::backpropagate(const vector<double>& output, const vector<double>& target) {
    vector<double> gradients;

    #if DEBUG == 1
        cout << "Layers: " << _layers.size() << endl;
        cout << "Back Propagating Layer " << _layers.size() << "/" << _layers.size() << " (OUTPUT)" << endl;
    #endif
    // Start in the last layer
    gradients = _outputLayer->calculateGradients(output, target);
    
    // Calculate the gradients in reverse order
    for (int i = _layers.size() - 2; i >= 1; --i) {
        #if DEBUG == 1
        cout << "Back Propagating Layer " << i+1 << "/" << _layers.size() << endl;
        #endif
        gradients = _layers[i]->calculateGradientsMedio(gradients, _layers[i+1]->weights());
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
    //delete _mlp;
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

    MLP* mlp = _mlp;
    _mlp = nullptr;

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
    //cout << mlp._inputLayer->weights() << endl;

    for (size_t i = 1; i < mlp._layers.size()-1; i++) {
        cout << "Hidden layer " << i+1 << ": " << endl;
        cout << "Weights of neurons (each line is a different neuron): " << endl;
        cout << mlp._layers[i]->weights() << endl;
    }

    cout << "Output layer: " << endl;
    cout << "Weights: " << endl;
    cout << mlp._outputLayer->weights() << endl;
}