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
    for (size_t i = 0; i < _layers.size(); ++i) {
        delete _layers[i];
    }

    _layers.clear();
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

    _inputLayer->setInputs(_inputs);
    
    vector<double> outputs = _inputLayer->feedForward();

    // Send the outputs of each layer to the next layer
    for (size_t i = 1; i < _layers.size(); i++) {
        _layers[i]->setInputs(outputs);
        outputs = _layers[i]->feedForward();
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

            //Removes the last element of the row (the class)
            //vector<double> training_row = trainingData.getRow(i);
            //training_row.pop_back();

            setInputs(trainingData.getRow(i));  //CAMBIAR POR UN METODO PARA SACAR LOS DATOS
            
            // Realiza el pase hacia adelante (feedforward)
            output = getOutputs();

            target = targetData.getRow(i);

            //Realiza el pase hacia atrás (backpropagation) y actualiza los pesos
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
    double accuracy = 1.0 - (numErrors / static_cast<double>(targetData.rows()));
    // std::cout << "Accuracy: " << accuracy << std::endl;
    return accuracy;
    
}

vector<Matrix> MLP::getWeights() const {
    vector<Matrix> weights;
    // Obtiene los pesos de cada capa
    for (size_t i = 0; i < _layers.size(); ++i) {
        weights.push_back(_layers[i]->weights());
    }

    return weights;
}

void MLP::setWeights(const vector<Matrix>& weights) {
    // Establece los pesos de cada capa
    // std::cout << "Setting weights" << std::endl;
    for (size_t i = 0; i < _layers.size(); ++i) {
        // std::cout << "Layer: " << i << std::endl;
        _layers[i]->setWeights(weights[i]);
    }
}

MLP* MLP::clone() const {
    // Clona la red
    MLP_Builder builder;
    builder.addLayer(_inputLayer->weights().rows(), _inputLayer->weights()[0].size() - 1, _inputLayer->activationFunction);
    for (size_t i = 1; i < _layers.size(); ++i) {
        builder.addLayer(_layers[i]->weights().rows(), _layers[i]->weights()[0].size() - 1, _layers[i]->activationFunction);
    }

    MLP* mlp = builder.build();
    mlp->setWeights(getWeights());

    return mlp;
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
    //cout << "layers: " << _layers.size() << "layers -2: " << _layers.size()-2 << endl;
    // Luego, calcula los gradientes de las capas ocultas en orden inverso
    for (int i = _layers.size() - 2; i >= 0; --i) {
        gradients = _layers[i]->calculateGradientsMedio(gradients, _layers[i+1]->getInputs());
    }
}

#include <fstream>

void MLP::saveWeights() {
    // Guarda los pesos de la red en un archivo
    // TODO

    // Creates the file
    std::ofstream file;
    file.open("weights.txt");


    // Writes the weights of the hidden layers line by line
    for (size_t i = 0; i < _layers.size(); i++) {
        for (size_t j = 0; j < _layers[i]->weights().rows(); j++) {
            for (size_t k = 0; k < _layers[i]->weights()[j].size(); k++) {
                file << _layers[i]->weights()[j][k] << "-";
            }
            file << endl;
        }
        file << "===" << endl;
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

//TODO: CHECK IF THIS WORKS ALONG WITH THE MLP::SAVEWEIGHTS() METHOD
MLP* MLP_Builder::build(string filename) {
    
    // Creates the MLP From a file

    // Creates the MLP
    MLP* mlp = new MLP();
    vector<vector<double>> weights_v;
    size_t contador = 0;
    //reads the file, line by line. Each line is a different layer neuron and each layer is separated by "==="

    // Opens the file
    ifstream file(filename);
    string line;

    // Reads the file line by line
    while (getline(file, line)) {
        // Checks if the line is a separator
        if (line == "===") {
            
            // Creates a Matrix with the weights
            Matrix weights(weights_v.size(), weights_v[0].size(), weights_v);

            // Adds the weights to the layer
            mlp->_layers.push_back(new NeuralNetworkLayer(weights.rows(), weights.cols()));
            mlp->_layers[mlp->_layers.size() - 1]->setWeights(weights);

            // Adds the layer to the MLP
            if (mlp->_inputLayer == nullptr) {
                mlp->_inputLayer = mlp->_layers[0];
            }

            mlp->_outputLayer = mlp->_layers[mlp->_layers.size() - 1];
            continue;

        }

        // Splits the line by the "-" separator
        vector<string> tokens;
        stringstream ss(line);
        string token;
        while (getline(ss, token, '-')) {
            tokens.push_back(token);
        }

        // Converts the tokens to doubles and adds them to the weights vector
        for (size_t i = 0; i < tokens.size(); i++) {
            weights_v[contador].push_back(stod(tokens[i]));
        }


    }

    // Closes the file
    file.close();

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

    for (size_t i = 1; i < mlp._layers.size()-1; i++) {
        cout << "Hidden layer " << i+1 << ": " << endl;
        cout << "Weights of neurons (each line is a different neuron): " << endl;
        cout << mlp._layers[i]->weights() << endl;
    }

    cout << "Output layer: " << endl;
    cout << "Weights: " << endl;
    cout << mlp._outputLayer->weights() << endl;
}