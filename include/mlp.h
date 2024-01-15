#ifndef __MLP_H__
#define __MLP_H__

#include "matrix.h"
#include "layer.h"
#include "functions.h"
#include <vector>
#include <iostream>

using namespace std;

class MLP;
class MLP_Builder;
class MLP_Display;

class MLP{
    public:
        ~MLP();

        void setInputs(const vector<double>& inputs);
        vector<double> getOutputs() const;
        vector<double> getInputs() const;
        void train(const Matrix& trainingData, const Matrix& targetData, size_t epochs, double learningRate);
        double test(const Matrix& testData, const Matrix& targetData);

    private:

        MLP();

        std::vector<NeuralNetworkLayer*> _layers;
        NeuralNetworkLayer* _inputLayer;
        NeuralNetworkLayer* _outputLayer;
        vector<double> _inputs;
        void updateWeights(double learningRate);
        void backpropagate(const vector<double>& output, const vector<double>& target);
        void saveWeights();

        friend class MLP_Builder;
        friend class MLP_Display;

};

class MLP_Builder{
    public:
        MLP_Builder();
        ~MLP_Builder();

        MLP_Builder& addLayer(std::size_t const& width_of_layer, std::size_t const& num_weights);
        MLP_Builder& addLayer(std::size_t const& width_of_layer, std::size_t const& num_weights, double (*f)(double));
        MLP_Builder& setActivationFunc(double (*f)(double));

        MLP* build();
        MLP* build(string filename);

    private:
        MLP* _mlp;
};

class MLP_Display{
    public:
        static void display(MLP const& mlp);
};

#endif