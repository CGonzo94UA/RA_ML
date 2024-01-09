#ifndef __MLP_H__
#define __MLP_H__

#include "matrix.h"
#include "layer.h"
#include "functions.h"

class MLP;
class MLP_Builder;
class MLP_Display;

class MLP{
    public:
        ~MLP();

        void setInputs(const std::vector<double>& inputs);
        std::vector<double> getOutputs() const;
        std::vector<double> getInputs() const;

    private:

        MLP();

        std::vector<NeuralNetworkLayer*> _layers;
        NeuralNetworkLayer* _inputLayer;
        NeuralNetworkLayer* _outputLayer;
        std::vector<double> _inputs;


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

    private:
        MLP* _mlp;
};

class MLP_Display{
    public:
        static void display(MLP const& mlp);
};

#endif