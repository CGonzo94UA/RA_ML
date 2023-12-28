#include "perceptron.h"
#include <vector>
#include <random>

// ============================================
// =============== Constructor ================
Perceptron::Perceptron(std::size_t const num_weights){
    _weights = generateRandomWeights(num_weights);
}

// ============================================
// =============== Destructor =================
Perceptron::~Perceptron(){}

// ============================================
// =============== Methods ====================
double Perceptron::sign(double const x){
    if (x >= 0)
        return 1.0;
    else
        return -1.0;
}

Matrix Perceptron::generateRandomWeights(std::size_t const num_weights){
    std::random_device seed;
	std::default_random_engine generator(seed());
	std::uniform_real_distribution<double> distributionDouble(-.5, .5);
    Matrix weights(num_weights, 1);

	for (int i = 0; i < weights.size(); i++) {
		weights[i][0] = distributionDouble(generator);
	}

    return weights;
}