#include "perceptron.h"
#include <vector>
#include <random>
#include <algorithm>

// ============================================
// =============== Constructor ================
Perceptron::Perceptron(std::size_t const num_weights){
    // One more weight for the bias --> the first weight is the bias
    _weights = generateRandomWeights(num_weights);
}

// ============================================
// =============== Methods ====================
double sign(double x){
    if (x >= 0)
        return 1.0;
    else
        return -1.0;
}

double Perceptron::dot(const vector<double>& v1, const vector<double>& v2) {
	double sum = 0;
	for (size_t i = 0; i < v1.size() - 1; i++) {
		sum += v1[i] * v2[i];
	}
	return sum;
    
    //return inner_product(v1.begin(), v1.end() - 1, v2.begin(), 0.0);
}

Matrix Perceptron::generateRandomWeights(std::size_t const num_weights){
    std::random_device seed;
	std::default_random_engine generator(seed());
	std::uniform_real_distribution<double> distributionDouble(-.5, .5);
    Matrix weights(num_weights, 1);

	for (int i = 1; i < weights.size(); i++) {
		weights[i][0] = distributionDouble(generator);
	}

    return weights;
}

/*
void Perceptron::train_(Matrix const& X, std::vector<double> const& Y, std::size_t const maxiter){
    int total_errors = 0;

    for (size_t i = 0; i < maxiter; ++i) {
        total_errors = 0;
        //std::random_device seed;
        //shuffle(X.matrix().begin(), X.matrix().end(), seed);

        for (size_t j = 0; j < X.rows(); j++) {
            double dot_product = dot(X[j], _weights);
            std::cout << "Dot product: " << dot_product << std::endl;

            // Add bias
            dot_product += _bias * _weights.back();
            std::cout << "Dot product + bias: " << dot_product << std::endl;

            double calculatedOutput = sign(dot_product);
            std::cout << "Calculated output: " << calculatedOutput << std::endl;
            double error = Y[j] - calculatedOutput;
            std::cout << "Error: " << error << std::endl;

            if (error != 0) {
                total_errors++;
                //Perform training
                for (int k = 0; k < _weights.size() - 1; k++) {
                    _weights[k] += X[j][k] * error;
                }
                _weights.back() += _bias * error;
            }
        }
        double accuracy = total_errors/X.rows();
        std::cout << "Iteration: " << i << " Accuracy: " << accuracy << std::endl;
        if (total_errors == 0) {
            break;
        }

    }
}*/

void Perceptron::train(Matrix const& X, Matrix const& Y, std::size_t const maxiter){
    
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,X.rows()-1);
    Matrix wBest(_weights.rows(), _weights.cols(), _weights.matrix());
    double bestAccuracy = 0.0;

    for(size_t i = 0; i < maxiter; ++i){
        // Multiply the input matrix by the weights
        Matrix product = X * _weights;
        // Get the predicted output
        Matrix predicted = product.apply(sign);
        // Get the number of errors
        Matrix errors = predicted != Y;
        double numErrors = errors.sumcol(0);
        double accuracy = 1.0 - (numErrors / static_cast<double>(Y.rows()));

        if (accuracy > bestAccuracy) {
            bestAccuracy = accuracy;
            wBest = _weights;  // Copy the weights
        }
        std::cout << "Iteration: " << i << " Accuracy: " << accuracy << std::endl;

        // If there are no errors, stop training
        if(numErrors == 0){
            break;
        }
        // Picks an example from (x1 , Y1) · · · (xN, YN) that is currently misclassified, call it (x(t) , y(t)), and
        //uses it to update w(t) . Since the example is misclassified, we have y (t) #­sign(wT(t)x(t)). 
        //The update rule is w(t + 1) = w(t) + y(t)x(t) . 

        // Pick a random example
        size_t index = distribution(generator);

        while (errors[index][0] != 1.0)
        {
            index = distribution(generator);
        }
        //std::cout << "Index: " << index << std::endl;
        //std::cout << "Weights: " << _weights << std::endl;
        // Update the weights
        Matrix xy = X.mult(index, Y[index][0]);
        //std::cout << "xy: " << xy << std::endl;
        _weights += xy.transpose();

        //Print the weights
        //std::cout << "Weights: " << _weights << std::endl;

        
    }

    // Copy the best weights
    _weights = wBest;

}

double Perceptron::classify(const Matrix& input){
    // Multiply the input matrix by the weights
    Matrix product = input * _weights;
    // Get the predicted output
    Matrix predicted = product.apply(sign);
    //std::cout << "Predicted: " << predicted << std::endl;
    return predicted[0][0];
}

