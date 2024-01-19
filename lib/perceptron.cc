#include "perceptron.h"
#include "randonn_generator.h"
#include "functions.h"
#include "environment.h"
#include "matrix.h"
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>

#define ITERATION_INFO 100

/* ============================================
*  Perceptron
*  Represents a perceptron
* ============================================
*/


// ============================================
// =============== Constructor ================
/// @brief Constructor of the class Perceptron
/// @param num_weights The number of weights of the perceptron
Perceptron::Perceptron(std::size_t const num_weights){
    // One more weight for the bias --> the first weight is the bias
    _weights = generateRandomWeights(num_weights);
}

// ============================================
// =============== Methods ====================
/// @brief Generates random weights for the perceptron
/// @param num_weights The number of weights of the perceptron
/// @return A matrix with the random weights
Matrix Perceptron::generateRandomWeights(std::size_t const num_weights){
    Randonn_generator generator;
    Matrix weights(num_weights, 1);

	for (int i = 1; i < weights.size(); i++) {
		weights[i][0] = generator.randomDouble(MIN_WEIGHT, MAX_WEIGHT);
	}

    return weights;
}

/// @brief Trains the perceptron
/// @param X The input matrix
/// @param Y The matrix with the class of each input
/// @param maxiter The maximum number of iterations
void Perceptron::train(Matrix const& X, Matrix const& Y, std::size_t const maxiter){
    Randonn_generator generator;

    Matrix wBest(_weights.rows(), _weights.cols(), _weights.matrix());
    double bestAccuracy = 0.0;

    for(size_t i = 0; i < maxiter; ++i){
        // Multiply the input matrix by the weights
        Matrix product = X * _weights;
        // Get the predicted output
        Matrix predicted = product.apply(ActivationFunctions::sign);
        // Get the number of errors
        Matrix errors = predicted != Y;
        double numErrors = errors.sumcol(0);
        double accuracy = 1.0 - (numErrors / static_cast<double>(Y.rows()));

        if (accuracy > bestAccuracy) {
            bestAccuracy = accuracy;
            wBest = _weights;  // Copy the weights
        }
        if (i % ITERATION_INFO == 0)
            std::cout << "Iteration: " << i << " Accuracy: " << accuracy << std::endl;

        // If there are no errors, stop training
        if(numErrors == 0){
            std::cout << "========================================" << std::endl;
            std::cout << "| No errors found. Stopping training." << std::endl;
            std::cout << "| Iteration: " << i << " Accuracy: " << accuracy << std::endl;
            std::cout << "========================================" << std::endl;
            break;
        }
        // Picks an example from (x1 , Y1) · · · (xN, YN) that is currently misclassified, call it (x(t) , y(t)), and
        //uses it to update w(t) . Since the example is misclassified, we have y (t) #­sign(wT(t)x(t)). 
        //The update rule is w(t + 1) = w(t) + y(t)x(t) . 

        // Pick a random example
        size_t index = generator.randomInt(0, X.rows() - 1);

        while (errors[index][0] != 1.0)
        {
            index = generator.randomInt(0, X.rows() - 1);
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

/// @brief Classifies the input matrix
/// @param input The input matrix
/// @return The predicted output
double Perceptron::classify(const Matrix& input){
    // Multiply the input matrix by the weights
    Matrix product = input * _weights;
    // Get the predicted output
    Matrix predicted = product.apply(ActivationFunctions::sign);
    //std::cout << "Predicted: " << predicted << std::endl;
    return predicted[0][0];
}

/// @brief Tests the perceptron
/// @param X The input matrix
/// @param Y The matrix with the class of each input
/// @return The accuracy of the perceptron
double Perceptron::test(Matrix const& X, Matrix const& Y) const{
    // Multiply the input matrix by the weights
    Matrix product = X * _weights;
    // Get the predicted output
    Matrix predicted = product.apply(ActivationFunctions::sign);
    // Get the number of errors
    Matrix errors = predicted != Y;
    double numErrors = errors.sumcol(0);
    double accuracy = 1.0 - (numErrors / static_cast<double>(Y.rows()));
    return accuracy;
}

// ============================================
// =============== Static methods ==============
/// @brief Creates a pair of matrices from a CSV file
std::pair<Matrix, Matrix> Perceptron::readFromCSV(std::string const& filename){
    std::ifstream file(filename);
    std::string line;

    std::vector<double> vectorX;
    std::vector<double> vectorY;
    std::size_t rowCount = 0;
    size_t num_inputs = 0;

    while (std::getline(file, line, '\n')) {
        std::stringstream ss(line);
        //std::cout << line << '\n';
        std::vector<std::string> tokens;
        
        // Dividir la línea en tokens utilizando el delimitador ","
        while (std::getline(ss, line, ',')) {
            tokens.push_back(line);
        }
        num_inputs = tokens.size() -1;

        // Leer los valores de los tokens
        // Hasta num_inputs para la X
        // El ultimo valor para la Y
        vectorX.push_back(1.0);
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            double value = std::stod(tokens[i]);
            //std::cout << "Value "<< value << "\n";
            if(i < num_inputs){
                vectorX.push_back(value);
            }else{
                // Leer ultimo valor en el vectorY
                vectorY.push_back(value);
            }
            
        }

        // Incrementar el contador de filas
        ++rowCount;
    }

    Matrix X{rowCount, num_inputs +1, vectorX};
    // Matrix X{rowCount, num_inputs, vectorX};
    Matrix Y{rowCount, 1, vectorY};
    
    return std::make_pair(X, Y);
     
}