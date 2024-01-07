#include "adaBoost.h"
#include "perceptron.h"
#include <math.h>

// ============================================
// =============== Constructor ================
AdaBoost::AdaBoost(std::size_t const num_weakLearners): _numWeakLearners(num_weakLearners), _weakLearnerWeights(num_weakLearners, 0.0) {    
    // Initialize the vector weights to 0.0
}


// ============================================
// =============== Destructor ================
AdaBoost::~AdaBoost(){
    for (auto& weakLearner : weakLearners){
        delete weakLearner;
    }
}

// ============================================
// =============== Methods ====================
Matrix AdaBoost::generateWeights(std::size_t const num_weights){
    Matrix weights(num_weights, 1);

    // Initialize the weights to 1/N
    for (int i = 1; i < weights.size(); i++) {
        weights[i][0] = 1.0 / static_cast<double>(weights.size());
    }

    return weights;
}

void AdaBoost::createWeakLearners(std::size_t const num_features){
    for (size_t i = 0; i < _numWeakLearners; ++i){
        weakLearners.push_back(new Perceptron(num_features));
    }
}

void AdaBoost::train(Matrix const& X, Matrix const& Y, std::size_t const maxiter){
    size_t numSamples = X.rows();
    // Initialize the weights to 1/N for every sample
    Matrix weights = generateWeights(numSamples);
    size_t numFeatures = X.cols();

    // Create and store the weak learners
    createWeakLearners(numFeatures);

    for(size_t t = 0; t < maxiter; ++t){
        // Train the weak learner
        weakLearners[t]->train(X, Y, maxiter);
        // Get the predicted output
        Matrix predicted = weakLearners[t]->classify_all(X);
        // Calculate the error of the weak learner
        Matrix errors = predicted != Y;
        double numErrors = errors.sumcol(0);
        double accuracy = 1.0 - (numErrors / static_cast<double>(Y.rows()));

        if(accuracy < 0.5){
            // Discard weak learner if accuracy is less than 0.5
            delete weakLearners[t];
            weakLearners.erase(weakLearners.begin() + t);
            // Delete the weight of the weak learner
            _weakLearnerWeights.erase(_weakLearnerWeights.begin() + t);
            // Update the number of weak learners
            _numWeakLearners--;
            // Update the number of iterations
            t--;
            continue;  // Skip to the next iteration
        }

        // Calculate the weight of the weak learner in the ensemble
        double alpha = 0.5 * log((1.0 - accuracy) / std::max(accuracy, 1e-10));

        // Update the weights of the samples
        for (size_t i = 0; i < numSamples; ++i) {
            double factor = exp(-alpha * Y[i][0]* predicted[i][0]);
            weights[i][0] *= factor;
        }

        // Normalize the weights of the samples
        double sumWeights = weights.sumcol(0);
        weights /= sumWeights;

        // Store the weak learner weight in the ensemble
        _weakLearnerWeights[t] = alpha;

    }

}

double AdaBoost::classify(const Matrix& X){
    return 0.0;
}


double AdaBoost::test(Matrix const& X, Matrix const& Y) const{
    return 0.0;
}