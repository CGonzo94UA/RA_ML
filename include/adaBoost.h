#ifndef __ADABOOST_H__
#define __ADABOOST_H__

#include "weakLearner.h"
#include "matrix.h"

// Clase que implementa el algoritmo de AdaBoost
class AdaBoost{
    public:
        AdaBoost(std::size_t const num_weakLearners);
        ~AdaBoost();

        std::vector<double> weights() const { return _weakLearnerWeights; }
        void train(Matrix const& X, Matrix const& Y, std::size_t const maxiter);
        double classify(const Matrix&);
        double test(Matrix const& X, Matrix const& Y) const;

    private:
        // Vector de punteros a los weak learners
        std::vector<WeakLearner*> weakLearners;
        // Vector con el peso asignado a cada weak learner
        std::vector<double> _weakLearnerWeights;
        // Número de weak learners
        size_t _numWeakLearners;

        // Métodos privados
        Matrix generateWeights(std::size_t const num_weights);
        void createWeakLearners(std::size_t const num_features);

};

#endif

