#ifndef __WEAKLEARNER_H__
#define __WEAKLEARNER_H__

#include "matrix.h"

class WeakLearner{
    public:
        virtual void train(Matrix const& X, Matrix const& Y, std::size_t const maxiter) = 0;
        virtual double classify(const Matrix&) = 0;
        virtual Matrix classify_all(const Matrix&) const = 0;
        virtual double test(Matrix const& X, Matrix const& Y) const = 0;
        virtual Matrix weights() const = 0;
        virtual ~WeakLearner() {}  // Virtual destructor for polymorphic behavior.

};

#endif