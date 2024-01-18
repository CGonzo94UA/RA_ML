#ifndef __INDIVIDUAL_H__
#define __INDIVIDUAL_H__

#include "mlp.h"
#include "randonn_generator.h"

/// ================
/// A class that represents an individual member of the population.
/// ================
class Individual 
{ 
private: 
    MLP* mlp;

    /// ================
    /// The fitness of this individual.
    /// ================
	double fitness; 

    static Randonn_generator generator;
    
    /// ================
    /// Initializes a new instance of Individual.
    /// ================
    Individual();
public:
    
    /// ================
    /// Initializes a new instance of Individual.
    /// ================
	Individual(MLP* mlp); 

    /// ================
    /// Destroys this instance of Individual.
    /// ================
    ~Individual();

    /// ================
    /// Creates a new Individual resulting of the combination of this instance and another Individual.
    /// ================
	Individual* mate(const Individual &parent2, double mutationChance = 0.); 

    /// ================
    /// Gets the fitness value of this Individual's chromosome.
    /// ================
    double getFitness() const { return fitness; }
    void setFitness(double fitness) { this->fitness = fitness; }
    MLP* getMLP() const { return mlp; }
    Individual* clone() const;
    static Individual* createRandomIndividual(vector<int> topology, const Matrix& X, const Matrix& Y);
    double calculateFitness(const Matrix& X, const Matrix& Y);

};

#endif
