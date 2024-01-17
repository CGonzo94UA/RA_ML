#ifndef __INDIVIDUAL_H__
#define __INDIVIDUAL_H__

#include "mlp.h"

/// <summary>
/// A class that represents an individual member of the population.
/// </summary>
class Individual 
{ 
private: 
    MLP* mlp;

    /// <summary>
    /// The fitness of this individual.
    /// </summary>
	int fitness; 
    
public:
    /// <summary>
    /// Initializes a new instance of Individual.
    /// </summary>
    Individual();
    
    /// <summary>
    /// Initializes a new instance of Individual.
    /// </summary>
    /// <param name="chromosome">The chromosome for the individual.</param>
	Individual(MLP* mlp); 

    /// <summary>
    /// Destroys this instance of Individual.
    /// </summary>
    ~Individual();

    /// <summary>
    /// Creates a new Individual resulting of the combination of this instance and another Individual.
    /// </summary>
    /// <param name="parent2">The other Individual to create the combination.</param>
	Individual* mate(const Individual &parent2, double mutationRate = 0.2, double mutationChance = 0.2); 

    /// <summary>
    /// Gets the fitness value of this Individual's chromosome.
    /// </summary>
    int getFitness() const { return fitness; }

    void setFitness(int fitness) { this->fitness = fitness; }

    MLP* getMLP() const { return mlp; }
};

std::random_device seed;
std::default_random_engine generator(seed());
std::uniform_real_distribution<double> distributionDouble(-0.5, 0.5);

#endif
