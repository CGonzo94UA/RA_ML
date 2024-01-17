#ifndef __INDIVIDUAL_H__
#define __INDIVIDUAL_H__

#include "mlp.h"
#include "randonn_generator.h"

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

    static Randonn_generator generator;
    
    /// <summary>
    /// Initializes a new instance of Individual.
    /// </summary>
    Individual();
public:
    
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
	Individual* mate(const Individual &parent2, double mutationChance = 0.95); 

    /// <summary>
    /// Gets the fitness value of this Individual's chromosome.
    /// </summary>
    int getFitness() const { return fitness; }

    void setFitness(int fitness) { this->fitness = fitness; }

    MLP* getMLP() const { return mlp; }

    Individual* clone() const;
};

#endif
