#include "individual.h"

/// <summary>
/// Initializes a new instance of Individual.
/// </summary>
Individual::Individual()
{
    this->mlp = NULL;
    this->fitness = 0;

    // std::random_device seed;
    // this->generator = std::default_random_engine(seed());
    // this->distributionDouble = std::uniform_real_distribution<double>(-0.5, 0.5);
};

/// <summary>
/// Initializes a new instance of Individual.
/// </summary>
Individual::Individual(MLP* mlp) : Individual()
{ 
    this->mlp = mlp;
};

/// <summary>
/// Destroys this instance of Individual.
/// </summary>
Individual::~Individual()
{
    if(mlp != NULL)
        delete mlp;
    mlp = NULL;
}

/// <summary>
/// Creates a new Individual resulting of the combination of this instance and another Individual.
/// </summary>
/// <param name="parent2">The other Individual to create the combination.</param>
Individual* Individual::mate(const Individual &par2, double mutationRate, double mutationChance) 
{ 
    Individual* child = new Individual();

    // chromosome for offspring
    std::vector<Matrix> childWeights = mlp->getWeights();
    std::vector<Matrix> parentWeights = par2.mlp->getWeights();
    
    int cut = mutationRate * childWeights.size();
    
    for(int i=0;i<cut;i++)
    {
        double p = distributionDouble(generator);
        if(p > mutationChance) {
            for (int j = 0; j < childWeights[i].rows(); j++) {
                for (int k = 0; k < childWeights[i].cols(); k++) {
                    childWeights[i][j][k] = distributionDouble(generator);
                }
            }
        }
    }
    
    for(int i=cut; i<childWeights.size();i++)
    {
        double p = distributionDouble(generator);
        if(p > mutationChance) {
            for (int j = 0; j < childWeights[i].rows(); j++) {
                for (int k = 0; k < childWeights[i].cols(); k++) {
                    childWeights[i][j][k] = distributionDouble(generator);
                }
            }
        } else {
            childWeights[i] = parentWeights[i];
        }
    }
    
    child->mlp->setWeights(childWeights);

    child->setFitness(child->getMLP()->getPuntuacion());

    return child;
}