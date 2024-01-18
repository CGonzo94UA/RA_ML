#include "individual.h"

/// <summary>
/// Initializes a new instance of Individual.
/// </summary>
Individual::Individual()
{
    this->mlp = NULL;
    this->fitness = 0;

}

/// <summary>
/// Initializes a new instance of Individual.
/// </summary>
Individual::Individual(MLP* mlp) : Individual()
{ 
    this->mlp = mlp;
}

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
Individual* Individual::mate(const Individual &par2, double mutationChance) 
{ 
    Individual* child = new Individual(this->getMLP()->clone());

    // chromosome for offspring
    std::vector<Matrix> childWeights = mlp->getWeights();
    std::vector<Matrix> parentWeights = par2.mlp->getWeights();
    
    int cut = generator.randomDouble(0.0, 1.0) * childWeights.size();
    
    // std::cout << "cut1: " << cut << std::endl;
    for(int i=0; i < cut; ++i)
    {
        double p = generator.randomDouble(0, 1);
        if(p > mutationChance) {
            for (int j = 0; j < childWeights[i].rows(); j++) {
                for (int k = 0; k < childWeights[i].cols(); k++) {
                    childWeights[i][j][k] = generator.randomDouble(-0.5, 0.5);
                }
            }
        }
    }
    
    // std::cout << "cut2: " << cut << std::endl;
    for(int i = cut; i < childWeights.size(); ++i)
    {
        double p = generator.randomDouble(0, 1);
        // std::cout << "p: " << p << std::endl;
        if(p > mutationChance) {
            // std::cout << "mutando" << std::endl;
            for (int j = 0; j < childWeights[i].rows(); ++j) {
                // std::cout << "j: " << j << std::endl;
                for (int k = 0; k < childWeights[i].cols(); ++k) {
                    // std::cout << "k: " << k << std::endl;
                    childWeights[i][j][k] = generator.randomDouble(-0.5, 0.5);
                }
            }
        } else {
            // std::cout << "no mutando" << std::endl;
            childWeights[i] = parentWeights[i];
        }
    }
    
    // std::cout << "setting weights" << std::endl;
    child->mlp->setWeights(childWeights);

    return child;
}

Individual* Individual::clone() const
{
    Individual* clone = new Individual(this->getMLP()->clone());
    clone->setFitness(this->getFitness());

    return clone;
}

Randonn_generator Individual::generator = Randonn_generator();

Individual* Individual::createRandomIndividual(vector<int> topology, const Matrix& X, const Matrix& Y)
{
    MLP_Builder builder = MLP_Builder();
    // topology es un vector de enteros que contiene el numero de neuronas de cada capa
    // el primer valor es el numero de neuronas de la capa de entrada
    // los siguientes valores son el numero de neuronas de las capas ocultas
    for (int i = 1; i < topology.size(); i++) {
        builder.addLayer(topology[i], topology[i-1]);
    }

    MLP* mlp = builder.build();

    Individual* ind = new Individual(mlp);
    ind->setFitness(ind->calculateFitness(X, Y));

    return ind;
}

double Individual::calculateFitness(const Matrix& X, const Matrix& Y) {
    double acc = mlp->test(X, Y);
    // std::cout << "Accuracy: " << acc << "\n";
    return acc;
}