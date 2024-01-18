#include "individual.h"

/* ============================================
*  Individual
*  Represents an individual of the population: a MLP and its fitness
* ============================================
*/

// ============================================
// =============== Constructors ===============
/// @brief Initializes a new instance of Individual.
Individual::Individual()
{
    this->mlp = NULL;
    this->fitness = 0.0;

}

/// @brief Initializes a new instance of Individual with a MLP.
Individual::Individual(MLP* mlp) : Individual()
{ 
    this->mlp = mlp;
    this->fitness = 0.0;
}

// ============================================
// =============== Destructor ===============
/// @brief Destroys this instance of Individual.
Individual::~Individual()
{
    if(mlp != NULL)
        delete mlp;
    mlp = NULL;
}

// ============================================
// ================= Methods ==================
/// @brief Creates a new Individual resulting of the combination of this instance and another Individual.
/// @param par2 The other Individual.
/// @param mutationChance The chance of mutation.
/// @return The new Individual.
Individual* Individual::mate(const Individual &par2, double mutationChance) 
{ 
    Individual* child = new Individual(this->getMLP()->clone());

    std::vector<Matrix> childWeights = mlp->getWeights();
    std::vector<Matrix> parentWeights = par2.mlp->getWeights();
    
    // Calculate the cut point
    int cut = round(generator.randomDouble(0.0, 1.0) * childWeights.size());
    
    //std::cout << "cut: " << cut << std::endl;
    for(int i=0; i < cut; ++i)
    {
        double p = generator.randomDouble(0, 1);
        if(p > mutationChance) {
            for (int j = 0; j < childWeights[i].rows(); j++) {
                for (int k = 0; k < childWeights[i].cols(); k++) {
                    double random = generator.randomDouble(-5, 5);
                    //std::cout << "random: " << random << std::endl;
                    childWeights[i][j][k] = random;
                }
            }
        }
    }
    
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
                    childWeights[i][j][k] = generator.randomDouble(-5, 5);
                }
            }
        } else {
            // std::cout << "no mutando" << std::endl;
            childWeights[i] = parentWeights[i];
        }
    }
    
    std::cout << "setting weights" << std::endl;
    for(size_t i = 0; i < childWeights.size(); ++i)
    {
        std::cout << "layer " << i << std::endl;
        std::cout << childWeights[i] << std::endl;
    }

    child->mlp->setWeights(childWeights);

    return child;
}

/// @brief Clones this instance of Individual.
/// @return The new Individual.
Individual* Individual::clone() const
{
    Individual* clone = new Individual(this->getMLP()->clone());
    clone->setFitness(this->getFitness());

    return clone;
}

Randonn_generator Individual::generator = Randonn_generator();

/// @brief Creates a new random Individual.
/// @param topology The topology of the MLP. Int vector with the number of neurons of each layer.
/// @param X The input matrix.
/// @param Y The classes matrix.
/// @return The new Individual.
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

/// @brief Calculates the fitness of this instance of Individual using the test method of the MLP.
/// @return The fitness of this instance of Individual (accuracy of the MLP).
double Individual::calculateFitness(const Matrix& X, const Matrix& Y) {
    double acc = mlp->test(X, Y);
    //std::cout << "Accuracy: " << acc << "\n";
    return acc;
}