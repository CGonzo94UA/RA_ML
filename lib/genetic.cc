#include "genetic.h"
#include "individual.h"
#include "mlp.h"

Genetic::Genetic(int population, std::function<Individual*()> createRandomIndividual)
{
    this->population = population;
    this->createRandomIndividual = createRandomIndividual;
    this->generation = 1;
}

Genetic::~Genetic()
{
    for(int i = 0; i < individuals.size();i++)
    {
        delete individuals[i];
    }
    
    individuals.clear();
    
    std::cout<<"Genetic destroyed"<<std::endl;
}

/// @brief Initializes the genetic algorithm
/// @details Initializes the genetic algorithm by creating a population of random individuals
void Genetic::initialize()
{
    individuals.clear();
    individuals.resize(population);
    
    for(int i =0;i<population;i++)
    { 
        individuals[i] = createRandomIndividual();
    }
}

/// @brief Updates the genetic algorithm
/// @details Updates the genetic algorithm by evolving the population
void Genetic::evolve()
{
    // std::vector<Individual*> best = bestIndividuals();
    std::vector<Individual*> nextGen = nextGeneration();
    
    for(int i = 0; i < individuals.size();i++)
    {
        delete individuals[i];
    }
    
    individuals.clear();
    
    individuals = nextGen;
    
    // for(int i = 0; i < best.size();i++)
    // {
    //     individuals.push_back(best[i]);
    // }
    
    generation++;
}

/// @brief Gets the best individuals of the population
/// @details Gets the best individuals of the population
std::vector<Individual*> Genetic::bestIndividuals(double n)
{
    std::vector<Individual*> best;
    
    std::sort(individuals.begin(), individuals.end(), [](Individual* a, Individual* b) -> bool
    {
        return a->getFitness() > b->getFitness();
    });
    
    for(int i = 0; i < individuals.size() * n;i++)
    {
        best.push_back(individuals[i]);
    }
    
    return best;
}

/// @brief Gets the next generation of the population
/// @details Gets the next generation of the population
std::vector<Individual*> Genetic::nextGeneration(double n)
{
    std::vector<Individual*> nextGen;
    std::uniform_real_distribution<double> dist(0.0, individuals.size() - 1.0);
    
    std::sort(individuals.begin(), individuals.end(), [](Individual* a, Individual* b) -> bool
    {
        return a->getFitness() > b->getFitness();
    });
    
    for(int i = 0; i < individuals.size() * n; ++i)
    {
        nextGen.push_back(individuals[i]);
    }
    
    for(int i = 0; i < individuals.size() * (1 - n); ++i)
    {
        int r = dist(generator);
        int r2 = dist(generator);

        
        Individual* child = individuals[r]->mate(*individuals[r2]);
        nextGen.push_back(child);
    }
    
    return nextGen;
}
