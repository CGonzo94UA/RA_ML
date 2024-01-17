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
    // std::cout << "Generation " << generation << std::endl;
    std::vector<Individual*> nextGen = nextGeneration();
    
    for(int i = 0; i < individuals.size();i++)
    {
        // std::cout << "Individual " << i << " fitness: " << individuals[i]->getFitness() << std::endl;
        delete individuals[i];
        // std::cout << "Individual " << i << " deleted" << std::endl;
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
    
    // std::cout << "Sorting" << std::endl;
    std::sort(individuals.begin(), individuals.end(), [](Individual* a, Individual* b) -> bool
    {
        return a->getFitness() > b->getFitness();
    });
    
    // std::cout << "Best individuals" << std::endl;
    for(int i = 0; i < individuals.size() * n; ++i)
    {
        // std::cout << "Best individual " << i << " fitness: " << individuals[i]->getFitness() << std::endl;
        nextGen.push_back(individuals[i]->clone());
    }
    
    // std::cout << "Mating" << std::endl;
    for(int i = 0; i < individuals.size() * (1 - n); ++i)
    {
        int r = generator.randomInt(0, individuals.size() - 1);
        int r2 = generator.randomInt(0, individuals.size() - 1);

        // std::cout << "Mating " << r << " and " << r2 << std::endl;
        Individual* child = individuals[r]->mate(*individuals[r2]);
        // std::cout << "Child created" << std::endl;
        nextGen.push_back(child);
    }

    // for (int i = 0; i < 5; ++i)
    // {
    //     std::cout << "BEST NEXT GENERATION Individual " << i << " fitness: " << nextGen[i]->getFitness() << std::endl;
    // }
    
    return nextGen;
}
