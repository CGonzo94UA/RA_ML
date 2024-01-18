#include "genetic.h"
#include "individual.h"
#include "mlp.h"

/* ============================================
*  Genetic
*  Represents the genetic algorithm
* ============================================
*/

// ============================================
// =============== Constructors ===============
/// @brief Initializes a new instance of Genetic.
/// @param population The size of the population.
/// @param createIndividual The function to create an individual.
Genetic::Genetic(int population, 
    std::function<Individual*()> createIndividual, std::function<double(Individual*)> calculateFitness)
{
    this->population = population;
    this->createIndividual = createIndividual;
    this->calculateFitness = calculateFitness;
    this->generation = 1;
}

// ========================================
// ============== Destructor ==============
Genetic::~Genetic()
{
    for(int i = 0; i < individuals.size();i++)
    {
        delete individuals[i];
    }
    
    individuals.clear();
    
}

// ============================================
// ================= Methods ==================
/// @brief Initializes the genetic algorithm by creating a population of random individuals
void Genetic::initialize()
{
    individuals.clear();
    individuals.resize(population);
    
    for(int i =0;i<population;i++)
    { 
        individuals[i] = createIndividual();
    }
}

/// @brief Updates the genetic algorithm by evolving the population
void Genetic::evolve()
{
    std::vector<Individual*> nextGen = nextGeneration();
    
    std::cout << "Best fitness: " << individuals[0]->getFitness() << std::endl;

    for(int i = 0; i < individuals.size();i++)
    {
        // std::cout << "Individual " << i << " fitness: " << individuals[i]->getFitness() << std::endl;
        delete individuals[i];
        // std::cout << "Individual " << i << " deleted" << std::endl;
    }
    
    individuals.clear();
    individuals = nextGen;    
    ++generation;
}

/// @brief Gets the best individuals of the population
/// @param n The percentage of individuals to get
/// @return The best individuals of the population
std::vector<Individual*> Genetic::bestIndividuals(double n)
{
    std::vector<Individual*> best;
    
    std::sort(individuals.begin(), individuals.end(), [](Individual* a, Individual* b) -> bool
    {
        return a->getFitness() > b->getFitness();
    });
    
    for(int i = 0; i < individuals.size() * n; ++i)
    {
        best.push_back(individuals[i]->clone());
    }
    
    return best;
}

/// @brief Gets the next generation of the population
/// @return The next generation of the population
std::vector<Individual*> Genetic::nextGeneration(double n)
{
    // getting the best individuals
    std::vector<Individual*> nextGen = bestIndividuals(n);

    // generating the rest of the population
    // based on the best individuals
    for(int i = 0; i < individuals.size() * (1 - n); ++i)
    {
        // random mating
        int r = generator.randomInt(0, individuals.size() * n);
        int r2 = generator.randomInt(0, individuals.size() * n);

        Individual* child = individuals[r]->mate(*individuals[r2]);
        double fitness = calculateFitness(child);
        child->setFitness(fitness);
        nextGen.push_back(child);
    }
    
    return nextGen;
}
