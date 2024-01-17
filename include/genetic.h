#ifndef __GENETIC_H__
#define __GENETIC_H__

#include "individual.h"
#include "randonn_generator.h"
#include <vector>
#include <functional>

class Genetic
{
public:
    Genetic(int population, std::function<Individual*()> createRandomIndividual);
    ~Genetic();
    void initialize();
    void evolve();

    std::vector<Individual*> getIndividuals() { return individuals; }
    int getGeneration() { return generation; }

private:
    Randonn_generator generator;

    std::function<Individual*()> createRandomIndividual;
    int population;
    std::vector<Individual*> individuals;
    int generation;

    std::vector<Individual*> bestIndividuals(double n = 0.1);
    std::vector<Individual*> nextGeneration(double n = 0.1);
    
    
};

#endif
