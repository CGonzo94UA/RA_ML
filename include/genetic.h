#ifndef __GENETIC_H__
#define __GENETIC_H__

#include <vector>
#include "individual.h"
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
    std::function<Individual*()> createRandomIndividual;
    int population;
    std::vector<Individual*> individuals;
    int generation;

    std::vector<Individual*> bestIndividuals(double n = 0.1);
    std::vector<Individual*> nextGeneration(double n = 0.1);
    
    
};

#endif
