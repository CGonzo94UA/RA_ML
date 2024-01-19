#ifndef __GENETIC_H__
#define __GENETIC_H__

#include "individual.h"
#include "randonn_generator.h"
#include <vector>
#include <functional>

class Genetic
{
public:
    Genetic(int population, 
        std::function<Individual*()> createIndividual, std::function<double(Individual*)> calculateFitness);
    ~Genetic();
    void initialize();
    void evolve();

    std::vector<Individual*> getIndividuals() { return individuals; }
    int getGeneration() { return generation; }

private:
    Randonn_generator generator;

    std::function<Individual*()> createIndividual;
    std::function<double(Individual*)> calculateFitness;
    vector<int> topology;
    Matrix X;
    Matrix Y;
    int population;
    std::vector<Individual*> individuals;
    int generation;

    std::vector<Individual*> bestIndividuals(double n = 0.5);
    std::vector<Individual*> nextGeneration(double mutationRate = 0.9);
    
    
};

#endif
