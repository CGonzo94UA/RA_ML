#include "genetic.h"

Individual* f()
{
    MLP_Builder builder = MLP_Builder();
    builder.addLayer(2, 3);
    builder.addLayer(1, 2);

    MLP* mlp = builder.build();

    Individual* ind = new Individual(mlp);

    ind->setFitness(ind->getMLP()->getPuntuacion());

    return ind;
}

int main(){

    // crea un elemento de la clase Genetic
    Genetic *genetic = new Genetic(50, f);

    // inicializa el algoritmo genetico
    genetic->initialize();

    // imprime la generacion actual
    // std::cout << genetic->getGeneration() << "\n";
    // std::cout << genetic->getIndividuals().size() << "\n";

    // evoluciona el algoritmo genetico
    for (int i = 0; i < 50; i++) {
        genetic->evolve();
        // imprime la generacion actual
        std::cout << "Generation: " << genetic->getGeneration() << "\n";

        // imprime los individuos de la generacion actual
        std::vector<Individual*> individuals = genetic->getIndividuals();
        // std::cout << individuals.size() << "\n";

        std::cout << "Fitness: " << "\n";
        for (int i = 0; i < 5; i++) {
            // std::cout << "Individual " << i << " fitness: " << std::endl;
            // std::cout << individuals[i]->getFitness() << "\n";
            std::cout << i << ": " << individuals[i]->getFitness() << "\n";
        }

        std::cout << "=======================================" << "\n";
    }



    delete genetic;

    return 0;
}