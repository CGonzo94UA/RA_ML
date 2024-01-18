#include "genetic.h"

int main(){
    auto [X, Y] = Matrix::readFromCSV("datasets/nonlinear_dataset.csv");
    vector<int> topology = {2, 3, 1};

    // crea un elemento de la clase Genetic
    Genetic *genetic = new Genetic(100, Individual::createRandomIndividual, topology, X, Y);

    // inicializa el algoritmo genetico
    genetic->initialize();

    // imprime la generacion actual
    // std::cout << genetic->getGeneration() << "\n";
    // std::cout << genetic->getIndividuals().size() << "\n";

    std::vector<Individual*> individuals = genetic->getIndividuals();
    // std::cout << "Pesos: " << "\n";
    // for (int i = 0; i < 5; i++) {
    //     std::cout << "Individual " << i << std::endl;
    //     // std::cout << individuals[i]->getFitness() << "\n";
    //     for (int j = 0; j < individuals[i]->getMLP()->getWeights().size(); j++) {
    //         std::cout << individuals[i]->getMLP()->getWeights()[j] << "\n";
    //     }
    //     std::cout << "------------------------------------" << "\n";
    // }

    // std::cout << "=======================================" << "\n";

    // evoluciona el algoritmo genetico
    for (int i = 0; i < 1000; i++) {
        // imprime la generacion actual
        std::cout << "Generation: " << genetic->getGeneration() << "\n";
        genetic->evolve();

        individuals = genetic->getIndividuals();

        // std::cout << "Pesos: " << "\n";
        // for (int i = 0; i < 5; i++) {
        //     std::cout << "Individual " << i << std::endl;
        //     // std::cout << individuals[i]->getFitness() << "\n";
        //     for (int j = 0; j < individuals[i]->getMLP()->getWeights().size(); j++) {
        //         std::cout << individuals[i]->getMLP()->getWeights()[j] << "\n";
        //     }
        //     std::cout << "------------------------------------" << "\n";
        // }

        // print precision final
        // if (i % 10 == 0)
        std::cout << "Precision: " << individuals[0]->getFitness() << "\n";

        std::cout << "=======================================" << "\n";
    }



    delete genetic;

    return 0;
}