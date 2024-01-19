#include "genetic.h"

int main(){
    auto [X, Y] = Matrix::readFromCSV("datasets/xor2.csv");
    vector<int> topology = {2, 3, 1};

    auto createIndividual = std::bind(&Individual::createRandomIndividual, topology, X, Y);
    auto calculateFitness = std::bind(&Individual::calculateFitness, std::placeholders::_1, X, Y);

    // crea un elemento de la clase Genetic
    Genetic *genetic = new Genetic(35, createIndividual, calculateFitness);

    // inicializa el algoritmo genetico
    genetic->initialize();

    // imprime la generacion actual
    // std::cout << genetic->getGeneration() << "\n";
    // std::cout << genetic->getIndividuals().size() << "\n";

    std::vector<Individual*> individuals = genetic->getIndividuals();

    std::cout << "=======================================" << "\n";
    MLP* mlp = individuals[0]->getMLP();
    MLP_Display::display(*mlp);
    std::cout << "=======================================" << "\n";

    //Set weights
    /*
    std::vector<double> weights1 = {1.1,  -0.81, 0.65};
    Matrix Matrix1()
    std::vector<double> weights2 = {1., 1.9, -0.18};
    std::vector<double> weights3 = {6.3, -4.6, -6.1};
    const std::vector<Matrix>& weights();
    mlp->setWeights(weights);*/

    // evoluciona el algoritmo genetico
    for (int i = 0; i < 10; i++) {
        // imprime la generacion actual
        std::cout << "Generation: " << genetic->getGeneration() << "\n";
        genetic->evolve();

        individuals = genetic->getIndividuals();

        std::cout << "Precision: " << individuals[0]->getFitness() << "\n";

        std::cout << "=======================================" << "\n";
    }

    mlp = individuals[0]->getMLP();
    mlp->setInputs(X[0]);
    std::cout << "MLP inputs " << mlp->getInputs().size() << "\n";
    MLP_Display::display(*mlp);

    delete genetic;

    return 0;
}