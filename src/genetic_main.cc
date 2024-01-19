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
    
    // std::vector<double> w1 = {4.4,  -0.80, 1.3};
    // Matrix Matrix1(1, 3, w1);
    // std::vector<double> w2 = {2.6, -2, -0.6};
    // Matrix Matrix2(1, 3, w2);
    // std::vector<double> w3 = {3.4, 0.54, 1.2};
    // Matrix Matrix3(1, 3, w3);
    // std::vector<double> w4 = {-1.9, -5.6, 2.6, 5.7};
    // Matrix Matrix4(1, 4, w4);
    // const std::vector<Matrix>& weights = {Matrix1, Matrix2, Matrix3, Matrix4};
    // mlp->setWeights(weights);

    // evoluciona el algoritmo genetico
    for (int i = 0; i < 100; i++) {
        // imprime la generacion actual
        std::cout << "Generation: " << genetic->getGeneration() << "\n";
        genetic->evolve();

        individuals = genetic->getIndividuals();

        std::cout << "Precision: " << individuals[0]->getFitness() << "\n";

        std::cout << "=======================================" << "\n";
    }

    mlp = individuals[0]->getMLP();
    // mlp->setInputs(X[0]);
    // std::cout << "MLP inputs " << mlp->getInputs().size() << "\n";
    MLP_Display::display(*mlp);

    delete genetic;

    return 0;
}