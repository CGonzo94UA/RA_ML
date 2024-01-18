#include "matrix.h"
#include "perceptron.h"
#include "mlp.h"
#include "layer.h"
#include <iostream>
#include <vector>
#include <sstream>

using namespace std;

template <typename T>
std::string printVector(std::vector<T> const& v) {
    std::stringstream ss;
    ss << "[";
    for (auto const& e : v) {
        ss << e << ", ";
    }
    ss << "]";
    return ss.str();
}

void p(string s) {
    cout << s << endl;
}

int main(){
    auto [X, Y] = Perceptron::readFromCSV("datasets/xor.csv");
    int maxiter = 1000;

    p("Creating MLP");
    MLP_Builder builder = MLP_Builder();
    builder.addLayer(2, 2);
    builder.addLayer(2, 2);
    builder.addLayer(1, 2);
    // builder.addLayer(128, 128);         // First layer
    // builder.addLayer(64, 64);           // Second layer
    // builder.addLayer(1, 1);             // Third layer (output layer)

    p("Building MLP");
    MLP* mlp = builder.build();

    // MLP_Display::display(*mlp);

    p("Training MLP");
    mlp->train(X, Y, maxiter, 0.1);

    // MLP_Display::display(*mlp);

    p("Testing MLP");
    std::cout << "Accuracy: " << mlp->test(X, Y) << "\n";

    return 0;
}
