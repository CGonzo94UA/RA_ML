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

MLP_Builder buildXOR(){
    MLP_Builder builder = MLP_Builder();
    builder.addLayer(2, 2);
    builder.addLayer(2, 2);
    builder.addLayer(1, 2);
    return builder;
}

MLP_Builder buildXOR2(){
    MLP_Builder builder = MLP_Builder();
    builder.addLayer(2, 2);
    builder.addLayer(2, 2);
    builder.addLayer(1, 2);
    return builder;
}

MLP_Builder build7entradas(){
    MLP_Builder builder = MLP_Builder();
    builder.addLayer(7, 7);
    builder.addLayer(4, 7);
    builder.addLayer(1, 4);
    return builder;
}

MLP_Builder build3entradas(){
    MLP_Builder builder = MLP_Builder();
    builder.addLayer(3, 3);
    builder.addLayer(3, 3);
    builder.addLayer(2, 3);
    builder.addLayer(1, 2);
    return builder;
}

MLP_Builder buildVersicolor(){
    MLP_Builder builder = MLP_Builder();
    builder.addLayer(4, 4);
    builder.addLayer(2, 4);
    builder.addLayer(1, 2);
    return builder;
}

MLP_Builder build128entradas(){
    MLP_Builder builder = MLP_Builder();
    builder.addLayer(128, 128);
    builder.addLayer(128, 128);
    builder.addLayer(64, 128);
    builder.addLayer(32, 64);
    builder.addLayer(1, 32);
    return builder;
}

int main(){
    auto [X, Y] = Matrix::readFromCSV("datasets/xor2.csv");
    int maxiter = 1000;


    p("Creating MLP");
    MLP_Builder builder = buildXOR2();

    p("Building MLP");
    MLP* mlp = builder.build();

    // MLP_Display::display(*mlp);

    p("Training MLP");
    mlp->train(X, Y, maxiter, 69420);

    // MLP_Display::display(*mlp);

    p("Testing MLP");
    std::cout << "Accuracy: " << mlp->test(X, Y) << "\n";

    delete mlp;

    return 0;
}
