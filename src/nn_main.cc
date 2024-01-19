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
    auto [X, Y] = Matrix::readFromCSV("datasets/xor2.csv");
    int maxiter = 1000;

    p("Creating MLP");
    MLP_Builder builder = MLP_Builder();
    builder.addLayer(3, 2);
    builder.addLayer(1, 3);


    p("Building MLP");
    MLP* mlp = builder.build();

    MLP_Display::display(*mlp);

    p("Training MLP");
    mlp->train(X, Y, maxiter, 0.2);

    // MLP_Display::display(*mlp);

    p("Testing MLP");
    std::cout << "Accuracy: " << mlp->test(X, Y) << "\n";

    delete mlp;

    return 0;
}
