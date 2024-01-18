#include "matrix.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>

using namespace std;

// ============================================
// =============== Constructors ===============
Matrix::Matrix() {
    _rows = 0;
    _cols = 0;
    _matrix = vector<vector<double>>(0, vector<double>(0));
}

Matrix::Matrix(size_t rows, size_t cols) {
    _rows = rows;
    _cols = cols;
    _matrix = vector<vector<double>>(rows, vector<double>(cols));
}

Matrix::Matrix(size_t rows, size_t cols, vector<vector<double>> matrix) {
    _rows = rows;
    _cols = cols;
    _matrix = matrix;
}

Matrix::Matrix(size_t rows, size_t cols, vector<double> matrix) {
    _rows = rows;
    _cols = cols;
    _matrix = vector<vector<double>>(rows, vector<double>(cols));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++)
            _matrix[i][j] = matrix[i * cols + j];
    }
}

Matrix::Matrix(const Matrix& m) {
    _rows = m._rows;
    _cols = m._cols;
    _matrix = m._matrix;
}

// ============================================
// ================= Methods ==================
size_t Matrix::rows() const {
    return _rows;
}

size_t Matrix::cols() const {
    return _cols;
}

size_t Matrix::size() const {
    return _rows * _cols;
}

vector<vector<double>> Matrix::matrix() const {
    return _matrix;
}

vector<double> Matrix::getCol(size_t col) const {
    vector<double> result(_rows);
    for (size_t i = 0; i < _rows; i++) {
        result[i] = _matrix[i][col];
    }
    return result;
}

vector<double> Matrix::getRow(size_t row) const {
    vector<double> result(_cols);
    for (size_t i = 0; i < _cols; i++) {
        result[i] = _matrix[row][i];
    }
    return result;
}

pair<Matrix, Matrix> Matrix::divide(const double ratio, bool shuffle, unsigned seed) const {
    // Shuffle the matrix
    vector<vector<double>> shuffled = _matrix;
    if (shuffle) {
        std::default_random_engine rng(seed);
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
    }

    // Divide the matrix
    size_t const trainSize = static_cast<size_t>(shuffled.size() * ratio);
    size_t const testSize = shuffled.size() - trainSize;

    Matrix train(trainSize, _cols);
    Matrix test(testSize, _cols);

    for (size_t i = 0; i < trainSize; ++i) {
        train[i] = shuffled[i];
    }

    for (size_t i = 0; i < testSize; ++i) {
        test[i] = shuffled[i + trainSize];
    }

    return {train, test};
}

vector<int> Matrix::kfold(const int k, bool shuffle, unsigned seed) const {
    vector<int> indices(_rows);

    for (size_t i = 0; i < _rows; ++i) {
        indices[i] = i%k;
    }

    if (shuffle) {
        std::default_random_engine rng(seed);
        std::shuffle(indices.begin(), indices.begin() + _rows, rng);
    }

    return indices;
}

pair<Matrix, Matrix> Matrix::getFold(const vector<int> &folds, const int k, const int i) const {
    size_t trainSize = 0;
    size_t testSize = 0;

    for (size_t j = 0; j < _rows; ++j) {
        if (folds[j] == i) {
            ++testSize;
        } else {
            ++trainSize;
        }
    }

    Matrix train(trainSize, _cols);
    Matrix test(testSize, _cols);

    size_t trainIndex = 0;
    size_t testIndex = 0;

    for (size_t j = 0; j < _rows; ++j) {
        if (folds[j] == i) {
            test[testIndex] = _matrix[j];
            ++testIndex;
        } else {
            train[trainIndex] = _matrix[j];
            ++trainIndex;
        }
    }

    return {train, test};
}

// ============================================
// =============== Access op. =================
vector<double>& Matrix::operator[](size_t i) {
    return _matrix[i];
}

const vector<double>& Matrix::operator[](size_t i) const {
    return _matrix[i];
}

// ============================================
// =============== Modifiers ==================
Matrix& Matrix::apply(double (*f)(double)) {
    for (size_t i = 0; i < _rows; i++) {
        for (size_t j = 0; j < _cols; j++)
            _matrix[i][j] = f(_matrix[i][j]);
    }
    return *this;
}

Matrix& Matrix::apply(vector<double> (*f)(vector<double>)) {
    for (size_t i = 0; i < _rows; i++) {
        _matrix[i] = f(_matrix[i]);
    }
    return *this;
}

// ============================================
// =============== Operators ==================
Matrix Matrix::operator+(const Matrix& R) const {
    if (_rows != R._rows || _cols != R._cols)
        throw "Matrix dimensions must agree";
    Matrix result(_rows, _cols);
    for (size_t i = 0; i < _rows; i++) {
        for (size_t j = 0; j < _cols; j++)
            result[i][j] = _matrix[i][j] + R[i][j];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& R) const {
    if (_rows != R._rows || _cols != R._cols)
        throw "Matrix dimensions must agree";
    Matrix result(_rows, _cols);
    for (size_t i = 0; i < _rows; i++) {
        for (size_t j = 0; j < _cols; j++)
            result[i][j] = _matrix[i][j] - R[i][j];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& R) const {
    if (_cols != R._rows)
        throw "Matrix dimensions must agree";
    Matrix result(_rows, R._cols);
    for (size_t i = 0; i < _rows; i++) {
        for (size_t j = 0; j < R._cols; j++) {
            for (size_t k = 0; k < _cols; k++)
                result[i][j] += _matrix[i][k] * R[k][j];
        }
    }
    return result;
}

Matrix Matrix::operator*(double c) const {
    Matrix result(_rows, _cols);
    for (size_t i = 0; i < _rows; i++) {
        for (size_t j = 0; j < _cols; j++)
            result[i][j] = _matrix[i][j] * c;
    }
    return result;
}

Matrix Matrix::operator!=(const Matrix& R) const {
    if (_rows != R._rows || _cols != R._cols)
        throw "Matrix dimensions must agree";
    Matrix result(_rows, _cols);
    for (size_t i = 0; i < _rows; i++) {
        for (size_t j = 0; j < _cols; j++){
            //Result is 1.0 if the elements are different, 0.0 otherwise
            if (_matrix[i][j] != R[i][j]){
                result[i][j] = 1.0;
            }
            else{
                result[i][j] = 0.0;
            }
        }
            
    }
    return result;
}

Matrix& Matrix::operator+=(const Matrix& R) {
    if (_rows != R._rows || _cols != R._cols)
        throw "Matrix dimensions must agree";

    for (size_t i = 0; i < _rows; i++) {
        for (size_t j = 0; j < _cols; j++){
            _matrix[i][j] += R[i][j];
        }
    }
    return *this;  // Return a reference to the modified object
}

// ============================================
// =============== Operations =================
double Matrix::sumcol(const size_t col) const {
    double sum = 0;
    for (size_t i = 0; i < _rows; i++)
        sum += _matrix[i][col];
    return sum;
}

Matrix Matrix::transpose() const {
    Matrix result(_cols, _rows);
    for (size_t i = 0; i < _rows; i++) {
        for (size_t j = 0; j < _cols; j++)
            result[j][i] = _matrix[i][j];
    }
    return result;
}

Matrix Matrix::mult(size_t index, double scalar) const{
    Matrix result(1, _cols, _matrix[index]);
    result = result * scalar;
    return result;
}


// ============================================
// =============== Friend functions ===========
Matrix operator*(double c, const Matrix& R) {
    return R * c;
}

ostream& operator<<(ostream& os, const Matrix& R) {
    // start in a new line
    os << "\n";
    for (size_t i = 0; i < R._rows; i++) {
        for (size_t j = 0; j < R._cols; j++)
            os << R[i][j] << " ";
        os << endl;
    }
    return os;
}




std::pair<Matrix, Matrix> Matrix::readFromCSV(std::string const& filename){
    // Rellenar y devolver matriz de prueba
    std::ifstream file(filename);
    std::string line;

    std::vector<double> vectorX;
    std::vector<double> vectorY;
    std::size_t rowCount = 0;
    size_t num_inputs = 0;

    while (std::getline(file, line, '\n')) {
        std::stringstream ss(line);
        //std::cout << line << '\n';
        std::vector<std::string> tokens;
        
        // Dividir la línea en tokens utilizando el delimitador ","
        while (std::getline(ss, line, ',')) {
            tokens.push_back(line);
        }
        num_inputs = tokens.size() -1;

        // Leer los primeros 7 valores y colocar un 1 en la primera posición en el vectorX
        vectorX.push_back(1.0);
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            double value = std::stod(tokens[i]);
            //std::cout << "Value "<< value << "\n";
            if(i < num_inputs){
                vectorX.push_back(value);
            }else{
                // Leer último valor en el vectorY
                vectorY.push_back(value);
            }
            
        }

        // Incrementar el contador de filas
        ++rowCount;
    }

    Matrix X{rowCount, num_inputs +1, vectorX};
    Matrix Y{rowCount, 1, vectorY};
    
    return std::make_pair(X, Y);
     
}