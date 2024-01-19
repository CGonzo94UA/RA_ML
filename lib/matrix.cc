#include "matrix.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>

using namespace std;

/* ============================================
*  Matrix
*  Represents a matrix and its operations
* ============================================
*/

// ============================================
// =============== Constructors ===============
/// @brief Default constructor of the class Matrix
Matrix::Matrix() {
    _rows = 0;
    _cols = 0;
    _matrix = vector<vector<double>>(0, vector<double>(0));
}

/// @brief Constructor of the class Matrix
/// @param rows The number of rows of the matrix
/// @param cols The number of columns of the matrix
Matrix::Matrix(size_t rows, size_t cols) {
    _rows = rows;
    _cols = cols;
    _matrix = vector<vector<double>>(rows, vector<double>(cols));
}

/// @brief Constructor of the class Matrix
/// @param rows The number of rows of the matrix
/// @param cols The number of columns of the matrix
/// @param matrix The matrix in vector of vectors form
Matrix::Matrix(size_t rows, size_t cols, vector<vector<double>> matrix) {
    _rows = rows;
    _cols = cols;
    _matrix = matrix;
}

/// @brief Constructor of the class Matrix
/// @param rows The number of rows of the matrix
/// @param cols The number of columns of the matrix
/// @param matrix The matrix in vector form
Matrix::Matrix(size_t rows, size_t cols, vector<double> matrix) {
    _rows = rows;
    _cols = cols;
    _matrix = vector<vector<double>>(rows, vector<double>(cols));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++)
            _matrix[i][j] = matrix[i * cols + j];
    }
}

/// @brief Constructor of the class Matrix
/// @param m The matrix to copy
Matrix::Matrix(const Matrix& m) {
    _rows = m._rows;
    _cols = m._cols;
    _matrix = m._matrix;
}

// ============================================
// ================= Methods ==================
/// @brief Returns the number of rows of the matrix
size_t Matrix::rows() const {
    return _rows;
}

/// @brief Returns the number of columns of the matrix
size_t Matrix::cols() const {
    return _cols;
}

/// @brief Returns the size of the matrix
size_t Matrix::size() const {
    return _rows * _cols;
}

/// @brief Returns the matrix in vector of vectors form
vector<vector<double>> Matrix::matrix() const {
    return _matrix;
}

/// @brief Returns the matrix in vector form
vector<double> Matrix::getCol(size_t col) const {
    vector<double> result(_rows);
    for (size_t i = 0; i < _rows; i++) {
        result[i] = _matrix[i][col];
    }
    return result;
}

/// @brief Returns the row of the matrix in vector form
vector<double> Matrix::getRow(size_t row) const {
    vector<double> result(_cols);
    for (size_t i = 0; i < _cols; i++) {
        result[i] = _matrix[row][i];
    }
    return result;
}

/// @brief Divides the matrix in two matrices, one for training and one for testing
/// @param ratio The ratio of the division
/// @param shuffle If true, the matrix is shuffled before dividing
/// @param seed The seed for the random number generator
/// @return A pair of matrices, the first one for training and the second one for testing
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

/// @brief Divides the matrix in k groups for k-fold cross validation
/// @param k The number of folds
/// @param shuffle If false, the matrix is not shuffled before dividing
/// @param seed The seed for the random number generator
/// @return A vector with the indices of the folds 
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

/// @brief Divides the matrix in two matrices, one for training and one for testing
/// based on the indices of the folds
/// @param folds The indices of the folds, obtained with the kfold method
/// @param k The number of folds
/// @param i The index of the fold to get
/// @return A pair of matrices, the first one for training and the second one for testing
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
/// @brief Returns the row of the matrix in vector form
vector<double>& Matrix::operator[](size_t i) {
    return _matrix[i];
}

/// @brief Returns the row of the matrix in vector form
/// @details This method is used when the matrix is constant
const vector<double>& Matrix::operator[](size_t i) const {
    return _matrix[i];
}

// ============================================
// =============== Modifiers ==================
/// @brief Apply a function to each element of the matrix
/// @param f The function to apply
/// @return A reference to the modified object
Matrix& Matrix::apply(double (*f)(double)) {
    for (size_t i = 0; i < _rows; i++) {
        for (size_t j = 0; j < _cols; j++)
            _matrix[i][j] = f(_matrix[i][j]);
    }
    return *this;
}

/// @brief Apply a function to each row of the matrix
/// @param f The function to apply
/// @return A reference to the modified object
Matrix& Matrix::apply(vector<double> (*f)(vector<double>)) {
    for (size_t i = 0; i < _rows; i++) {
        _matrix[i] = f(_matrix[i]);
    }
    return *this;
}

// ============================================
// =============== Operators ==================
/// @brief Adds two matrices
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

/// @brief Subtracts two matrices
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

/// @brief Multiplies two matrices
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

/// @brief Multiplies a matrix by a scalar
Matrix Matrix::operator*(double c) const {
    Matrix result(_rows, _cols);
    for (size_t i = 0; i < _rows; i++) {
        for (size_t j = 0; j < _cols; j++)
            result[i][j] = _matrix[i][j] * c;
    }
    return result;
}

/// @brief  Compares two matrices
/// @param R The matrix to compare
/// @return A new matrix with 1 if the elements are different and 0 if they are equal
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

/// @brief Adds a matrix to the current matrix
/// @param R The matrix to add
/// @return A reference to the modified matrix
Matrix& Matrix::operator+=(const Matrix& R) {
    if (_rows != R._rows || _cols != R._cols)
        throw "Matrix dimensions must agree";

    for (size_t i = 0; i < _rows; i++) {
        for (size_t j = 0; j < _cols; j++){
            _matrix[i][j] += R[i][j];
        }
    }
    return *this;  // Return a reference to the modified matrix
}

// ============================================
// =============== Operations =================
/// @brief Returns the sum of the elements of one column of the matrix
/// @param col The column to sum
/// @return The sum of the elements of the column
double Matrix::sumcol(const size_t col) const {
    double sum = 0;
    for (size_t i = 0; i < _rows; i++)
        sum += _matrix[i][col];
    return sum;
}

/// @brief Transposes the matrix
/// @return The transposed matrix
Matrix Matrix::transpose() const {
    Matrix result(_cols, _rows);
    for (size_t i = 0; i < _rows; i++) {
        for (size_t j = 0; j < _cols; j++)
            result[j][i] = _matrix[i][j];
    }
    return result;
}

/// @brief Multiply a row of the matrix by a scalar
/// @param index The index of the row to multiply
/// @param scalar The scalar to multiply
/// @return A new matrix with the row multiplied by the scalar
Matrix Matrix::mult(size_t index, double scalar) const{
    Matrix result(1, _cols, _matrix[index]);
    result = result * scalar;
    return result;
}


// ============================================
// =============== Friend functions ===========
/// @brief Multiplies a scalar by a matrix
/// @param c The scalar
/// @param R The matrix
/// @return A new matrix with the scalar multiplied by the matrix
Matrix operator*(double c, const Matrix& R) {
    return R * c;
}

/// @brief Prints the matrix
ostream& operator<<(ostream& os, const Matrix& R) {
    // start in a new line
    //os << "\n";
    for (size_t i = 0; i < R._rows; i++) {
        for (size_t j = 0; j < R._cols; j++)
            os << R[i][j] << " ";
        os << endl;
    }
    return os;
}


// ============================================
// =============== Static methods ==============
/// @brief Creates a pair of matrices from a CSV file
std::pair<Matrix, Matrix> Matrix::readFromCSV(std::string const& filename){
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

        // Leer los valores de los tokens
        // Hasta num_inputs para la X
        // El ultimo valor para la Y
        // vectorX.push_back(1.0);
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            double value = std::stod(tokens[i]);
            //std::cout << "Value "<< value << "\n";
            if(i < num_inputs){
                vectorX.push_back(value);
            }else{
                // Leer ultimo valor en el vectorY
                vectorY.push_back(value);
            }
            
        }

        // Incrementar el contador de filas
        ++rowCount;
    }

    // Matrix X{rowCount, num_inputs +1, vectorX};
    Matrix X{rowCount, num_inputs, vectorX};
    Matrix Y{rowCount, 1, vectorY};
    
    return std::make_pair(X, Y);
     
}