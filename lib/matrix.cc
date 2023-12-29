#include "matrix.h"
#include <iostream>
#include <vector>
#include <numeric>

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