#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <vector>
#include <iostream>

class Matrix {
private:
    std::size_t _rows;
    std::size_t _cols;
    std::vector<std::vector<double>> _matrix;

public:
    // Constructors
    Matrix();
    Matrix(std::size_t, std::size_t);
    Matrix(std::size_t, std::size_t, std::vector<std::vector<double>>);
    Matrix(std::size_t, std::size_t, std::vector<double>);
    Matrix(const Matrix&);

    // Methods
    std::size_t rows() const;
    std::size_t cols() const;
    std::size_t size() const;
    std::vector<std::vector<double>> matrix() const;
    std::vector<double> getCol(std::size_t) const;
    std::vector<double> getRow(std::size_t) const;
    std::pair<Matrix, Matrix> divide(const double ratio, bool shuffle = true, unsigned seed = 0) const;
    std::vector<int> kfold(const int k, bool shuffle = true, unsigned seed = 0) const;
    std::pair<Matrix, Matrix> getFold(const std::vector<int> &folds, const int k, const int i) const;

    // Access operator
    std::vector<double>& operator[](std::size_t);
    const std::vector<double>& operator[](std::size_t) const;

    // Modifiers
    Matrix& apply(double (*f)(double));
    Matrix& apply(std::vector<double> (*f)(std::vector<double>));

    // Operators
    Matrix operator+(const Matrix&) const;
    Matrix operator-(const Matrix&) const;
    Matrix operator*(const Matrix&) const;
    Matrix operator*(double) const;
    Matrix operator!=(const Matrix&) const;
    Matrix& operator+=(const Matrix& R);
    // Matrix operator==(const Matrix&) const;
    
    // Operations
    double sumcol(const std::size_t) const;
    Matrix transpose() const;
    Matrix mult(std::size_t, double) const;


    // Friend functions
    friend Matrix operator*(double, const Matrix&);
    friend std::ostream& operator<<(std::ostream&, const Matrix&);
    
};

#endif