#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <vector>
#include <iostream>

using namespace std;

class Matrix {
    size_t _rows;
    size_t _cols;
    vector<vector<double>> _matrix;

    public:
    // Constructors
    Matrix();
    Matrix(size_t, size_t);
    Matrix(size_t, size_t, vector<vector<double>>);
    Matrix(size_t, size_t, vector<double>);
    Matrix(const Matrix&);

    // Methods
    size_t rows() const;
    size_t cols() const;
    vector<vector<double>> matrix() const;

    // Access operator
    vector<double>& operator[](size_t);
    const vector<double>& operator[](size_t) const;

    // Modifiers
    Matrix& apply(double (*f)(double));
    Matrix& apply(vector<double> (*f)(vector<double>));

    // Operators
    Matrix operator+(const Matrix&) const;
    Matrix operator-(const Matrix&) const;
    Matrix operator*(const Matrix&) const;
    Matrix operator*(double) const;
    // Matrix operator!=(const Matrix&) const;
    // Matrix operator==(const Matrix&) const;
    
    // Operations
    size_t sumcol(const size_t) const;
    Matrix transpose() const;

    // Friend functions
    friend Matrix operator*(double, const Matrix&);
    friend ostream& operator<<(ostream&, const Matrix&);
    
};

#endif