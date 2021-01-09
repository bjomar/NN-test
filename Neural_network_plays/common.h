#pragma once
#include <vector>
#include <deque>
#include <cmath>
#include <tuple>
#include <iostream>

using WeightMatrix = std::vector<std::vector<double>>;
using Layer = std::vector<double>;

using ActivationFunctionPTR = double(*)(double);
using ErrorFunctionPTR = double(*)(double);

// vector multiplication
static Layer operator*(Layer const& a, Layer const& b) {
	if (a.size() != b.size())
		throw std::exception("size missmatch");

	Layer result(a.size());

	for (size_t i = 0; i < a.size(); i++)
	{
		result[i] = a[i] * b[i];
	}

	return result;
}

//scalar multiplication
static Layer operator* (double x, Layer const& l) {
	Layer result;
	for (auto d : l) {
		result.push_back(d * x);
	}
	return result;
}

// adds two weight matricies, storing them in a
// it assumes that the dimensions fit
static void operator += (WeightMatrix& a, WeightMatrix const& b) {
	
	for (size_t i = 0; i < a.size(); i++)
	{
		for (size_t j = 0; j < a[0].size(); j++)
		{
			a[i][j] = a[i][j] + b[i][j];
		}
	}
}

static void operator *= (WeightMatrix& a, double scalar) {
	for (size_t i = 0; i < a.size(); i++)
	{
		for (size_t j = 0; j < a[0].size(); j++)
		{
			a[i][j] *= scalar;
		}
	}
}

inline double MSE(double x) {
	return std::pow(x, 2);
}

inline double identity(double x) {
	return x;
}

inline double sigmoid(double x) {
	return 1. / (1. + exp(-x));
}

inline double tanH(double x) {
	return std::tanh(x);
}

inline double dtanh(double y) {
	return 1. - y * y;
}

/// <summary>
/// actual sigmoid derivative
/// </summary>
/// <param name="x">a double precision value</param>
/// <returns>sig(x)</returns>
inline double dsigmoid(double x) {
	return (sigmoid(x) * (1. - sigmoid(x)));
}

/// <summary>
/// sigmoid derivative, but with y beeing already processed by the sigmoid function
/// </summary>
/// <param name="y">the value of sigmoid x</param>
/// <returns>sig'(x)</returns>
inline double dsigmoid2(double y) {
	return y * (1. - y);
}