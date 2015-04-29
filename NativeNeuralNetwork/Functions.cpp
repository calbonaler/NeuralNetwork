#include <limits>
#include <cmath>
#include "Functions.h"

const ActivationFunction* ActivationFunction::Sigmoid()
{
	static ActivationFunction sigmoid([](const Indexer& x, double* res, int len)
	{
#pragma omp parallel for
		for (int i = 0; i < len; i++)
			res[i] = 1 / (1 + exp(-x(i)));
	}, [](double y) { return y * (1 - y); });
	return &sigmoid;
}

void ActivationFunction::Identity(const Indexer& input, double* result, int length)
{
#pragma omp parallel for
	for (int i = 0; i < length; i++)
		result[i] = input(i);
}

void ActivationFunction::SoftMax(const Indexer& input, double* result, int length)
{
	double max = -std::numeric_limits<double>::infinity();
	for (int i = 0; i < length; i++)
		max = fmax(result[i] = input(i), max);
	double sum = 0;
	for (int i = 0; i < length; i++)
		sum += result[i] = exp(result[i] - max);
#pragma omp parallel for
	for (int i = 0; i < length; i++)
		result[i] /= sum;
}

double ErrorFunction::BiClassCrossEntropy(const double* source, const double* target, int length)
{
	double sum = 0;
	for (int i = 0; i < length; i++)
		sum -= target[i] * log(source[i] + 1e-10) + (1 - target[i]) * log(1 - source[i] + 1e-10);
	return sum;
}

double ErrorFunction::MultiClassCrossEntropy(const double* source, const double* target, int length)
{
	double sum = 0;
	for (int i = 0; i < length; i++)
		sum -= target[i] * log(source[i] + 1e-10);
	return sum;
}

double ErrorFunction::LeastSquaresMethod(const double* source, const double* target, int length)
{
	double sum = 0;
	for (int i = 0; i < length; i++)
	{
		double x = source[i] - target[i];
		sum += x * x;
	}
	return sum / 2;
}
