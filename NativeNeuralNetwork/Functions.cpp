#include <limits>
#include <cmath>
#include "Functions.h"

const ActivationFunction* ActivationFunction::Sigmoid()
{
	static ActivationFunction sigmoid([](const Indexer& x, std::vector<double>& res)
	{
		int length = (int)res.size();
#pragma omp parallel for
		for (int i = 0; i < length; i++)
			res[(unsigned)i] = 1 / (1 + exp(-x(i)));
	}, [](double y) { return y * (1 - y); });
	return &sigmoid;
}

void ActivationFunction::Identity(const Indexer& input, std::vector<double>& result)
{
	int length = (int)result.size();
#pragma omp parallel for
	for (int i = 0; i < length; i++)
		result[(unsigned)i] = input(i);
}

void ActivationFunction::SoftMax(const Indexer& input, std::vector<double>& result)
{
	int length = (int)result.size();
	double max = -std::numeric_limits<double>::infinity();
	for (int i = 0; i < length; i++)
		max = fmax(result[(unsigned)i] = input(i), max);
	double sum = 0;
	for (int i = 0; i < length; i++)
		sum += result[(unsigned)i] = exp(result[(unsigned)i] - max);
#pragma omp parallel for
	for (int i = 0; i < length; i++)
		result[(unsigned)i] /= sum;
}

double ErrorFunction::BiClassCrossEntropy(const std::vector<double>& source, const std::vector<double>& target)
{
	double sum = 0;
	for (unsigned int i = 0; i < source.size(); i++)
		sum -= target[i] * log(source[i] + 1e-10) + (1 - target[i]) * log(1 - source[i] + 1e-10);
	return sum;
}

double ErrorFunction::MultiClassCrossEntropy(const std::vector<double>& source, const std::vector<double>& target)
{
	double sum = 0;
	for (unsigned int i = 0; i < source.size(); i++)
		sum -= target[i] * log(source[i] + 1e-10);
	return sum;
}

double ErrorFunction::LeastSquaresMethod(const std::vector<double>& source, const std::vector<double>& target)
{
	double sum = 0;
	for (unsigned int i = 0; i < source.size(); i++)
	{
		double x = source[i] - target[i];
		sum += x * x;
	}
	return sum / 2;
}
