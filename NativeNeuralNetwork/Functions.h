#pragma once

#include "Utility.h"

class ActivationFunction : private Noncopyable
{
public:
	typedef std::function<void(const Indexer&, double*, unsigned int)> NormalForm;
	typedef std::function<double(double)> DifferentiatedForm;

	const NormalForm Normal;
	const DifferentiatedForm Differentiated;

	static const ActivationFunction* Sigmoid();
	static void Identity(const Indexer& input, double* result, unsigned int length);
	static void SoftMax(const Indexer& input, double* result, unsigned int length);

private:
	ActivationFunction(const NormalForm& normal, const DifferentiatedForm& differentiated) : Normal(normal), Differentiated(differentiated) { }
};

namespace ErrorFunction
{
	double BiClassCrossEntropy(const double* source, const double* target, unsigned int length);
	double MultiClassCrossEntropy(const double* source, const double* target, unsigned int length);
	double LeastSquaresMethod(const double* source, const double* target, unsigned int length);
}