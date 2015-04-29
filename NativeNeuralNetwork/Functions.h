#pragma once

#include "Utility.h"

class ActivationFunction
{
public:
	typedef std::function<void(const Indexer&, double*, int)> NormalForm;
	typedef std::function<double(double)> DifferentiatedForm;

	FORCE_UNCOPYABLE(ActivationFunction);

	const NormalForm Normal;
	const DifferentiatedForm Differentiated;

	static const ActivationFunction* Sigmoid();
	static void Identity(const Indexer& input, double* result, int length);
	static void SoftMax(const Indexer& input, double* result, int length);

private:
	ActivationFunction(const NormalForm& normal, const DifferentiatedForm& differentiated) : Normal(normal), Differentiated(differentiated) { }
};

namespace ErrorFunction
{
	double BiClassCrossEntropy(const double* source, const double* target, int length);
	double MultiClassCrossEntropy(const double* source, const double* target, int length);
	double LeastSquaresMethod(const double* source, const double* target, int length);
}