#pragma once

#include "Utility.h"

class ActivationFunction
{
public:
	typedef std::function<void(const Indexer&, std::vector<double>&)> NormalForm;
	typedef std::function<double(double)> DifferentiatedForm;

	FORCE_UNCOPYABLE(ActivationFunction);

	const NormalForm Normal;
	const DifferentiatedForm Differentiated;

	static const ActivationFunction* Sigmoid();
	static void Identity(const Indexer& input, std::vector<double>& result);
	static void SoftMax(const Indexer& input, std::vector<double>& result);

private:
	ActivationFunction(const NormalForm& normal, const DifferentiatedForm& differentiated) : Normal(normal), Differentiated(differentiated) { }
};

namespace ErrorFunction
{
	double BiClassCrossEntropy(const std::vector<double>& source, const std::vector<double>& target);
	double MultiClassCrossEntropy(const std::vector<double>& source, const std::vector<double>& target);
	double LeastSquaresMethod(const std::vector<double>& source, const std::vector<double>& target);
}