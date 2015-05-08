#pragma once

#include "Utility.h"

typedef std::function<ValueType(unsigned int)> Indexer;

class ActivationFunction : private boost::noncopyable
{
public:
	typedef std::function<void(const Indexer&, VectorType&)> NormalForm;
	typedef std::function<ValueType(ValueType)> DifferentiatedForm;

	const NormalForm Normal;
	const DifferentiatedForm Differentiated;

	static const ActivationFunction* Sigmoid();
	static void Identity(const Indexer& input, VectorType& result);
	static void SoftMax(const Indexer& input, VectorType& result);

private:
	ActivationFunction(const NormalForm& normal, const DifferentiatedForm& differentiated) : Normal(normal), Differentiated(differentiated) { }
};

namespace ErrorFunction
{
	ValueType BiClassCrossEntropy(const VectorType& source, const VectorType& target);
	ValueType MultiClassCrossEntropy(const VectorType& source, const VectorType& target);
	ValueType LeastSquaresMethod(const VectorType& source, const VectorType& target);
}