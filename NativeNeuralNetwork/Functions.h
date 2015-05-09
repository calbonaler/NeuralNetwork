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

	static const ActivationFunction* Sigmoid()
	{
		static ActivationFunction sigmoid([](const Indexer& x, VectorType& res)
		{
#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(res.size()); i++)
				res[static_cast<unsigned int>(i)] = 1 / (1 + exp(-x(static_cast<unsigned int>(i))));
		}, [](ValueType y) { return y * (1 - y); });
		return &sigmoid;
	}
	static void Identity(const Indexer& input, VectorType& result)
	{
#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(result.size()); i++)
			result[static_cast<unsigned int>(i)] = input(static_cast<unsigned int>(i));
	}
	static void SoftMax(const Indexer& input, VectorType& result)
	{
		ValueType max = -std::numeric_limits<ValueType>::infinity();
		for (unsigned int i = 0; i < result.size(); i++)
			max = std::max(result[i] = input(i), max);
		ValueType sum = 0;
		for (unsigned int i = 0; i < result.size(); i++)
			sum += result[i] = exp(result[i] - max);
#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(result.size()); i++)
			result[static_cast<unsigned int>(i)] /= sum;
	}

private:
	ActivationFunction(const NormalForm& normal, const DifferentiatedForm& differentiated) : Normal(normal), Differentiated(differentiated) { }
};

class ErrorFunction
{
public:
	static ValueType BiClassCrossEntropy(const VectorType& source, const VectorType& target)
	{
		ValueType sum = 0;
		for (unsigned int i = 0; i < source.size(); i++)
			sum -= target[i] * log(source[i] + static_cast<ValueType>(1e-10)) + (1 - target[i]) * log(1 - source[i] + static_cast<ValueType>(1e-10));
		return sum;
	}
	static ValueType MultiClassCrossEntropy(const VectorType& source, const VectorType& target)
	{
		ValueType sum = 0;
		for (unsigned int i = 0; i < source.size(); i++)
			sum -= target[i] * log(source[i] + static_cast<ValueType>(1e-10));
		return sum;
	}
	static ValueType LeastSquaresMethod(const VectorType& source, const VectorType& target)
	{
		ValueType sum = 0;
		for (unsigned int i = 0; i < source.size(); i++)
		{
			ValueType x = source[i] - target[i];
			sum += x * x;
		}
		return sum / 2;
	}
};