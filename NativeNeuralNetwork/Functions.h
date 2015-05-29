#pragma once

#include "Utility.h"

class Indexer
{
public:
	Indexer(ValueType** weight, const VectorType& bias, const VectorType& input, bool transpose = false) :
		_weight(static_cast<int>(bias.size()), static_cast<int>(input.size()), weight[0]),
		_bias(static_cast<int>(bias.size()), &bias[0]),
		_input(static_cast<int>(input.size()), &input[0]),
		_transpose(transpose)
	{
	}

	Indexer(const Indexer& right) : _weight(right._weight), _bias(right._bias), _input(right._input), _transpose(right._transpose) { }
	
	Indexer& operator=(const Indexer& right)
	{
		_weight = right._weight;
		_bias = right._bias;
		_input = right._input;
		_transpose = right._transpose;
		return *this;
	}

	ValueType operator()(concurrency::index<1> index) const restrict(cpu, amp)
	{
		ValueType ret = _bias[index];
		for (int k = 0; k < _weight.extent[1]; k++)
			ret += _input[k] * (_transpose ? _weight[k][index[0]] : _weight[index[0]][k]);
		return ret;
	}

private:
	concurrency::array_view<const ValueType, 2> _weight;
	concurrency::array_view<const ValueType, 1> _bias;
	concurrency::array_view<const ValueType, 1> _input;
	bool _transpose;
};

class ActivationFunction : private boost::noncopyable
{
public:
	typedef std::function<void(const Indexer&, VectorType&)> NormalForm;
	typedef std::function<ValueType(ValueType)> DifferentiatedForm;

	const NormalForm Normal;
	const DifferentiatedForm Differentiated;

	static const ActivationFunction* LogisticSigmoid() { return &_logisticSigmoid; }
	static const ActivationFunction* Tanh() { return &_tanh; }
	static const ActivationFunction* RectifiedLinear() { return &_rectifiedLinear; }
	static const ActivationFunction* SoftPlus() { return &_softplus; }
	static const ActivationFunction* Identity() { return &_identity; }
	static void SoftMax(const Indexer& input, VectorType& result);

private:
	ActivationFunction(const NormalForm& normal, const DifferentiatedForm& differentiated) : Normal(normal), Differentiated(differentiated) { }
	~ActivationFunction() { }
	static ActivationFunction _logisticSigmoid;
	static ActivationFunction _tanh;
	static ActivationFunction _rectifiedLinear;
	static ActivationFunction _softplus;
	static ActivationFunction _identity;
};

class CostFunction
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