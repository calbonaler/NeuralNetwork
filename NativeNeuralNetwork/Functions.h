#pragma once

#include "Utility.h"

class Indexer : private boost::noncopyable
{
public:
	Indexer(ValueType** weight, const VectorType& bias, const VectorType& input, bool transpose = false) :
#ifdef NEURALNETWORK_USE_GPU
		_weight(static_cast<int>(bias.size()), static_cast<int>(input.size()), weight[0]),
		_bias(static_cast<int>(bias.size()), &const_cast<VectorType&>(bias)[0]),
		_input(static_cast<int>(input.size()), &const_cast<VectorType&>(input)[0]),
#else
		_weight(weight),
		_bias(bias),
		_input(input),
#endif
		_transpose(transpose)
	{
	}
	
	ValueType operator()(unsigned int index) const
#ifdef NEURALNETWORK_USE_GPU
		restrict(cpu, amp)
#endif
	{
		ValueType ret = 0;
#ifdef NEURALNETWORK_USE_GPU
		for (int k = 0; k < _weight.extent[1]; k++)
			ret += _input[k] * (_transpose ? _weight[k][static_cast<int>(index)] : _weight[static_cast<int>(index)][k]);
		return ret + _bias[static_cast<int>(index)];
#else
		for (unsigned int k = 0; k < _input.size(); k++)
			ret += _input[k] * (_transpose ? _weight[k][index] : _weight[index][k]);
		return ret + _bias[index];
#endif
	}

private:
#ifdef NEURALNETWORK_USE_GPU
	concurrency::array_view<ValueType, 2> _weight;
	concurrency::array_view<ValueType, 1> _bias;
	concurrency::array_view<ValueType, 1> _input;
#else
	ValueType** _weight;
	const VectorType& _bias;
	const VectorType& _input;
#endif
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
		return sum / source.size();
	}
	static ValueType MultiClassCrossEntropy(const VectorType& source, const VectorType& target)
	{
		ValueType sum = 0;
		for (unsigned int i = 0; i < source.size(); i++)
			sum -= target[i] * log(source[i] + static_cast<ValueType>(1e-10));
		return sum / source.size();
	}
	static ValueType LeastSquaresMethod(const VectorType& source, const VectorType& target)
	{
		ValueType sum = 0;
		for (unsigned int i = 0; i < source.size(); i++)
		{
			ValueType x = source[i] - target[i];
			sum += x * x;
		}
		return sum / 2 / source.size();
	}
};