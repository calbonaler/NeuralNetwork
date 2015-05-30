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

struct ActivationFunction
{
	typedef void(*NormalForm)(const Indexer&, concurrency::array_view<ValueType, 1>&);

	static void LogisticSigmoid(const Indexer& x, concurrency::array_view<ValueType, 1>& result)
	{
		result.discard_data();
		concurrency::parallel_for_each(result.extent, [=](concurrency::index<1> i) restrict(amp) { result[i] = 1 / (1 + concurrency::fast_math::exp(-x(i))); });
	}
	static ValueType LogisticSigmoidDifferentiated(ValueType y) restrict (cpu, amp) { return y * (1 - y); }
	static void Tanh(const Indexer& x, concurrency::array_view<ValueType, 1>& result)
	{
		result.discard_data();
		concurrency::parallel_for_each(result.extent, [=](concurrency::index<1> i) restrict(amp) { result[i] = concurrency::fast_math::tanh(x(i)); });
	}
	static ValueType TanhDifferentiated(ValueType y) restrict(cpu, amp) { return 1 - y * y; }
	static void RectifiedLinear(const Indexer& x, concurrency::array_view<ValueType, 1>& result)
	{
		result.discard_data();
		concurrency::parallel_for_each(result.extent, [=](concurrency::index<1> i) restrict(amp)
		{
			ValueType xs = x(i);
			result[i] = xs > 0 ? xs : 0;
		});
	}
	static ValueType RectifiedLinearDifferentiated(ValueType y) restrict(cpu, amp) { return static_cast<ValueType>(y > 0 ? 1 : 0); }
	static void SoftPlus(const Indexer& x, concurrency::array_view<ValueType, 1>& result)
	{
		result.discard_data();
		concurrency::parallel_for_each(result.extent, [=](concurrency::index<1> i) restrict(amp)
		{
			ValueType xs = x(i);
			result[i] = xs > 0 ? xs + concurrency::fast_math::log(1 + concurrency::fast_math::exp(-xs)) : concurrency::fast_math::log(1 + concurrency::fast_math::exp(xs));
		});
	}
	static ValueType SoftPlusDifferentiated(ValueType y) restrict(cpu, amp) { return 1 - concurrency::fast_math::exp(-y); }
	static void Identity(const Indexer& x, concurrency::array_view<ValueType, 1>& result)
	{
		result.discard_data();
		concurrency::parallel_for_each(result.extent, [=](concurrency::index<1> i) restrict(amp) { result[i] = x(i); });
	}
	static ValueType IdentityDifferentiated(ValueType) restrict(cpu, amp) { return 1; }
	static void SoftMax(const Indexer& x, concurrency::array_view<ValueType, 1>& result)
	{
		result.discard_data();
		ValueType max = -std::numeric_limits<ValueType>::infinity();
		for (int i = 0; i < result.extent[0]; i++)
			max = (std::max)(result[i] = x(concurrency::index<1>(i)), max);
		ValueType sum = 0;
		for (int i = 0; i < result.extent[0]; i++)
			sum += result[i] = concurrency::fast_math::exp(result[i] - max);
		concurrency::parallel_for_each(result.extent, [=](concurrency::index<1> i) restrict(amp) { result[i] /= sum; });
	}
};

struct CostFunction
{
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