#pragma once

#include "Utility.h"

struct ActivationFunction
{
	typedef void(*NormalForm)(const concurrency::array_view<const ValueType, 2>& weight, bool transposed, const concurrency::array_view<const ValueType>& bias, const concurrency::array<ValueType>& input, concurrency::array<ValueType>& result);

	static void LogisticSigmoid(const concurrency::array_view<const ValueType, 2>& weight, bool transposed, const concurrency::array_view<const ValueType>& bias, const concurrency::array<ValueType>& input, concurrency::array<ValueType>& result)
	{
		concurrency::parallel_for_each(result.extent, [=, &input, &result](concurrency::index<1> i) restrict(amp) { result[i] = 1 / (1 + concurrency::fast_math::exp(-ComputeNeuron(weight, transposed, bias, input, i))); });
	}
	static ValueType LogisticSigmoidDifferentiated(ValueType y) restrict (cpu, amp) { return y * (1 - y); }
	static void Tanh(const concurrency::array_view<const ValueType, 2>& weight, bool transposed, const concurrency::array_view<const ValueType>& bias, const concurrency::array<ValueType>& input, concurrency::array<ValueType>& result)
	{
		concurrency::parallel_for_each(result.extent, [=, &input, &result](concurrency::index<1> i) restrict(amp) { result[i] = concurrency::fast_math::tanh(ComputeNeuron(weight, transposed, bias, input, i)); });
	}
	static ValueType TanhDifferentiated(ValueType y) restrict(cpu, amp) { return 1 - y * y; }
	static void RectifiedLinear(const concurrency::array_view<const ValueType, 2>& weight, bool transposed, const concurrency::array_view<const ValueType>& bias, const concurrency::array<ValueType>& input, concurrency::array<ValueType>& result)
	{
		concurrency::parallel_for_each(result.extent, [=, &input, &result](concurrency::index<1> i) restrict(amp)
		{
			ValueType xs = ComputeNeuron(weight, transposed, bias, input, i);
			result[i] = xs > 0 ? xs : 0;
		});
	}
	static ValueType RectifiedLinearDifferentiated(ValueType y) restrict(cpu, amp) { return static_cast<ValueType>(y > 0 ? 1 : 0); }
	static void SoftPlus(const concurrency::array_view<const ValueType, 2>& weight, bool transposed, const concurrency::array_view<const ValueType>& bias, const concurrency::array<ValueType>& input, concurrency::array<ValueType>& result)
	{
		concurrency::parallel_for_each(result.extent, [=, &input, &result](concurrency::index<1> i) restrict(amp)
		{
			ValueType xs = ComputeNeuron(weight, transposed, bias, input, i);
			result[i] = xs > 0 ? xs + concurrency::fast_math::log(1 + concurrency::fast_math::exp(-xs)) : concurrency::fast_math::log(1 + concurrency::fast_math::exp(xs));
		});
	}
	static ValueType SoftPlusDifferentiated(ValueType y) restrict(cpu, amp) { return 1 - concurrency::fast_math::exp(-y); }
	static void Identity(const concurrency::array_view<const ValueType, 2>& weight, bool transposed, const concurrency::array_view<const ValueType>& bias, const concurrency::array<ValueType>& input, concurrency::array<ValueType>& result)
	{
		concurrency::parallel_for_each(result.extent, [=, &input, &result](concurrency::index<1> i) restrict(amp) { result[i] = ComputeNeuron(weight, transposed, bias, input, i); });
	}
	static ValueType IdentityDifferentiated(ValueType) restrict(cpu, amp) { return 1; }
	static void SoftMax(const concurrency::array_view<const ValueType, 2>& weight, bool transposed, const concurrency::array_view<const ValueType>& bias, const concurrency::array<ValueType>& input, concurrency::array<ValueType>& result)
	{
		concurrency::parallel_for_each(result.extent, [=, &input, &result](concurrency::index<1> i) restrict(amp) { result[i] = ComputeNeuron(weight, transposed, bias, input, i); });
		concurrency::array_view<ValueType> resultView = result;
		ValueType max = -std::numeric_limits<ValueType>::infinity();
		for (int i = 0; i < resultView.extent[0]; i++)
			max = (std::max)(resultView[i], max);
		ValueType sum = 0;
		for (int i = 0; i < resultView.extent[0]; i++)
			sum += resultView[i] = concurrency::fast_math::exp(resultView[i] - max);
		resultView.synchronize(concurrency::access_type_read_write);
		concurrency::parallel_for_each(result.extent, [=, &result](concurrency::index<1> i) restrict(amp) { result[i] /= sum; });
	}

private:
	static ValueType ComputeNeuron(const concurrency::array_view<const ValueType, 2>& weight, bool transposed, const concurrency::array_view<const ValueType>& bias, const concurrency::array<ValueType>& input, concurrency::index<1> index) restrict(amp)
	{
		ValueType ret = bias[index];
		for (int k = 0; k < input.extent[0]; k++)
			ret += input[k] * (transposed ? weight[k][index[0]] : weight[index[0]][k]);
		return ret;
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