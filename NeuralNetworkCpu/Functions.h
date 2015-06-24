#pragma once

#include "Utility.h"

struct ActivationFunction
{
	typedef void(*NormalForm)(ValueType** weight, bool transposed, const VectorType& bias, const VectorType& input, VectorType& result);
	typedef ValueType(*DifferentiatedForm)(ValueType y);

	static void LogisticSigmoid(ValueType** weight, bool transposed, const VectorType& bias, const VectorType& input, VectorType& result)
	{
#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(result.size()); i++)
			result[static_cast<size_t>(i)] = 1 / (1 + exp(-ComputeNeuron(weight, transposed, bias, input, i)));
	}
	static ValueType LogisticSigmoidDifferentiated(ValueType y) { return y * (1 - y); }
	static void Tanh(ValueType** weight, bool transposed, const VectorType& bias, const VectorType& input, VectorType& result)
	{
#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(result.size()); i++)
			result[static_cast<size_t>(i)] = tanh(ComputeNeuron(weight, transposed, bias, input, i));
	}
	static ValueType TanhDifferentiated(ValueType y) { return 1 - y * y; }
	static void RectifiedLinear(ValueType** weight, bool transposed, const VectorType& bias, const VectorType& input, VectorType& result)
	{
#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(result.size()); i++)
		{
			ValueType xs = ComputeNeuron(weight, transposed, bias, input, i);
			result[static_cast<size_t>(i)] = xs > 0 ? xs : 0;
		}
	}
	static ValueType RectifiedLinearDifferentiated(ValueType y) { return static_cast<ValueType>(y > 0 ? 1 : 0); }
	static void SoftPlus(ValueType** weight, bool transposed, const VectorType& bias, const VectorType& input, VectorType& result)
	{
#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(result.size()); i++)
		{
			ValueType xs = ComputeNeuron(weight, transposed, bias, input, i);
			result[static_cast<size_t>(i)] = xs > 0 ? xs + log(1 + exp(-xs)) : log(1 + exp(xs));
		}
	}
	static ValueType SoftPlusDifferentiated(ValueType y) { return 1 - exp(-y); }
	static void Identity(ValueType** weight, bool transposed, const VectorType& bias, const VectorType& input, VectorType& result)
	{
#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(result.size()); i++)
			result[static_cast<size_t>(i)] = ComputeNeuron(weight, transposed, bias, input, i);
	}
	static ValueType IdentityDifferentiated(ValueType) { return 1; }
	static void SoftMax(ValueType** weight, bool transposed, const VectorType& bias, const VectorType& input, VectorType& result)
	{
#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(result.size()); i++)
			result[static_cast<size_t>(i)] = ComputeNeuron(weight, transposed, bias, input, i);
		ValueType max = -std::numeric_limits<ValueType>::infinity();
		for (int i = 0; i < static_cast<int>(result.size()); i++)
			max = (std::max)(result[static_cast<size_t>(i)], max);
		ValueType sum = 0;
		for (int i = 0; i < static_cast<int>(result.size()); i++)
			sum += result[static_cast<size_t>(i)] = exp(result[static_cast<size_t>(i)] - max);
#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(result.size()); i++)
			result[static_cast<size_t>(i)] /= sum;
	}

private:
	static ValueType ComputeNeuron(ValueType** weight, bool transposed, const VectorType& bias, const VectorType& input, int index)
	{
		ValueType ret = bias[static_cast<size_t>(index)];
		for (size_t k = 0; k < input.size(); k++)
			ret += input[k] * (transposed ? weight[k][static_cast<size_t>(index)] : weight[static_cast<size_t>(index)][k]);
		return ret;
	}
};

struct CostFunction
{
	static ValueType BiClassCrossEntropy(const VectorType& source, const VectorType& target)
	{
		ValueType sum = 0;
		for (size_t i = 0; i < source.size(); i++)
			sum -= target[i] * log(source[i] + static_cast<ValueType>(1e-10)) + (1 - target[i]) * log(1 - source[i] + static_cast<ValueType>(1e-10));
		return sum;
	}
	static ValueType MultiClassCrossEntropy(const VectorType& source, const VectorType& target)
	{
		ValueType sum = 0;
		for (size_t i = 0; i < source.size(); i++)
			sum -= target[i] * log(source[i] + static_cast<ValueType>(1e-10));
		return sum;
	}
	static ValueType LeastSquaresMethod(const VectorType& source, const VectorType& target)
	{
		ValueType sum = 0;
		for (size_t i = 0; i < source.size(); i++)
		{
			ValueType x = source[i] - target[i];
			sum += x * x;
		}
		return sum / 2;
	}
};