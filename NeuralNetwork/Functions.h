#pragma once

namespace ActivationFunction
{
	template <class TComputer> static auto LogisticSigmoid(const TComputer& neuronComputer) -> std::valarray<std::decay_t<decltype(neuronComputer[0])>>
	{
		std::valarray<std::decay_t<decltype(neuronComputer[0])>> result(neuronComputer.size());
#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(result.size()); i++)
			result[static_cast<size_t>(i)] = 1 / (1 + exp(-neuronComputer[static_cast<size_t>(i)]));
		return std::move(result);
	}
	template <class T> static T LogisticSigmoidDifferentiated(T y) { return y * (1 - y); }
	template <class TComputer> static auto SoftMax(const TComputer& neuronComputer) -> std::valarray<std::decay_t<decltype(neuronComputer[0])>>
	{
		std::valarray<std::decay_t<decltype(neuronComputer[0])>> result(neuronComputer.size());
#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(result.size()); i++)
			result[static_cast<size_t>(i)] = neuronComputer[static_cast<size_t>(i)];
		std::decay_t<decltype(neuronComputer[0])> max = -std::numeric_limits<std::decay_t<decltype(neuronComputer[0])>>::infinity();
		for (int i = 0; i < static_cast<int>(result.size()); i++)
			max = (std::max)(result[static_cast<size_t>(i)], max);
		std::decay_t<decltype(neuronComputer[0])> sum = 0;
		for (int i = 0; i < static_cast<int>(result.size()); i++)
			sum += result[static_cast<size_t>(i)] = exp(result[static_cast<size_t>(i)] - max);
#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(result.size()); i++)
			result[static_cast<size_t>(i)] /= sum;
		return std::move(result);
	}
};

namespace CostFunction
{
	template <class T> static auto BiClassCrossEntropy(const T& source, const T& target) -> std::decay_t<decltype(source[0])>
	{
		std::decay_t<decltype(source[0])> sum = 0;
		auto eps = std::decay_t<decltype(source[0])>(1e-10);
		for (size_t i = 0; i < source.size(); i++)
			sum -= target[i] * log(source[i] + eps) + (1 - target[i]) * log(1 - source[i] + eps);
		return sum;
	}
	template <class T> static auto MultiClassCrossEntropy(const T& source, const T& target) -> std::decay_t<decltype(source[0])>
	{
		std::decay_t<decltype(source[0])> sum = 0;
		auto eps = std::decay_t<decltype(source[0])>(1e-10);
		for (size_t i = 0; i < source.size(); i++)
			sum -= target[i] * log(source[i] + eps);
		return sum;
	}
};