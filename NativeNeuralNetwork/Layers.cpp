#include "Layers.h"

Layer::Layer(unsigned int nIn, unsigned int nOut, const ActivationFunction::NormalForm& activation) : nIn(nIn), nOut(nOut), Weight(new ValueType*[nOut]), Bias(0.0, nOut), Activation(activation)
{
	assert(nIn * nOut > 0);
	Weight[0] = new ValueType[nIn * nOut]();
	for (unsigned int i = 1; i < nOut; i++)
		Weight[i] = Weight[i - 1] + nOut;
}

Layer::~Layer()
{
	delete[] Weight[0];
	delete[] Weight;
}

VectorType Layer::Compute(const VectorType& input) const
{
	VectorType output = VectorType(nOut);
	Activation([&](unsigned int j)
	{
		ValueType ret = 0;
		for (unsigned int k = 0; k < nIn; k++)
			ret += input[k] * Weight[j][k];
		return ret + Bias[j];
	}, output);
	return std::move(output);
}

VectorType Layer::Learn(const VectorType& input, const VectorType& output, const Indexer& upperInfo, ValueType learningRate)
{
	VectorType lowerInfo = VectorType(0.0, nIn);
	for (unsigned int i = 0; i < nOut; i++)
	{
		ValueType deltaI = GetDelta(output[i], upperInfo(i));
		for (unsigned int j = 0; j < nIn; j++)
		{
			lowerInfo[j] += Weight[i][j] * deltaI;
			Weight[i][j] -= learningRate * (deltaI * input[j]);
		}
		Bias[i] -= learningRate * deltaI;
	}
	return std::move(lowerInfo);
}

HiddenLayer::HiddenLayer(unsigned int nIn, unsigned int nOut, const ActivationFunction* activation, HiddenLayerCollection* hiddenLayers) : Layer(nIn, nOut, activation->Normal), differentiatedActivation(activation->Differentiated), visibleBias(0.0, nIn), hiddenLayers(hiddenLayers)
{
	assert(activation);
	assert(hiddenLayers);
	std::uniform_real_distribution<ValueType> dist(0, nextafter(static_cast<ValueType>(1.0), std::numeric_limits<ValueType>::max()));
	for (unsigned int j = 0; j < nOut; j++)
	{
		for (unsigned int i = 0; i < nIn; i++)
		{
			Weight[j][i] = (2 * dist(hiddenLayers->RandomNumberGenerator) - 1) * sqrt(static_cast<ValueType>(6.0) / (nIn + nOut));
			if (activation == ActivationFunction::Sigmoid())
				Weight[j][i] *= 4;
		}
	}
}

ValueType HiddenLayer::Train(const DataSet& dataset, ValueType learningRate, Floating noise)
{
	std::uniform_real_distribution<Floating> dist(0, 1);
	VectorType corrupted = VectorType(nIn);
	VectorType latent = VectorType(nOut);
	VectorType reconstructed = VectorType(nIn);
	VectorType delta = VectorType(nOut);
	ValueType cost = 0;
	for (unsigned int n = 0; n < dataset.Count(); n++)
	{
		auto image = hiddenLayers->Compute(dataset.Images()[n], this);
		for (unsigned int i = 0; i < nIn; i++)
			corrupted[i] = dist(hiddenLayers->RandomNumberGenerator) < noise ? 0 : image.get()[i];
		Activation([&](unsigned int j)
		{
			ValueType ret = 0;
			for (unsigned int i = 0; i < nIn; i++)
				ret += corrupted[i] * Weight[j][i];
			return ret + Bias[j];
		}, latent);
		Activation([&](unsigned int j)
		{
			ValueType ret = 0;
			for (unsigned int i = 0; i < nOut; i++)
				ret += latent[i] * Weight[i][j];
			return ret + visibleBias[j];
		}, reconstructed);
		cost += ErrorFunction::BiClassCrossEntropy(image.get(), reconstructed);
		for (unsigned int i = 0; i < nOut; i++)
		{
			delta[i] = 0;
			for (unsigned int j = 0; j < nIn; j++)
				delta[i] += (reconstructed[j] - image.get()[j]) * Weight[i][j];
			delta[i] *= differentiatedActivation(latent[i]);
			Bias[i] -= learningRate * delta[i];
		};
		for (unsigned int j = 0; j < nIn; j++)
		{
			for (unsigned int i = 0; i < nOut; i++)
				Weight[i][j] -= learningRate * ((reconstructed[j] - image.get()[j]) * latent[i] + delta[i] * corrupted[j]);
			visibleBias[j] -= learningRate * (reconstructed[j] - image.get()[j]);
		};
	}
	return cost / dataset.Count();
}

ValueType HiddenLayer::ComputeCost(const DataSet& dataset, Floating noise) const
{
	std::uniform_real_distribution<Floating> dist(0, 1);
	VectorType corrupted = VectorType(nIn);
	VectorType latent = VectorType(nOut);
	VectorType reconstructed = VectorType(nIn);
	ValueType cost = 0;
	for (unsigned int n = 0; n < dataset.Count(); n++)
	{
		auto image = hiddenLayers->Compute(dataset.Images()[n], this);
		for (unsigned int i = 0; i < nIn; i++)
			corrupted[i] = dist(hiddenLayers->RandomNumberGenerator) < noise ? 0 : image.get()[i];
		Activation([&](unsigned int j)
		{
			ValueType ret = 0;
			for (unsigned int i = 0; i < nIn; i++)
				ret += corrupted[i] * Weight[j][i];
			return ret + Bias[j];
		}, latent);
		Activation([&](unsigned int j)
		{
			ValueType ret = 0;
			for (unsigned int i = 0; i < nOut; i++)
				ret += latent[i] * Weight[i][j];
			return ret + visibleBias[j];
		}, reconstructed);
		cost += ErrorFunction::BiClassCrossEntropy(image.get(), reconstructed);
	}
	return cost / dataset.Count();
}

ReferableVector HiddenLayerCollection::Compute(const VectorType& input, const HiddenLayer* stopLayer) const
{
	ReferableVector result(input);
	for (unsigned int i = 0; i < items.size() && items[i].get() != stopLayer; i++)
		result = items[i]->Compute(result.get());
	return std::move(result);
}

void HiddenLayerCollection::Set(size_t index, unsigned int neurons)
{
	if (frozen)
		throw std::domain_error("固定されたコレクションの隠れ層を設定することはできません。");
	if (index > items.size())
		throw std::out_of_range("index");
	if (index == items.size())
	{
		items.push_back(std::unique_ptr<HiddenLayer>(new HiddenLayer(nextLayerInputUnits, neurons, ActivationFunction::Sigmoid(), this)));
		nextLayerInputUnits = neurons;
		return;
	}
	items[index] = std::unique_ptr<HiddenLayer>(new HiddenLayer(items[index]->nIn, neurons, ActivationFunction::Sigmoid(), this));
	if (index < items.size() - 1)
		items[index + 1] = std::unique_ptr<HiddenLayer>(new HiddenLayer(neurons, items[index + 1]->nOut, ActivationFunction::Sigmoid(), this));
}

unsigned int LogisticRegressionLayer::Predict(const VectorType& input) const
{
	auto computed = Compute(input);
	unsigned int maxIndex = 0;
	for (unsigned int i = 1; i < nOut; i++)
	{
		if (computed[i] > computed[maxIndex])
			maxIndex = i;
	}
	return maxIndex;
}
