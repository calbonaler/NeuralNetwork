#include <cassert>
#include <memory>
#include "Layers.h"

Layer::Layer(int nIn, int nOut, const ActivationFunction::NormalForm& activation) : nIn(nIn), nOut(nOut), Weight(new double*[(unsigned)nOut]), Bias((unsigned)nOut, 0.0), Activation(activation)
{
	assert(nIn * nOut > 0);
	Weight[0] = new double[(unsigned)(nIn * nOut)]();
	for (int i = 1; i < nOut; i++)
		Weight[i] = Weight[i - 1] + nOut;
}

Layer::~Layer()
{
	delete[] Weight[0];
	delete[] Weight;
}

std::vector<double> Layer::Compute(const std::vector<double>& input) const
{
	std::vector<double> output = std::vector<double>((unsigned)nOut);
	Activation([&](int j)
	{
		double ret = 0;
		for (int k = 0; k < nIn; k++)
			ret += input[(unsigned)k] * Weight[j][k];
		return ret + Bias[(unsigned)j];
	}, output);
	return std::move(output);
}

std::vector<double> Layer::Learn(const std::vector<double>& input, const std::vector<double>& output, const Indexer& upperInfo, double learningRate)
{
	std::vector<double> lowerInfo = std::vector<double>((unsigned)nIn, 0.0);
	for (int i = 0; i < nOut; i++)
	{
		auto deltaI = GetDelta(output[(unsigned)i], upperInfo(i));
		for (int j = 0; j < nIn; j++)
		{
			lowerInfo[(unsigned)j] += Weight[i][j] * deltaI;
			Weight[i][j] -= learningRate * (deltaI * input[(unsigned)j]);
		}
		Bias[(unsigned)i] -= learningRate * deltaI;
	}
	return std::move(lowerInfo);
}

HiddenLayer::HiddenLayer(int nIn, int nOut, const ActivationFunction* activation, HiddenLayerCollection* hiddenLayers) : Layer(nIn, nOut, activation->Normal), differentiatedActivation(activation->Differentiated), visibleBias((unsigned)nIn, 0.0), hiddenLayers(hiddenLayers)
{
	assert(activation);
	assert(hiddenLayers);
	std::uniform_real_distribution<double> dist(0, nextafter(1.0, std::numeric_limits<double>::max()));
	for (int j = 0; j < nOut; j++)
	{
		for (int i = 0; i < nIn; i++)
		{
			Weight[j][i] = (2 * dist(hiddenLayers->RandomNumberGenerator) - 1) * sqrt(6.0 / (nIn + nOut));
			if (activation == ActivationFunction::Sigmoid())
				Weight[j][i] *= 4;
		}
	}
}

double HiddenLayer::Train(const DataSet& dataset, double learningRate, double noise)
{
	std::uniform_real_distribution<double> dist(0, 1);
	auto corrupted = std::vector<double>((unsigned)nIn);
	auto latent = std::vector<double>((unsigned)nOut);
	auto reconstructed = std::vector<double>((unsigned)nIn);
	auto delta = std::vector<double>((unsigned)nOut);
	double cost = 0;
	for (unsigned int n = 0; n < dataset.Count(); n++)
	{
		auto image = hiddenLayers->Compute(dataset.Images()[n], this);
		for (int i = 0; i < nIn; i++)
			corrupted[(unsigned)i] = dist(hiddenLayers->RandomNumberGenerator) < noise ? 0 : image.get()[(unsigned)i];
		Activation([&](int j)
		{
			double ret = 0;
			for (int i = 0; i < nIn; i++)
				ret += corrupted[(unsigned)i] * Weight[j][i];
			return ret + Bias[(unsigned)j];
		}, latent);
		Activation([&](int j)
		{
			double ret = 0;
			for (int i = 0; i < nOut; i++)
				ret += latent[(unsigned)i] * Weight[i][j];
			return ret + visibleBias[(unsigned)j];
		}, reconstructed);
		cost += ErrorFunction::BiClassCrossEntropy(image.get(), reconstructed);
		for (int i = 0; i < nOut; i++)
		{
			delta[(unsigned)i] = 0;
			for (int j = 0; j < nIn; j++)
				delta[(unsigned)i] += (reconstructed[(unsigned)j] - image.get()[(unsigned)j]) * Weight[i][j];
			delta[(unsigned)i] *= differentiatedActivation(latent[(unsigned)i]);
			Bias[(unsigned)i] -= learningRate * delta[(unsigned)i];
		};
		for (int j = 0; j < nIn; j++)
		{
			for (int i = 0; i < nOut; i++)
				Weight[i][j] -= learningRate * ((reconstructed[(unsigned)j] - image.get()[(unsigned)j]) * latent[(unsigned)i] + delta[(unsigned)i] * corrupted[(unsigned)j]);
			visibleBias[(unsigned)j] -= learningRate * (reconstructed[(unsigned)j] - image.get()[(unsigned)j]);
		};
	}
	return cost / dataset.Count();
}

double HiddenLayer::ComputeCost(const DataSet& dataset, double noise) const
{
	std::uniform_real_distribution<double> dist(0, 1);
	auto corrupted = std::vector<double>((unsigned)nIn);
	auto latent = std::vector<double>((unsigned)nOut);
	auto reconstructed = std::vector<double>((unsigned)nIn);
	double cost = 0;
	for (unsigned int n = 0; n < dataset.Count(); n++)
	{
		auto image = hiddenLayers->Compute(dataset.Images()[n], this);
		for (int i = 0; i < nIn; i++)
			corrupted[(unsigned)i] = dist(hiddenLayers->RandomNumberGenerator) < noise ? 0 : image.get()[(unsigned)i];
		Activation([&](int j)
		{
			double ret = 0;
			for (int i = 0; i < nIn; i++)
				ret += corrupted[(unsigned)i] * Weight[j][i];
			return ret + Bias[(unsigned)j];
		}, latent);
		Activation([&](int j)
		{
			double ret = 0;
			for (int i = 0; i < nOut; i++)
				ret += latent[(unsigned)i] * Weight[i][j];
			return ret + visibleBias[(unsigned)j];
		}, reconstructed);
		cost += ErrorFunction::BiClassCrossEntropy(image.get(), reconstructed);
	}
	return cost / dataset.Count();
}

referenceable_vector<double> HiddenLayerCollection::Compute(const std::vector<double>& input, const HiddenLayer* stopLayer) const
{
	referenceable_vector<double> result(input);
	for (unsigned int i = 0; i < items.size() && items[i].get() != stopLayer; i++)
		result = items[i]->Compute(result.get());
	return std::move(result);
}

void HiddenLayerCollection::Set(size_t index, int neurons)
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

int LogisticRegressionLayer::Predict(const std::vector<double>& input) const
{
	auto computed = Compute(input);
	int maxIndex = 0;
	for (int i = 1; i < nOut; i++)
	{
		if (computed[(unsigned)i] > computed[(unsigned)maxIndex])
			maxIndex = i;
	}
	return maxIndex;
}
