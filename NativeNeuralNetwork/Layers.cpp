#include <cassert>
#include "Layers.h"

Layer::Layer(unsigned int nIn, unsigned int nOut, const ActivationFunction::NormalForm& activation) : nIn(nIn), nOut(nOut), Weight(new double*[nOut]), Bias(new double[nOut]()), activation(activation)
{
	assert(nIn * nOut > 0);
	Weight[0] = new double[nIn * nOut]();
	for (unsigned int i = 1; i < nOut; i++)
		Weight[i] = Weight[i - 1] + nOut;
}

Layer::~Layer()
{
	delete[] Weight[0];
	delete[] Weight;
	delete[] Bias;
}

double* Layer::Compute(const double* input) const
{
	assert(input);
	double* output = new double[nOut];
	activation([&](unsigned int j)
	{
		double ret = 0;
		for (unsigned int k = 0; k < nIn; k++)
			ret += input[k] * Weight[j][k];
		return ret + Bias[j];
	}, output, nOut);
	return output;
}

double* Layer::Learn(const double* input, const double* output, Indexer upperInfo, double learningRate)
{
	assert(input);
	assert(output);
	double* lowerInfo = new double[nIn]();
	// TODO: Check initialized
	for (unsigned int i = 0; i < nOut; i++)
	{
		auto deltaI = GetDelta(output[i], upperInfo(i));
		for (unsigned int j = 0; j < nIn; j++)
		{
			lowerInfo[j] += Weight[i][j] * deltaI;
			Weight[i][j] -= learningRate * (deltaI * input[j]);
		}
		Bias[i] -= learningRate * deltaI;
	}
	return lowerInfo;
}

HiddenLayer::HiddenLayer(unsigned int nIn, unsigned int nOut, const ActivationFunction* activation, HiddenLayerCollection* hiddenLayers) : Layer(nIn, nOut, activation->Normal), differentiatedActivation(activation->Differentiated), visibleBias(new double[nIn]()), hiddenLayers(hiddenLayers)
{
	assert(activation);
	assert(hiddenLayers);
	std::uniform_real_distribution<double> dist(0, nextafter(1.0, std::numeric_limits<double>::max()));
	for (unsigned int j = 0; j < nOut; j++)
	{
		for (unsigned int i = 0; i < nIn; i++)
		{
			Weight[j][i] = (2 * dist(hiddenLayers->RandomNumberGenerator) - 1) * sqrt(6.0 / (nIn + nOut));
			if (activation == ActivationFunction::Sigmoid())
				Weight[j][i] *= 4;
		}
	}
}

HiddenLayer::~HiddenLayer()
{
	delete[] visibleBias;
}

double HiddenLayer::Train(const DataSet& dataset, double learningRate, double noise)
{
	std::uniform_real_distribution<double> dist(0, 1);
	auto corrupted = std::make_unique<double[]>(nIn);
	auto latent = std::make_unique<double[]>(nOut);
	auto reconstructed = std::make_unique<double[]>(nIn);
	auto delta = std::make_unique<double[]>(nOut);
	double cost = 0;
	for (unsigned int n = 0; n < dataset.Count(); n++)
	{
		auto image = hiddenLayers->Compute(dataset.Images()[n], this);
		for (unsigned int i = 0; i < nIn; i++)
			corrupted[i] = dist(hiddenLayers->RandomNumberGenerator) < noise ? 0 : image[i];
		ActivationFunction::Sigmoid()->Normal([&](int j)
		{
			double ret = 0;
			for (unsigned int i = 0; i < nIn; i++)
				ret += corrupted[i] * Weight[j][i];
			return ret + Bias[j];
		}, latent.get(), nOut);
		ActivationFunction::Sigmoid()->Normal([&](int j)
		{
			double ret = 0;
			for (unsigned int i = 0; i < nOut; i++)
				ret += latent[i] * Weight[i][j];
			return ret + visibleBias[j];
		}, reconstructed.get(), nIn);
		cost += ErrorFunction::BiClassCrossEntropy(image, reconstructed.get(), nIn);
		for (unsigned int i = 0; i < nOut; i++)
		{
			delta[i] = 0;
			for (unsigned int j = 0; j < nIn; j++)
				delta[i] += (reconstructed[j] - image[j]) * Weight[i][j];
			delta[i] *= ActivationFunction::Sigmoid()->Differentiated(latent[i]);
			Bias[i] -= learningRate * delta[i];
		};
		for (unsigned int j = 0; j < nIn; j++)
		{
			for (unsigned int i = 0; i < nOut; i++)
				Weight[i][j] -= learningRate * ((reconstructed[j] - image[j]) * latent[i] + delta[i] * corrupted[j]);
			visibleBias[j] -= learningRate * (reconstructed[j] - image[j]);
		};
		if (image != dataset.Images()[n])
			delete[] image;
	}
	return cost / dataset.Count();
}

double HiddenLayer::ComputeCost(const DataSet& dataset, double noise) const
{
	std::uniform_real_distribution<double> dist(0, 1);
	auto corrupted = std::make_unique<double[]>(nIn);
	auto latent = std::make_unique<double[]>(nOut);
	auto reconstructed = std::make_unique<double[]>(nIn);
	double cost = 0;
	for (unsigned int n = 0; n < dataset.Count(); n++)
	{
		auto image = hiddenLayers->Compute(dataset.Images()[n], this);
		for (unsigned int i = 0; i < nIn; i++)
			corrupted[i] = dist(hiddenLayers->RandomNumberGenerator) < noise ? 0 : image[i];
		ActivationFunction::Sigmoid()->Normal([&](int j)
		{
			double ret = 0;
			for (unsigned int i = 0; i < nIn; i++)
				ret += corrupted[i] * Weight[j][i];
			return ret + Bias[j];
		}, latent.get(), nOut);
		ActivationFunction::Sigmoid()->Normal([&](int j)
		{
			double ret = 0;
			for (unsigned int i = 0; i < nOut; i++)
				ret += latent[i] * Weight[i][j];
			return ret + visibleBias[j];
		}, reconstructed.get(), nIn);
		cost += ErrorFunction::BiClassCrossEntropy(image, reconstructed.get(), nIn);
		if (image != dataset.Images()[n])
			delete[] image;
	}
	return cost / dataset.Count();
}

const double* HiddenLayerCollection::Compute(const double* input, const HiddenLayer* stopLayer) const
{
	for (unsigned int i = 0; i < items.size() && items[i].get() != stopLayer; i++)
	{
		auto newInput = items[i]->Compute(input);
		if (i > 0)
			delete[] input;
		input = newInput;
	}
	return input;
}

void HiddenLayerCollection::Set(unsigned int index, unsigned int neurons)
{
	if (frozen)
		throw std::domain_error("ŒÅ’è‚³‚ê‚½ƒRƒŒƒNƒVƒ‡ƒ“‚Ì‰B‚ê‘w‚ðÝ’è‚·‚é‚±‚Æ‚Í‚Å‚«‚Ü‚¹‚ñB");
	if (index > items.size())
		throw std::out_of_range("index");
	if (index == items.size())
	{
		items.push_back(std::make_unique<HiddenLayer>(nextLayerInputUnits, neurons, ActivationFunction::Sigmoid(), this));
		nextLayerInputUnits = neurons;
		return;
	}
	items[index] = std::make_unique<HiddenLayer>(items[index]->nIn, neurons, ActivationFunction::Sigmoid(), this);
	if (index < items.size() - 1)
		items[index + 1] = std::make_unique<HiddenLayer>(neurons, items[index + 1]->nOut, ActivationFunction::Sigmoid(), this);
}

unsigned int LogisticRegressionLayer::Predict(const double* input) const
{
	auto computed = std::unique_ptr<double[]>(Compute(input));
	unsigned int maxIndex = 0;
	for (unsigned int i = 1; i < nOut; i++)
	{
		if (computed[i] > computed[maxIndex])
			maxIndex = i;
	}
	return maxIndex;
}