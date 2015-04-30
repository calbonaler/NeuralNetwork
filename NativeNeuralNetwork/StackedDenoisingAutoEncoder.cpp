#include "StackedDenoisingAutoEncoder.h"

void StackedDenoisingAutoEncoder::SetLogisticRegressionLayer(int neurons)
{
	outputLayer = std::unique_ptr<LogisticRegressionLayer>(new LogisticRegressionLayer(HiddenLayers[HiddenLayers.Count() - 1].nOut, neurons));
	HiddenLayers.Freeze();
}

void StackedDenoisingAutoEncoder::FineTune(const DataSet& dataset, double learningRate)
{
	auto inputs = std::vector<referenceable_vector<double>>(HiddenLayers.Count() + 2);
	for (unsigned int d = 0; d < dataset.Count(); d++)
	{
		inputs[0] = dataset.Images()[d];
		unsigned int n = 0;
		for (; n < HiddenLayers.Count(); n++)
			inputs[n + 1] = HiddenLayers[n].Compute(inputs[n].get());
		inputs[n + 1] = outputLayer->Compute(inputs[n].get());
		Indexer upperInfo = [&](int i) { return i == dataset.Labels()[d] ? 1.0 : 0.0; };
		std::vector<double> lowerInfo(outputLayer->Learn(inputs[n].get(), inputs[n + 1].get(), upperInfo, learningRate));
		while (--n <= HiddenLayers.Count())
		{
			upperInfo = [&](int i) { return lowerInfo[(unsigned)i]; };
			lowerInfo = HiddenLayers[n].Learn(inputs[n].get(), inputs[n + 1].get(), upperInfo, learningRate);
		}
	}
}

double StackedDenoisingAutoEncoder::ComputeErrorRates(const DataSet& dataset)
{
	int sum = 0;
	for (unsigned int i = 0; i < dataset.Count(); i++)
	{
		auto hidden = HiddenLayers.Compute(dataset.Images()[i], nullptr);
		if (outputLayer->Predict(hidden.get()) != dataset.Labels()[i])
			sum++;
	}
	return static_cast<double>(sum) / dataset.Count();
}