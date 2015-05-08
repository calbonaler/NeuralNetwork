#include "StackedDenoisingAutoEncoder.h"

void StackedDenoisingAutoEncoder::SetLogisticRegressionLayer(unsigned int neurons)
{
	outputLayer = std::unique_ptr<LogisticRegressionLayer>(new LogisticRegressionLayer(HiddenLayers[HiddenLayers.Count() - 1].nOut, neurons));
	HiddenLayers.Freeze();
}

void StackedDenoisingAutoEncoder::FineTune(const DataSet& dataset, ValueType learningRate)
{
	auto inputs = std::vector<ReferableVector>(HiddenLayers.Count() + 2);
	for (unsigned int d = 0; d < dataset.Count(); d++)
	{
		inputs[0] = dataset.Images()[d];
		unsigned int n = 0;
		for (; n < HiddenLayers.Count(); n++)
			inputs[n + 1] = HiddenLayers[n].Compute(inputs[n].get());
		inputs[n + 1] = outputLayer->Compute(inputs[n].get());
		Indexer upperInfo = [&](unsigned int i) { return i == dataset.Labels()[d] ? static_cast<ValueType>(1.0) : static_cast<ValueType>(0.0); };
		VectorType lowerInfo(outputLayer->Learn(inputs[n].get(), inputs[n + 1].get(), upperInfo, learningRate));
		while (--n <= HiddenLayers.Count())
		{
			upperInfo = [&](unsigned int i) { return lowerInfo[i]; };
			lowerInfo = HiddenLayers[n].Learn(inputs[n].get(), inputs[n + 1].get(), upperInfo, learningRate);
		}
	}
}

Floating StackedDenoisingAutoEncoder::ComputeErrorRates(const DataSet& dataset)
{
	int sum = 0;
	for (unsigned int i = 0; i < dataset.Count(); i++)
	{
		if (outputLayer->Predict(HiddenLayers.Compute(dataset.Images()[i], nullptr).get()) != dataset.Labels()[i])
			sum++;
	}
	return static_cast<Floating>(sum) / dataset.Count();
}