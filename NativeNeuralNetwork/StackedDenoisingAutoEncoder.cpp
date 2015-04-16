#include "StackedDenoisingAutoEncoder.h"

void StackedDenoisingAutoEncoder::SetLogisticRegressionLayer(unsigned int neurons)
{
	outputLayer = std::make_unique<LogisticRegressionLayer>(HiddenLayers[HiddenLayers.Count() - 1].nOut, neurons);
	HiddenLayers.Freeze();
}

void StackedDenoisingAutoEncoder::FineTune(const DataSet& dataset, double learningRate)
{
	std::unique_ptr<double*[]> inputs = std::make_unique<double*[]>(HiddenLayers.Count() + 2);
	for (unsigned int d = 0; d < dataset.Count(); d++)
	{
		inputs[0] = dataset.Images()[d];
		unsigned int n = 0;
		for (; n < HiddenLayers.Count(); n++)
			inputs[n + 1] = HiddenLayers[n].Compute(inputs[n]);
		inputs[n + 1] = outputLayer->Compute(inputs[n]);
		Indexer upperInfo = [&](unsigned int i) { return i == dataset.Labels()[d] ? 1.0 : 0.0; };
		std::unique_ptr<double[]> lowerInfo(outputLayer->Learn(inputs[n], inputs[n + 1], upperInfo, learningRate));
		while (--n <= HiddenLayers.Count())
		{
			upperInfo = [&](unsigned int i) { return lowerInfo[i]; };
			lowerInfo = std::unique_ptr<double[]>(HiddenLayers[n].Learn(inputs[n], inputs[n + 1], upperInfo, learningRate));
			delete[] inputs[n + 1];
		}
	}
}

double StackedDenoisingAutoEncoder::ComputeErrorRates(const DataSet& dataset)
{
	int sum = 0;
	for (unsigned int i = 0; i < dataset.Count(); i++)
	{
		auto hidden = HiddenLayers.Compute(dataset.Images()[i], nullptr);
		if (outputLayer->Predict(hidden) != dataset.Labels()[i])
			sum++;
		if (hidden != dataset.Images()[i])
			delete[] hidden;
	}
	return static_cast<double>(sum) / dataset.Count();
}