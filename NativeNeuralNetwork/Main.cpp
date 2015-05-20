#include "StackedDenoisingAutoEncoder.h"

void TestSdA(const LearningSet& datasets);

int main()
{
	TestSdA(LearningSet(LearningSetLoader::LoadMnistSet("MNIST"), 5000, 1000));
	return 0;
}

const int PreTrainingEpochs = 15;
const ValueType PreTrainingLearningRate = static_cast<ValueType>(0.001);
const struct
{
	unsigned int MinNeurons;
	unsigned int MaxNeurons;
	unsigned int NeuronIncrement;
	Floating Noise;
} PreTrainingConfigurations[]
{
	{ 100, 100, 1, static_cast<Floating>(0.1) },
	{ 100, 100, 1, static_cast<Floating>(0.2) },
	{ 100, 100, 1, static_cast<Floating>(0.3) },
};

const int FineTuningEpochs = 30;
const ValueType FineTuningLearningRate = static_cast<ValueType>(0.01);

void TestSdA(const LearningSet& datasets)
{
	// seed: 89677
	std::random_device random;
	for (int i = 0; i < 10; i++)
	{
		StackedDenoisingAutoEncoder sda(random(), datasets.TrainingData().Row() * datasets.TrainingData().Column());
		for (unsigned int i = 0; i < sizeof(PreTrainingConfigurations) / sizeof(PreTrainingConfigurations[0]); i++)
		{
			for (unsigned int neurons = PreTrainingConfigurations[i].MinNeurons; neurons <= PreTrainingConfigurations[i].MaxNeurons; neurons += PreTrainingConfigurations[i].NeuronIncrement)
			{
				sda.HiddenLayers.Set(i, neurons);
				std::cout << "\"" << "Number of neurons of pre-training layer " << i << " is " << neurons << "\"" << std::endl;
				for (unsigned int epoch = 1; epoch <= PreTrainingEpochs; epoch++)
				{
					ValueType costTrain = sda.HiddenLayers[i].Train(datasets.TrainingData(), PreTrainingLearningRate, PreTrainingConfigurations[i].Noise);
					ValueType costTest = sda.HiddenLayers[i].ComputeCost(datasets.TestData(), PreTrainingConfigurations[i].Noise);
					std::cout << epoch << " " << costTrain << " " << costTest << std::endl;
				}
			}
		}

		sda.SetLogisticRegressionLayer(datasets.ClassCount);
		std::cout << "\"" << "Fine-Tuning..." << "\"" << std::endl;
		for (unsigned int epoch = 1; epoch <= FineTuningEpochs; epoch++)
		{
			sda.FineTune(datasets.TrainingData(), FineTuningLearningRate);
			Floating thisTestLoss = sda.ComputeErrorRates(datasets.TestData());
			std::cout << epoch << " " << thisTestLoss * 100.0 << "%" << std::endl;
		}
	}
}