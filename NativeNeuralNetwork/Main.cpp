#include "StackedDenoisingAutoEncoder.h"

void TestSdA(const LearningSet& datasets);

int main()
{
	TestSdA(LearningSet(LearningSetLoader::LoadMnistSet("MNIST"), 5000, 1000));
	return 0;
}

const int PreTrainingEpochs = 15;
const ValueType PreTrainingLearningRate = static_cast<ValueType>(0.001);
const Floating PreTrainingCorruptionLevels[] { static_cast<Floating>(0.1), static_cast<Floating>(0.2), static_cast<Floating>(0.3) };

const int FineTuningEpochs = 100;
const ValueType FineTuningLearningRate = static_cast<ValueType>(0.01);

void TestSdA(const LearningSet& datasets)
{
	StackedDenoisingAutoEncoder sda(89677, datasets.TrainingData().Row() * datasets.TrainingData().Column());
	std::ofstream log("Experiments (Variable Neurons).log", std::ios::out);
	for (unsigned int i = 0; i < 3; i++)
	{
		for (unsigned int neurons = 100; neurons <= 100; neurons += 100)
		{
			sda.HiddenLayers.Set(i, neurons);
			std::cout << "Number of neurons of layer " << i << " is " << neurons << std::endl;
			std::cout << "... pre-training the model" << std::endl;
			ValueType costTrain = 0;
			for (unsigned int epoch = 1; epoch <= PreTrainingEpochs; epoch++)
			{
				costTrain = sda.HiddenLayers[i].Train(datasets.TrainingData(), PreTrainingLearningRate, PreTrainingCorruptionLevels[i]);
				std::cout << "Pre-training layer " << i << ", epoch " << epoch << ", cost " << costTrain << std::endl;
			}
			ValueType costTest = sda.HiddenLayers[i].ComputeCost(datasets.TestData(), PreTrainingCorruptionLevels[i]);
			std::cout << "Pre-training layer " << i << " complete with training cost " << costTrain << ", test cost " << costTest << std::endl;
			log << i << ", " << neurons << ", " << costTrain << ", " << costTest << std::endl;
		}
	}
	
	sda.SetLogisticRegressionLayer(datasets.ClassCount);
	std::cout << "... finetunning the model" << std::endl;
	Floating testScore = std::numeric_limits<Floating>::infinity();
	unsigned int bestEpoch = 0;
	for (unsigned int epoch = 1; epoch < FineTuningEpochs; epoch++)
	{
		sda.FineTune(datasets.TrainingData(), FineTuningLearningRate);
		Floating thisTestLoss = sda.ComputeErrorRates(datasets.TestData());
		std::cout << "epoch " << epoch << ", test score " << thisTestLoss * 100.0 << " %" << std::endl;
		if (thisTestLoss < testScore)
		{
			testScore = thisTestLoss;
			bestEpoch = epoch;
		}
	}
	std::cout << "Optimization complete with best test score of " << testScore * 100.0 << " %, on epoch " << bestEpoch << std::endl;
}