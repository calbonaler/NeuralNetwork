#include <iostream>
#include <fstream>
#include "StackedDenoisingAutoEncoder.h"

void TestSdA(const LearningSet& datasets);

int main()
{
	TestSdA(LearningSet(LoadMnistSet("MNIST"), 5000, 1000));
	return 0;
}

const int PreTrainingEpochs = 15;
const double PreTrainingLearningRate = 0.001;
const double PreTrainingCorruptionLevels[] { 0.1, 0.2, 0.3 };

const int FineTuningEpochs = 100;
const double FineTuningLearningRate = 0.01;

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
			double costTrain = 0;
			for (int epoch = 1; epoch <= PreTrainingEpochs; epoch++)
			{
				costTrain = sda.HiddenLayers[i].Train(datasets.TrainingData(), PreTrainingLearningRate, PreTrainingCorruptionLevels[i]);
				std::cout << "Pre-training layer " << i << ", epoch " << epoch << ", cost " << costTrain << std::endl;
			}
			auto costTest = sda.HiddenLayers[i].ComputeCost(datasets.TestData(), PreTrainingCorruptionLevels[i]);
			std::cout << "Pre-training layer " << i << " complete with training cost " << costTrain << ", test cost " << costTest << std::endl;
			log << i << ", " << neurons << ", " << costTrain << ", " << costTest << std::endl;
		}
	}
	
	sda.SetLogisticRegressionLayer(datasets.ClassCount);
	std::cout << "... finetunning the model" << std::endl;
	auto testScore = std::numeric_limits<double>::infinity();
	auto bestEpoch = 0;
	for (int epoch = 1; epoch < FineTuningEpochs; epoch++)
	{
		sda.FineTune(datasets.TrainingData(), FineTuningLearningRate);
		auto thisTestLoss = sda.ComputeErrorRates(datasets.TestData());
		std::cout << "epoch " << epoch << ", test score " << thisTestLoss * 100.0 << " %" << std::endl;
		if (thisTestLoss < testScore)
		{
			testScore = thisTestLoss;
			bestEpoch = epoch;
		}
	}
	std::cout << "Optimization complete with best test score of " << testScore * 100.0 << " %, on epoch " << bestEpoch << std::endl;
}