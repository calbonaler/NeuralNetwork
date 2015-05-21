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
	{ 500, 500, 1, static_cast<Floating>(0.1) },
	{ 500, 10000, 100, static_cast<Floating>(0.2) },
	{ 500, 10000, 100, static_cast<Floating>(0.3) },
};

const int FineTuningEpochs = 30;
const ValueType FineTuningLearningRate = static_cast<ValueType>(0.01);

// ニューロン数自動化 メモ
// できるだけ計算をさせないことが重要
// 最終エポックでのテストコスト予測はもちろん、ニューロン数ごとの最終テストコストも予測することで
// チェックすべきニューロン数の組み合わせを減少させる。

void TestSdA(const LearningSet& datasets)
{
	// seed: 89677
	std::random_device random;
	for (int i = 0; i < 10; i++)
	{
		StackedDenoisingAutoEncoder sda(random(), datasets.TrainingData().Row() * datasets.TrainingData().Column());
		ValueType lastLayerTestCost = 0;
		for (unsigned int i = 0; i < sizeof(PreTrainingConfigurations) / sizeof(PreTrainingConfigurations[0]); i++)
		{
			ValueType testCost = 0;
			for (unsigned int neurons = PreTrainingConfigurations[i].MinNeurons; neurons <= PreTrainingConfigurations[i].MaxNeurons; neurons += PreTrainingConfigurations[i].NeuronIncrement)
			{
				sda.HiddenLayers.Set(i, neurons);
				std::cout << "\"" << "Number of neurons of pre-training layer " << i << " is " << neurons << "\"" << std::endl;
				//ValueType lastEpochTestCost = 0;
				for (unsigned int epoch = 1; epoch <= PreTrainingEpochs; epoch++)
				{
					//ValueType last2EpochTestCost = lastEpochTestCost;
					//lastEpochTestCost = testCost;
					ValueType trainCost = sda.HiddenLayers[i].Train(datasets.TrainingData(), PreTrainingLearningRate, PreTrainingConfigurations[i].Noise);
					testCost = sda.HiddenLayers[i].ComputeCost(datasets.TestData(), PreTrainingConfigurations[i].Noise);
					// TODO: 最終エポックでのテストコストの予測が不完全であるため、早期終了は一時取りやめ
					//if (epoch == 3 && i > 0)
					//{
					//	ValueType coeff = (testCost - last2EpochTestCost) / (static_cast<ValueType>(1) / epoch - static_cast<ValueType>(1) / (epoch - 2));
					//	ValueType bias = testCost - coeff / epoch;
					//	ValueType guessedTestCost = coeff / 15 + bias;
					//	std::cout << epoch << " " << trainCost << " " << testCost << " " << guessedTestCost << std::endl;
					//	if (guessedTestCost > lastLayerTestCost)
					//		goto increaseNeuron;
					//}
					//else
					std::cout << epoch << " " << trainCost << " " << testCost << " none" << std::endl;
				}
				if (testCost <= lastLayerTestCost)
					break;
			increaseNeuron:;
			}
			lastLayerTestCost = testCost;
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