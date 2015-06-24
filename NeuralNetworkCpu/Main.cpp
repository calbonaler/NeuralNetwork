#include "StackedDenoisingAutoEncoder.h"
#include "ShiftRegister.h"

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
	{ 500, 500, 100, static_cast<Floating>(0) }, // Noise: 0.1
	{ 100, 5000, 100, static_cast<Floating>(0) }, // Noise: 0.2
	{ 100, 100, 100, static_cast<Floating>(0.3) }, // Noise: 0.3
};

const int FineTuningEpochs = 30;
const ValueType FineTuningLearningRate = static_cast<ValueType>(0.01);

// ニューロン数自動化 メモ
// できるだけ計算をさせないことが重要
// 最終エポックでのテストコスト予測はもちろん、ニューロン数ごとの最終テストコストも予測することで
// チェックすべきニューロン数の組み合わせを減少させる。

struct LossPredictor
{
	LossPredictor() : a(0), b(0), c(0) { }
	LossPredictor(const LossPredictor& src) : a(src.a), b(src.b), c(src.c) { }
	LossPredictor& operator=(const LossPredictor& src)
	{
		a = src.a;
		b = src.b;
		c = src.c;
		return *this;
	}
	double operator()(unsigned int n) { return a * pow(n, b) + c; }
	template <class Vector1, class Vector2> void Setup(const Vector1& sources, const Vector2& targets)
	{
		auto constant = (targets[2] - targets[1]) / (targets[1] - targets[0]);
		auto f = [&](double b) { return (pow(sources[2], b) - pow(sources[1], b)) / (pow(sources[1], b) - pow(sources[0], b)) - constant; };
		auto df = [&](double b)
		{
			double denomi = pow(sources[1], b) - pow(sources[0], b);
			double sum = 0;
			for (unsigned int i = 0; i < 3; i++)
			{
				double x_i1 = sources[(i + 1) % 3];
				sum += pow(sources[i], b) * pow(x_i1, b) * log(x_i1 / sources[i]);
			}
			return sum / denomi / denomi;
		};
		auto d2f = [&](double b)
		{
			double denomi = pow(sources[1], b) - pow(sources[0], b);
			double res = 2 * (pow(sources[1], b) * log(sources[1]) - pow(sources[0], b) * log(sources[0]));
			double sum = 0;
			for (unsigned int i = 0; i < 3; i++)
			{
				double x_i1 = sources[(i + 1) % 3];
				sum += (res - denomi * log(sources[i] * x_i1)) * pow(sources[i], b) * pow(x_i1, b) * log(x_i1 / sources[i]);
			}
			return sum / denomi / denomi / denomi;
		};
		double b_hat = -1, delta;
		do
		{
			auto y = f(b_hat);
			auto dy = df(b_hat);
			delta = 2 * dy * y / (2 * dy * dy - y * d2f(b_hat));
			b_hat -= delta;
		} while (abs(delta) > 1e-10);
		a = (targets[1] - targets[0]) / (pow(sources[1], b_hat) - pow(sources[0], b_hat));
		b = b_hat;
		c = targets[0] - a * pow(sources[0], b_hat);
	}

private:
	double a, b, c;
};

bool TrainLayer(std::ofstream& output, StackedDenoisingAutoEncoder& sda, unsigned int i, unsigned int neurons, const LearningSet& datasets, ShiftRegister<ValueType, 3>* testCosts, LossPredictor* predictor)
{
	sda.HiddenLayers.Set(i, neurons);
	std::cout << "\"" << "Number of neurons of pre-training layer " << i << " is " << neurons << "\"" << std::endl;
	for (unsigned int epoch = 1; epoch <= PreTrainingEpochs; epoch++)
	{
		ValueType trainCost = sda.HiddenLayers[i].Train(datasets.TrainingData(), PreTrainingLearningRate, PreTrainingConfigurations[i].Noise);
		ValueType testCost = sda.HiddenLayers[i].ComputeCost(datasets.TestData(), PreTrainingConfigurations[i].Noise);
		if (testCosts)
			testCosts->Push(testCost);
		if (predictor && epoch >= 4)
		{
			if (epoch == 4)
			{
				double sources[] { epoch - 2, epoch - 1, epoch };
				predictor->Setup(sources, *testCosts);
			}
			double predictedTestCost = (*predictor)(epoch);
			std::cout << epoch << " " << trainCost << " " << testCost << " " << predictedTestCost << std::endl;
			output << neurons << " " << epoch << " " << trainCost << " " << testCost << " " << predictedTestCost << std::endl;
			//if (predictor(PreTrainingEpochs) > lastLayerTestCost)
			//	goto increaseNeuron;
		}
		else
		{
			std::cout << epoch << " " << trainCost << " " << testCost << " none" << std::endl;
			output << neurons << " " << epoch << " " << trainCost << " " << testCost << std::endl;
		}
	}
	return true;
}

void TestSdA(const LearningSet& datasets)
{
	// seed: 89677
	std::random_device random;
	StackedDenoisingAutoEncoder sda(random(), datasets.TrainingData().Row() * datasets.TrainingData().Column());
	ValueType lastLayerTestCost = 0;
	LossPredictor predictor;
	std::ofstream output("output.log");

	TrainLayer(output, sda, 0, PreTrainingConfigurations[0].MinNeurons, datasets, nullptr, nullptr);

	for (unsigned int i = 1; i < sizeof(PreTrainingConfigurations) / sizeof(PreTrainingConfigurations[0]); i++)
	{
		ShiftRegister<ValueType, 3> testCosts;
		for (unsigned int neurons = PreTrainingConfigurations[i].MinNeurons; neurons <= PreTrainingConfigurations[i].MaxNeurons; neurons += PreTrainingConfigurations[i].NeuronIncrement)
		{
			if (TrainLayer(output, sda, i, neurons, datasets, &testCosts, &predictor))
				continue;
			if (testCosts[-1] <= lastLayerTestCost)
				break;
		}
		return;
		lastLayerTestCost = testCosts[-1];
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