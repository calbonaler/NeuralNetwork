#include "StackedDenoisingAutoEncoder.h"
#include "ShiftRegister.h"

template <class TValue> void TestSdA(const LearningSet<TValue>& datasets);

typedef double Floating;

int main()
{
	auto ls = MnistLoader<Floating>().Load("MNIST");
	//auto ls = PatternRecognitionLoader<Floating>().Load("PR");
	TestSdA(ls);
	return 0;
}

// ニューロン数自動化 メモ
// できるだけ計算をさせないことが重要
// 最終エポックでのテストコスト予測はもちろん、ニューロン数ごとの最終テストコストも予測することで
// チェックすべきニューロン数の組み合わせを減少させる。

template <class T, size_t N> class LossPredictor final
{
public:
	LossPredictor() : a(), b(), c() { }
	LossPredictor(const LossPredictor& src) : a(src.a), b(src.b), c(src.c) { }
	LossPredictor& operator=(const LossPredictor& src)
	{
		a = src.a;
		b = src.b;
		c = src.c;
		return *this;
	}
	void PushLoss(const T& loss) { losses.Push(loss); }
	void PushLoss(T&& loss) { losses.Push(std::move(loss)); }
	void Setup(unsigned int currentEpoch)
	{
		T sources[N];
		for (size_t i = 0; i < N; i++)
			sources[i] = currentEpoch - N + i + 1;
		Setup(sources, losses);
	}
	double operator()(unsigned int epoch) const { return a * pow(epoch, b) + c; }
	std::string GetExpression() const
	{
		std::stringstream ss;
		ss << boost::format("%lf * x ** %lf + %lf") % a % b % c;
		return std::move(ss.str());
	}

private:
	T a, b, c;
	ShiftRegister<T, N> losses;

	template <class Vector1, class Vector2> void Setup(const Vector1& sources, const Vector2& targets)
	{
		auto constant = (targets[2] - targets[1]) / (targets[1] - targets[0]);
		auto f = [&](const T& b) { return (pow(sources[2], b) - pow(sources[1], b)) / (pow(sources[1], b) - pow(sources[0], b)) - constant; };
		auto df = [&](const T& b)
		{
			T denomi = pow(sources[1], b) - pow(sources[0], b);
			T sum = 0;
			for (unsigned int i = 0; i < 3; i++)
			{
				const T& x_i1 = sources[(i + 1) % 3];
				sum += pow(sources[i], b) * pow(x_i1, b) * log(x_i1 / sources[i]);
			}
			return sum / denomi / denomi;
		};
		auto d2f = [&](const T& b)
		{
			T denomi = pow(sources[1], b) - pow(sources[0], b);
			T res = 2 * (pow(sources[1], b) * log(sources[1]) - pow(sources[0], b) * log(sources[0]));
			T sum = 0;
			for (unsigned int i = 0; i < 3; i++)
			{
				const T& x_i1 = sources[(i + 1) % 3];
				sum += (res - denomi * log(sources[i] * x_i1)) * pow(sources[i], b) * pow(x_i1, b) * log(x_i1 / sources[i]);
			}
			return sum / denomi / denomi / denomi;
		};
		T b_hat = -1;
		while (true)
		{
			auto y = f(b_hat);
			auto dy = df(b_hat);
			auto delta = 2 * dy * y / (2 * dy * dy - y * d2f(b_hat));
			b_hat -= delta;
			if (abs(delta) <= 1e-10)
				break;
		}
		a = (targets[1] - targets[0]) / (pow(sources[1], b_hat) - pow(sources[0], b_hat));
		b = b_hat;
		c = targets[0] - a * pow(sources[0], b_hat);
	}
};

const double PredictionEpsilon = 0.01;

inline bool IsConverged(double lastTestCost, double testCost) { return abs(testCost - lastTestCost) / testCost <= PredictionEpsilon; }

template <class TValue, class TNoise> bool PreTrain(std::ofstream& output, HiddenLayerCollection<TValue>& hiddenLayers, unsigned int i, unsigned int neurons, TNoise noise, const LearningSet<TValue>& datasets, double& lastNeuronCost)
{
	const int PreTrainingEpochs = 15;
	const TValue PreTrainingLearningRate = static_cast<TValue>(0.001);

	//LossPredictor<double, 3> predictor;
	hiddenLayers.Set(i, neurons);
	std::cout << boost::format("Number of neurons of pre-training layer %d is %d") % i % neurons << std::endl;
	for (unsigned int epoch = 1; epoch <= PreTrainingEpochs; epoch++)
	{
		hiddenLayers[i].Train(datasets.TrainingData(), PreTrainingLearningRate, noise);
		auto testCost = hiddenLayers[i].ComputeCost(datasets.TestData(), noise);
		//predictor.PushLoss(testCost);
		//if (epoch >= 4)
		//{
		//	if (epoch == 4)
		//	{
		//		predictor.Setup(epoch);
		//		std::cout << "Prediction Expression: " << predictor.GetExpression() << std::endl;
		//	}
		//	auto predictedTestCost = predictor(epoch);
		//	std::cout << boost::format("%2d %10lf %10lf") % epoch % testCost % predictedTestCost << std::endl;
		//	output << boost::format("PT %d %d %lf %lf") % neurons % epoch % testCost % predictedTestCost << std::endl;
		//	//if (epoch == 4 && !IsConverged(lastNeuronCost, predictor(PreTrainingEpochs), output))
		//	//{
		//	//	lastNeuronCost = predictor(PreTrainingEpochs);
		//	//	return false;
		//	//}
		//}
		//else
		{
			std::cout << boost::format("%2d %lf") % epoch % testCost << std::endl;
			output << boost::format("PT %d %d %d %lf") % i % neurons % epoch % testCost << std::endl;
		}
	}
	//bool result = IsConverged(lastNeuronCost, testCosts[-1], output);
	//lastNeuronCost = testCosts[-1];
	//return result;
	return false;
}

template <class TValue> void FineTune(std::ofstream& output, StackedDenoisingAutoEncoder<TValue>& sda, const LearningSet<TValue>& datasets)
{
	const unsigned int FineTuningEpochs = 1000;
	const TValue FineTuningLearningRate = static_cast<TValue>(0.01);
	const unsigned int DefaultPatience = 10;
	const double ImprovementThreshold = 0.995;
	const unsigned int PatienceIncrease = 2;

	double bestTestScore = std::numeric_limits<double>::infinity();
	sda.SetLogisticRegressionLayer(datasets.ClassCount);
	std::cout << "Fine-Tuning..." << std::endl;
	for (unsigned int epoch = 1, patience = DefaultPatience; epoch <= FineTuningEpochs && epoch <= patience; epoch++)
	{
		sda.FineTune(datasets.TrainingData(), FineTuningLearningRate);
		auto thisTestScore = sda.ComputeErrorRates<Floating>(datasets.TestData());
		std::cout << boost::format("%4d %lf%%") % epoch % (thisTestScore * 100.0) << std::endl;
		output << boost::format("FT %d %lf") % epoch % thisTestScore << std::endl;

		if (thisTestScore < bestTestScore)
		{
			if (thisTestScore < bestTestScore * ImprovementThreshold)
				patience = std::max(patience, epoch * PatienceIncrease);
			bestTestScore = thisTestScore;
		}
	}
}

template <class TValue> void TestSdA(const LearningSet<TValue>& datasets)
{
	using namespace std::chrono;

	auto start = system_clock::now();

	const struct
	{
		unsigned int MinNeurons;
		unsigned int MaxNeurons;
		unsigned int NeuronIncrement;
		Floating Noise;
	} LayerConfigurations[]
	{
		{ 1, 5000, 8, static_cast<Floating>(0.1) }, // Noise: 0.1
		{ 1, 5000, 8, static_cast<Floating>(0.2) }, // Noise: 0.2
		{ 1, 5000, 8, static_cast<Floating>(0.3) }, // Noise: 0.3
	};

	// seed: 89677
	std::random_device random;
	
	std::ofstream output("output.log");
	double lastNeuronCost = std::numeric_limits<double>::infinity();

	//for (unsigned int i = 0; i < sizeof(LayerConfiguration) / sizeof(LayerConfiguration[0]); i++)
	//{
	//	for (unsigned int neurons = LayerConfiguration[i].MinNeurons; neurons <= LayerConfiguration[i].MaxNeurons; neurons += LayerConfiguration[i].NeuronIncrement)
	//	{
	//		if (PreTrain(output, sda.HiddenLayers, i, neurons, LayerConfiguration[i].Noise, datasets, lastNeuronCost))
	//			break;
	//	}
	//	return;
	//}

	StackedDenoisingAutoEncoder<TValue> sda(random(), datasets.TrainingData().Row() * datasets.TrainingData().Column());
	PreTrain(output, sda.HiddenLayers, 0, 512, LayerConfigurations[0].Noise, datasets, lastNeuronCost);
	PreTrain(output, sda.HiddenLayers, 1, 512, LayerConfigurations[1].Noise, datasets, lastNeuronCost);
	PreTrain(output, sda.HiddenLayers, 2, 512, LayerConfigurations[2].Noise, datasets, lastNeuronCost);
	FineTune(output, sda, datasets);

	auto end = system_clock::now();
	std::cout << "Elapsed time (seconds): " << duration_cast<seconds>(end - start).count() << std::endl;
}