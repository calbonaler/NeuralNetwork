#include "StackedDenoisingAutoEncoder.h"
#include "ShiftRegister.h"

enum class DataSetKind
{
	MNIST,
	Cifar10,
	Caltech101Silhouettes,
	PR,
};

template <class TValue> LearningSet<TValue> LoadLearningSet(DataSetKind kind);
template <class TValue> void TestSdA(const LearningSet<TValue>& datasets);

typedef double Floating;

int main()
{
	TestSdA(LoadLearningSet<Floating>(DataSetKind::Caltech101Silhouettes));
	return 0;
}

template <class TValue> LearningSet<TValue> LoadLearningSet(DataSetKind kind)
{
	if (kind == DataSetKind::MNIST)
	{
		auto ls = MnistLoader<TValue>().Load("MNIST");
		LearningSet<TValue> newLs;
		newLs.ClassCount = ls.ClassCount;
		newLs.TrainingData().From(std::move(ls.TrainingData()), 0, 50000);
		newLs.ValidationData().From(std::move(ls.TrainingData()), 50000, 10000);
		newLs.TestData().From(std::move(ls.TestData()), 0, 10000);
		return std::move(newLs);
	}
	else if (kind == DataSetKind::Cifar10)
	{
		auto ls = Cifar10Loader<TValue>().Load("cifar-10-batches-bin");
		LearningSet<TValue> newLs;
		newLs.ClassCount = ls.ClassCount;
		newLs.TrainingData().From(std::move(ls.TrainingData()), 0, 40000);
		newLs.ValidationData().From(std::move(ls.TrainingData()), 40000, 10000);
		newLs.TestData().From(std::move(ls.TestData()), 0, 10000);
		return std::move(newLs);
	}
	else if (kind == DataSetKind::Caltech101Silhouettes)
		return std::move(Caltech101SilhouettesLoader<TValue>().Load("Caltech101Silhouettes"));
	else if (kind == DataSetKind::PR)
		return std::move(PatternRecognitionLoader<TValue>().Load("PR"));
	else
		return std::move(LearningSet<TValue>());
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

const double ConvergeConstant = 0.5;

template <class TValue, class TNoise> bool PreTrain(std::ofstream& output, HiddenLayerCollection<TValue>& hiddenLayers, unsigned int i, unsigned int neurons, TNoise noise, const LearningSet<TValue>& datasets, double& lastNeuronCost, unsigned int lastNeurons)
{
	const int PreTrainingEpochs = 15;
	const TValue PreTrainingLearningRate = static_cast<TValue>(0.001);

	//LossPredictor<double, 3> predictor;
	double currentTestCost;
	hiddenLayers.Set(i, neurons);
	std::cout << boost::format("Number of neurons of pre-training layer %d is %d") % i % neurons << std::endl;
	for (unsigned int epoch = 1; epoch <= PreTrainingEpochs; epoch++)
	{
		hiddenLayers[i].Train(datasets.TrainingData(), PreTrainingLearningRate, noise);
		currentTestCost = hiddenLayers[i].ComputeCost(datasets.ValidationData(), noise);
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
			std::cout << boost::format("%2d %lf") % epoch % currentTestCost << std::endl;
			output << boost::format("PT %d %d %d %lf") % i % neurons % epoch % currentTestCost << std::endl;
		}
	}
	auto costRelativeError = abs((currentTestCost - lastNeuronCost) / (neurons - lastNeurons));
	std::cout << "Cost relative error: " << costRelativeError << std::endl;
	lastNeuronCost = currentTestCost;
	return costRelativeError <= ConvergeConstant;
}

template <class TValue> void FineTune(std::ofstream& output, StackedDenoisingAutoEncoder<TValue>& sda, const LearningSet<TValue>& datasets)
{
	const unsigned int FineTuningEpochs = 1000;
	const TValue FineTuningLearningRate = static_cast<TValue>(0.01);
	const unsigned int DefaultPatience = 10;
	const double ImprovementThreshold = 1;//0.995;
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
	} LayerConfiguration[]
	{
		{ 1, 5000, 2, static_cast<Floating>(0.1) }, // Noise: 0.1
		{ 1, 5000, 2, static_cast<Floating>(0.2) }, // Noise: 0.2
		{ 1, 5000, 2, static_cast<Floating>(0.3) }, // Noise: 0.3
	};

	// seed: 89677
	std::random_device random;
	StackedDenoisingAutoEncoder<TValue> sda(random(), datasets.TrainingData().AllComponents());
	
	std::ofstream output("output.log");
	double lastNeuronCost;

	for (unsigned int i = 0; i < sizeof(LayerConfiguration) / sizeof(LayerConfiguration[0]); i++)
	{
		lastNeuronCost = std::numeric_limits<double>::infinity();
		for (unsigned int neurons = LayerConfiguration[i].MinNeurons; neurons <= LayerConfiguration[i].MaxNeurons; neurons *= LayerConfiguration[i].NeuronIncrement)
		{
			std::cout << "Last neuron cost: " << lastNeuronCost << std::endl;
			if (PreTrain(output, sda.HiddenLayers, i, neurons, LayerConfiguration[i].Noise, datasets, lastNeuronCost, neurons / LayerConfiguration[i].NeuronIncrement))
				break;
		}
	}

	FineTune(output, sda, datasets);

	auto end = system_clock::now();
	std::cout << "Elapsed time (seconds): " << duration_cast<seconds>(end - start).count() << std::endl;
}