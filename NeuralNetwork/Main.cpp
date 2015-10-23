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

// Pre-Training Parameters

const int PreTrainingEpochs = 15;
const double PreTrainingLearningRate = 0.001;

// Fine-Tuning Parameters

const unsigned int FineTuningEpochs = 1000;
const double FineTuningLearningRate = 0.01;
const unsigned int DefaultPatience = 10;
const double ImprovementThreshold = 1;//0.995;
const unsigned int PatienceIncrease = 2;

// Denoising Auto-Encoder Parameters

const Floating DaNoises[]
{
	static_cast<Floating>(0.1),
	static_cast<Floating>(0.2),
	static_cast<Floating>(0.3),
};

// Number of Neuron Automatic Decision Parameters

const unsigned int MinNeurons = 1;
const unsigned int NeuronIncease = 2;
const double ConvergeConstant = 0.5;

class teed_out
{
public:
	teed_out() { }
	~teed_out() { close(); }

	void open(const std::string& outFileName)
	{
		outfile.open(outFileName);
		s.open(boost::iostreams::tee_device<std::ostream, std::ofstream>(std::cout, outfile));
	}
	void close()
	{
		s.close();
		outfile.flush();
		outfile.close();
	}
	
	boost::iostreams::stream<boost::iostreams::tee_device<std::ostream, std::ofstream>> s;

private:
	std::ofstream outfile;
};

teed_out tout;

void ShowParameters()
{
	tout.s << "All parameters of this experiment are as follows: " << std::endl;
	tout.s << "Pre-Training: " << std::endl;
	tout.s << "    Epochs: " << PreTrainingEpochs << std::endl;
	tout.s << "    Learning Rate: " << PreTrainingLearningRate << std::endl;
	tout.s << "    Noise Rate: " << std::endl;
	for (size_t i = 0; i < sizeof(DaNoises) / sizeof(DaNoises[0]); i++)
		tout.s << "        HL " << i << ": " << DaNoises[i] << std::endl;
	tout.s << "Fine-Tuning: " << std::endl;
	tout.s << "    Max Epochs: " << FineTuningEpochs << std::endl;
	tout.s << "    Learning Rate: " << FineTuningLearningRate << std::endl;
	tout.s << "    Early Stopping Parameters: " << std::endl;
	tout.s << "        Default Patience: " << DefaultPatience << std::endl;
	tout.s << "        Improvement Threshold: " << ImprovementThreshold << std::endl;
	tout.s << "        Patience Increase: " << PatienceIncrease << std::endl;
	tout.s << "Number of Neuron Automatic Decision Parameters: " << std::endl;
	tout.s << "    Minimum Number of Neurons: " << MinNeurons << std::endl;
	tout.s << "    Number of Neuron Increase: " << NeuronIncease << std::endl;
	tout.s << "    Converge Constant: " << ConvergeConstant << std::endl;
}

int main()
{
	tout.open("output.log");
	ShowParameters();
	auto ls = LoadLearningSet<Floating>(DataSetKind::Caltech101Silhouettes);
	auto start = std::chrono::system_clock::now();
	TestSdA(ls);
	auto end = std::chrono::system_clock::now();
	tout.s << "Elapsed time (seconds): " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << std::endl;
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

template <class TValue, class TNoise> bool PreTrain(HiddenLayerCollection<TValue>& hiddenLayers, unsigned int i, unsigned int neurons, TNoise noise, const LearningSet<TValue>& datasets, double& lastNeuronCost, unsigned int lastNeurons)
{
	//LossPredictor<double, 3> predictor;
	double currentTestCost;
	hiddenLayers.Set(i, neurons);
	tout.s << boost::format("Number of neurons of pre-training layer %d is %d") % i % neurons << std::endl;
	for (unsigned int epoch = 1; epoch <= PreTrainingEpochs; epoch++)
	{
		hiddenLayers[i].Train(datasets.TrainingData(), static_cast<TValue>(PreTrainingLearningRate), noise);
		currentTestCost = hiddenLayers[i].ComputeCost(datasets.ValidationData(), noise);
		//predictor.PushLoss(testCost);
		//if (epoch >= 4)
		//{
		//	if (epoch == 4)
		//	{
		//		predictor.Setup(epoch);
		//		tout.s << "Prediction Expression: " << predictor.GetExpression() << std::endl;
		//	}
		//	auto predictedTestCost = predictor(epoch);
		//	tout.s << boost::format("%2d %10lf %10lf") % epoch % testCost % predictedTestCost << std::endl;
		//	//if (epoch == 4 && !IsConverged(lastNeuronCost, predictor(PreTrainingEpochs)))
		//	//{
		//	//	lastNeuronCost = predictor(PreTrainingEpochs);
		//	//	return false;
		//	//}
		//}
		//else
		{
			tout.s << boost::format("%d %lf") % epoch % currentTestCost << std::endl;
		}
	}
	auto costRelativeError = abs((currentTestCost - lastNeuronCost) / (neurons - lastNeurons));
	tout.s << "Cost relative error: " << costRelativeError << std::endl;
	lastNeuronCost = currentTestCost;
	return costRelativeError <= ConvergeConstant;
}

template <class TValue> void FineTune(StackedDenoisingAutoEncoder<TValue>& sda, const LearningSet<TValue>& datasets)
{
	double bestTestScore = std::numeric_limits<double>::infinity();
	sda.SetLogisticRegressionLayer(datasets.ClassCount);
	tout.s << "Fine-Tuning..." << std::endl;
	for (unsigned int epoch = 1, patience = DefaultPatience; epoch <= FineTuningEpochs && epoch <= patience; epoch++)
	{
		sda.FineTune(datasets.TrainingData(), static_cast<TValue>(FineTuningLearningRate));
		auto thisTestScore = sda.ComputeErrorRates<Floating>(datasets.TestData());
		tout.s << boost::format("%d %lf%%") % epoch % (thisTestScore * 100.0) << std::endl;

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
	// seed: 89677
	std::random_device random;
	StackedDenoisingAutoEncoder<TValue> sda(random(), datasets.TrainingData().AllComponents());
	
	for (unsigned int i = 0; i < sizeof(DaNoises) / sizeof(DaNoises[0]); i++)
	{
		double lastNeuronCost = std::numeric_limits<double>::infinity();
		for (unsigned int neurons = MinNeurons; ; neurons *= NeuronIncease)
		{
			if (PreTrain(sda.HiddenLayers, i, neurons, DaNoises[i], datasets, lastNeuronCost, neurons / NeuronIncease))
				break;
		}
	}

	FineTune(sda, datasets);
}