﻿#include "StackedDenoisingAutoEncoder.h"
#include "ShiftRegister.h"

enum class DataSetKind
{
	MNIST,
	Cifar10,
	Caltech101Silhouettes,
	PR,
};
const char* DataSetNames[]
{
	"MNIST",
	"Cifar-10",
	"Caltech 101 Silhouettes",
	"Pattern Recognition Data Set"
};

template <class TValue> LearningSet<TValue> LoadLearningSet(DataSetKind kind);
template <class TValue> void TestSdA(const LearningSet<TValue>& datasets);

typedef double Floating;

// Pre-Training Parameters

const int PreTrainingEpochs = 15;
const double PreTrainingLearningRate = 0.001;

const std::vector<Floating> DaNoises
{
	static_cast<Floating>(0.1),
	static_cast<Floating>(0.2),
	static_cast<Floating>(0.3),
};

// Fine-Tuning Parameters

const unsigned int FineTuningEpochs = 1000;
const double FineTuningLearningRate = 0.01;
const unsigned int DefaultPatience = 10;
const double ImprovementThreshold = 1;//0.995;
const unsigned int PatienceIncrease = 2;

// Number of Neuron Automatic Decision Parameters

const unsigned int MinNeurons = 1;
const double NeuronIncease = 4.0 / 3.0;
const unsigned int CostCheckEpoch = 1;
const double ConvergeConstant = 0.1;

// Using Data Set
const DataSetKind UsingDataSet = DataSetKind::Caltech101Silhouettes;

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
	if (!DaNoises.empty())
	{
		tout.s << "Pre-Training: " << std::endl;
		tout.s << "    Epochs: " << PreTrainingEpochs << std::endl;
		tout.s << "    Learning Rate: " << PreTrainingLearningRate << std::endl;
		tout.s << "    Noise Rate: " << std::endl;
		for (size_t i = 0; i < DaNoises.size(); i++)
			tout.s << "        HL " << i << ": " << DaNoises[i] << std::endl;
	}
	tout.s << "Fine-Tuning: " << std::endl;
	tout.s << "    Max Epochs: " << FineTuningEpochs << std::endl;
	tout.s << "    Learning Rate: " << FineTuningLearningRate << std::endl;
	tout.s << "    Early Stopping Parameters: " << std::endl;
	tout.s << "        Default Patience: " << DefaultPatience << std::endl;
	tout.s << "        Improvement Threshold: " << ImprovementThreshold << std::endl;
	tout.s << "        Patience Increase: " << PatienceIncrease << std::endl;
	if (!DaNoises.empty())
	{
		tout.s << "Number of Neuron Automatic Decision Parameters: " << std::endl;
		//tout.s << "    Minimum Number of Neurons: " << MinNeurons << std::endl;
		//tout.s << "    Number of Neuron Increase: " << NeuronIncease << std::endl;
		tout.s << "    Converge Constant: " << ConvergeConstant << std::endl;
	}
}

int main()
{
	std::time_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	std::tm tm;
	localtime_s(&tm, &time);
	std::ostringstream sout;
	sout << "Outputs/";
	sout << DataSetNames[static_cast<size_t>(UsingDataSet)] << "/";
	_mkdir(sout.str().c_str());
	sout << std::setfill('0') << std::setw(4) << tm.tm_year + 1900 << "-"
		<< std::setfill('0') << std::setw(2) << tm.tm_mon + 1 << "-"
		<< std::setfill('0') << std::setw(2) << tm.tm_mday << " "
		<< std::setfill('0') << std::setw(2) << tm.tm_hour << "-"
		<< std::setfill('0') << std::setw(2) << tm.tm_min << ".log";
	tout.open(sout.str());
	ShowParameters();
	auto ls = LoadLearningSet<Floating>(UsingDataSet);
	auto start = std::chrono::system_clock::now();
	TestSdA(ls);
	auto end = std::chrono::system_clock::now();
	tout.s << "Elapsed Time (Seconds): " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << std::endl;
	return 0;
}

template <class TValue> LearningSet<TValue> LoadLearningSet(DataSetKind kind)
{
	if (kind == DataSetKind::MNIST)
	{
		auto ls = MnistLoader<TValue>().Load("MNIST");
		LearningSet<TValue> newLs;
		newLs.ClassCount = ls.ClassCount;
		newLs.TrainingData().From(std::move(ls.TrainingData()), 0, 5000);
		newLs.ValidationData().From(std::move(ls.TrainingData()), 50000, 1000);
		newLs.TestData().From(std::move(ls.TestData()), 0, 1000);
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

template <class TValue> void TestSdA(const LearningSet<TValue>& datasets)
{
	for (unsigned int neuronIncrease = 25; neuronIncrease <= 1000; neuronIncrease += 25)
	{
		tout.s << "Current Number of Neuron Increase: " << neuronIncrease << std::endl;

		// seed: 89677
		std::random_device random;
		StackedDenoisingAutoEncoder<TValue> sda(random(), datasets.TrainingData().AllComponents());

		for (unsigned int i = 0; i < DaNoises.size(); i++)
		{
			auto lastNeuronCost = std::numeric_limits<TValue>::infinity();
			for (unsigned int neurons = neuronIncrease, prevNeurons = 0; ; )
			{
				sda.HiddenLayers.Set(i, neurons);
				tout.s << "Number of Neurons of Hidden Layer " << i << ": " << neurons << std::endl;
				auto currentTestCost = static_cast<TValue>(0);
				for (unsigned int epoch = 1; epoch <= CostCheckEpoch; epoch++)
				{
					sda.HiddenLayers[i].Train(datasets.TrainingData(), static_cast<TValue>(PreTrainingLearningRate), DaNoises[i]);
					currentTestCost = sda.HiddenLayers[i].ComputeCost(datasets.ValidationData(), DaNoises[i]);
					tout.s << epoch << " " << currentTestCost << std::endl;
				}
				auto costDifference = (currentTestCost - lastNeuronCost) / (neurons - prevNeurons);
				tout.s << "Cost Difference per Neuron: " << costDifference << std::endl;
				if (abs(costDifference) <= ConvergeConstant)
					break;
				auto newNeurons = neurons + neuronIncrease;
				prevNeurons = neurons;
				neurons = newNeurons;
				lastNeuronCost = currentTestCost;
			}
			for (unsigned int epoch = CostCheckEpoch + 1; epoch <= PreTrainingEpochs; epoch++)
			{
				sda.HiddenLayers[i].Train(datasets.TrainingData(), static_cast<TValue>(PreTrainingLearningRate), DaNoises[i]);
				auto currentTestCost = sda.HiddenLayers[i].ComputeCost(datasets.ValidationData(), DaNoises[i]);
				tout.s << epoch << " " << currentTestCost << std::endl;
			}
		}

		if (!DaNoises.empty())
		{
			tout.s << "Decided Number of Neurons: " << std::endl;
			for (unsigned int i = 0; i < sda.HiddenLayers.Count(); i++)
				tout.s << "    Number of Neurons of Hidden Layer " << i << ": " << sda.HiddenLayers[i].Weight.Row() << std::endl;
		}

		auto bestTestScore = std::numeric_limits<Floating>::infinity();
		sda.SetLogisticRegressionLayer(datasets.ClassCount);
		tout.s << "Fine-Tuning..." << std::endl;
		for (unsigned int epoch = 1, patience = DefaultPatience; epoch <= FineTuningEpochs && epoch <= patience; epoch++)
		{
			sda.FineTune(datasets.TrainingData(), static_cast<TValue>(FineTuningLearningRate));
			auto thisTestScore = sda.ComputeErrorRates<Floating>(datasets.TestData());
			tout.s << epoch << " " << thisTestScore * 100.0 << "% Patience: " << patience << std::endl;

			if (thisTestScore < bestTestScore)
			{
				tout.s << epoch << " Training Score: " << sda.ComputeErrorRates<Floating>(datasets.TrainingData()) * 100.0 << "%" << std::endl;
				if (thisTestScore < bestTestScore * ImprovementThreshold)
					patience = std::max(patience, epoch * PatienceIncrease);
				bestTestScore = thisTestScore;
			}
		}
		tout.s << "Best Test Score of Fine-Tuning: " << bestTestScore * 100.0 << "%" << std::endl;
	}
}