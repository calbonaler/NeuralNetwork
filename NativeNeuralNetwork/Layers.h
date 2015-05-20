#include "Functions.h"
#include "LearningSet.h"

#pragma warning (push)
// Because IHiddenLayerCollection is Interface (or Mix-in), destructor is not needed.
#pragma warning (disable : 4265)

/// <summary>ニューラルネットワークの層を表す抽象クラスです。</summary>
class Layer : private boost::noncopyable
{
public:
	/// <summary>この層を破棄します。</summary>
	virtual ~Layer()
	{
		delete[] Weight[0];
		delete[] Weight;
	}

	/// <summary>この層の結合重みを示します。</summary>
	ValueType** const Weight;
	/// <summary>この層のバイアスを示します。</summary>
	VectorType Bias;
	/// <summary>この層の入力ユニット数を示します。</summary>
	unsigned int const nIn;
	/// <summary>この層の出力ユニット数を示します。</summary>
	unsigned int const nOut;

	/// <summary>この層の入力に対する出力を計算します。</summary>
	/// <param name="input">層に入力するベクトルを指定します。</param>
	/// <returns>この層の出力を示すベクトル。</returns>
	VectorType Compute(const VectorType& input) const
	{
		VectorType output(nOut);
		Activation(Indexer(Weight, Bias, input), output);
		return std::move(output);
	}

	/// <summary>この層の学習を行い、下位層の学習に必要な情報を返します。</summary>
	/// <param name="input">この層への入力を示すベクトルを指定します。</param>
	/// <param name="output">この層からの出力を示すベクトルを指定します。</param>
	/// <param name="upperInfo">上位層から得られた学習に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
	/// <param name="learningRate">結合重みとバイアスをどれほど更新するかを示す値を指定します。</param>
	/// <returns>下位層の学習に必要な情報。</returns>
	VectorType Learn(const VectorType& input, const VectorType& output, const std::function<ValueType(unsigned int)>& upperInfo, ValueType learningRate)
	{
		VectorType lowerInfo(static_cast<ValueType>(0), nIn);
		for (unsigned int i = 0; i < nOut; i++)
		{
			ValueType deltaI = GetDelta(output[i], upperInfo(i));
			for (unsigned int j = 0; j < nIn; j++)
			{
				lowerInfo[j] += Weight[i][j] * deltaI;
				Weight[i][j] -= learningRate * (deltaI * input[j]);
			}
			Bias[i] -= learningRate * deltaI;
		}
		return std::move(lowerInfo);
	}

protected:
	/// <summary>入力されるニューロン数、この層のニューロン数、活性化関数を使用して、<see cref="Layer"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="nIn">この層に入力される層のニューロン数を指定します。</param>
	/// <param name="nOut">この層のニューロン数を指定します。</param>
	/// <param name="activation">この層に適用する活性化関数を指定します。</param>
	Layer(unsigned int nIn, unsigned int nOut, const ActivationFunction::NormalForm& activation) : nIn(nIn), nOut(nOut), Weight(new ValueType*[nOut]), Bias(static_cast<ValueType>(0), nOut), Activation(activation)
	{
		if (nIn <= 0 || nOut <= 0)
			throw std::invalid_argument("nIn and nOut must not be 0");
		Weight[0] = new ValueType[nIn * nOut]();
		for (unsigned int i = 1; i < nOut; i++)
			Weight[i] = Weight[i - 1] + nIn;
	}

	/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) の要素を計算します。</summary>
	/// <param name="output">この層からの出力を示すベクトルの要素を指定します。</param>
	/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
	/// <returns>勾配ベクトルの要素。</returns>
	virtual ValueType GetDelta(ValueType output, ValueType upperInfo) const = 0;

	const ActivationFunction::NormalForm Activation;
};

class HiddenLayer;

/// <summary>隠れ層のコレクションに対するインターフェイスを表します。</summary>
class IHiddenLayerCollection
{
public:
	/// <summary>隠れ層の計算に使用される乱数生成器を取得します。</summary>
	virtual std::mt19937& GetRandomNumberGenerator() = 0;

	/// <summary>指定された層の入力ベクトルを計算します。層が指定されない場合、このメソッドは出力層の入力ベクトルを計算します。</summary>
	/// <param name="input">最初の隠れ層に与える入力を指定します。</param>
	/// <param name="stopLayer">入力ベクトルを計算する層を指定します。この引数は省略可能です。</param>
	/// <returns>指定された層の入力ベクトル。層が指定されなかった場合は出力層の入力ベクトルを返します。</returns>
	virtual ReferableVector Compute(const VectorType& input, const HiddenLayer* stopLayer) const = 0;
};

/// <summary>
/// 多層パーセプトロンの典型的な隠れ層を表します。
/// ユニット間は全結合されており、活性化関数を適用できます。
/// これは雑音除去自己符号化器を内部に含みます。
/// </summary>
/// <remarks>
/// 雑音除去自己符号化器は破壊された入力から、入力をまず隠れ空間に投影しその後入力空間に再投影することで、もとの入力の復元を試みます。
/// 詳しい情報が必要な場合は Vincent et al., 2008 を参照してください。
/// 
/// x を入力とすると、式(1)は確率的写像 q_D の手段によって部分的に破壊された入力を計算します。
/// 式(2)は入力から隠れ空間に対する投影を計算します。
/// 式(3)は入力の再構築を行い、式(4)が再構築誤差を計算します。
///		¥tilde{x} ‾ q_D(¥tilde{x}|x)                                     (1)
///		y = s(W ¥tilde{x} + b)                                           (2)
///		x = s(W' y  + b')                                                (3)
///		L(x,z) = -sum_{k=1}^d [x_k ¥log z_k + (1-x_k) ¥log(1-z_k)]       (4)
/// </remarks>
class HiddenLayer : public Layer
{
public:
	/// <summary><see cref="HiddenLayer"/> クラスを入出力の次元数、活性化関数および下層を使用して初期化します。</summary>
	/// <param name="nIn">入力の次元数を指定します。</param>
	/// <param name="nOut">隠れ素子の数を指定します。</param>
	/// <param name="activation">隠れ層に適用される活性化関数を指定します。</param>
	/// <param name="hiddenLayers">この隠れ層が所属している Stacked Denoising Auto-Encoder のすべての隠れ層を表すリストを指定します。</param>
	HiddenLayer(unsigned int nIn, unsigned int nOut, const ActivationFunction* activation, IHiddenLayerCollection* hiddenLayers) : Layer(nIn, nOut, activation->Normal), differentiatedActivation(activation->Differentiated), visibleBias(static_cast<ValueType>(0), nIn), hiddenLayers(hiddenLayers)
	{
		if (!activation)
			throw std::invalid_argument("activation must not be null pointer");
		if (!hiddenLayers)
			throw std::invalid_argument("hiddenLayers must not be null pointer");
		std::uniform_real_distribution<ValueType> dist(0, nextafter(static_cast<ValueType>(1.0), (std::numeric_limits<ValueType>::max)()));
		for (unsigned int j = 0; j < nOut; j++)
		{
			for (unsigned int i = 0; i < nIn; i++)
			{
				Weight[j][i] = (2 * dist(hiddenLayers->GetRandomNumberGenerator()) - 1) * sqrt(static_cast<ValueType>(6.0) / (nIn + nOut));
				if (activation == ActivationFunction::LogisticSigmoid())
					Weight[j][i] *= 4;
			}
		}
	}

	/// <summary>この層から雑音除去自己符号化器を構成し、指定されたデータセットを使用して訓練した結果のコストを返します。</summary>
	/// <param name="dataset">訓練に使用するデータセットを指定します。</param>
	/// <param name="learningRate">学習率を指定します。</param>
	/// <param name="noise">構成された雑音除去自己符号化器の入力を生成する際のデータの欠損率を指定します。</param>
	/// <returns>構成された雑音除去自己符号化器の入力に対するコスト。</returns>
	ValueType Train(const DataSet& dataset, ValueType learningRate, Floating noise)
	{
		VectorType delta(nOut);
		return ComputeCost(dataset, noise, [&](const VectorType& image, const VectorType& corrupted, const VectorType& latent, const VectorType& reconstructed)
		{
#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(nOut); i++)
			{
				delta[static_cast<unsigned int>(i)] = 0;
				for (unsigned int j = 0; j < nIn; j++)
					delta[static_cast<unsigned int>(i)] += (reconstructed[j] - image[j]) * Weight[static_cast<unsigned int>(i)][j];
				delta[static_cast<unsigned int>(i)] *= differentiatedActivation(latent[static_cast<unsigned int>(i)]);
				Bias[static_cast<unsigned int>(i)] -= learningRate * delta[static_cast<unsigned int>(i)];
			};
#pragma omp parallel for
			for (int j = 0; j < static_cast<int>(nIn); j++)
			{
				for (unsigned int i = 0; i < nOut; i++)
					Weight[i][static_cast<unsigned int>(j)] -= learningRate * ((reconstructed[static_cast<unsigned int>(j)] - image[static_cast<unsigned int>(j)]) * latent[i] + delta[i] * corrupted[static_cast<unsigned int>(j)]);
				visibleBias[static_cast<unsigned int>(j)] -= learningRate * (reconstructed[static_cast<unsigned int>(j)] - image[static_cast<unsigned int>(j)]);
			};
		});
	}

	/// <summary>この層から雑音除去自己符号化器を構成し、指定されたデータセットのコストを計算します。</summary>
	/// <param name="dataset">コストを計算するデータセットを指定します。</param>
	/// <param name="noise">構成された雑音除去自己符号化器の入力を生成する際のデータの欠損率を指定します。</param>
	/// <returns>構成された雑音除去自己符号化器の入力に対するコスト。</returns>
	ValueType ComputeCost(const DataSet& dataset, Floating noise) const { return ComputeCost(dataset, noise, [](const VectorType&, const VectorType&, const VectorType&, const VectorType&) { }); }

protected:
	/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) の要素を計算します。</summary>
	/// <param name="output">この層からの出力を示すベクトルの要素を指定します。</param>
	/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
	/// <returns>勾配ベクトルの要素。</returns>
	ValueType GetDelta(ValueType output, ValueType upperInfo) const { return upperInfo * differentiatedActivation(output); }

private:
	const ActivationFunction::DifferentiatedForm differentiatedActivation;
	IHiddenLayerCollection* const hiddenLayers;
	VectorType visibleBias;

	ValueType ComputeCost(const DataSet& dataset, Floating noise, const std::function<void(const VectorType&, const VectorType&, const VectorType&, const VectorType&)>& update) const
	{
		std::uniform_real_distribution<Floating> dist(0, 1);
		VectorType corrupted(nIn);
		VectorType latent(nOut);
		VectorType reconstructed(nIn);
		ValueType cost = 0;
		for (unsigned int n = 0; n < dataset.Count(); n++)
		{
			auto image = hiddenLayers->Compute(dataset.Images()[n], this);
			for (unsigned int i = 0; i < nIn; i++)
				corrupted[i] = dist(hiddenLayers->GetRandomNumberGenerator()) < noise ? 0 : image[i];
			Activation(Indexer(Weight, Bias, corrupted), latent);
			Activation(Indexer(Weight, visibleBias, latent, true), reconstructed);
			// This cost function may not match to "Soft Plus" activation function.
			// But I cannot figure out substitute one...
			cost += CostFunction::LeastSquaresMethod(image, reconstructed);
			update(image, corrupted, latent, reconstructed);
		}
		return cost / dataset.Count();
	}
};

/// <summary>隠れ層のコレクションを表します。</summary>
class HiddenLayerCollection : public IHiddenLayerCollection, private boost::noncopyable
{
public:
	/// <summary>乱数生成器のシード値と入力層のユニット数を指定して、<see cref="HiddenLayerCollection"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="rngSeed">隠れ層の計算に使用される乱数生成器のシード値を指定します。</param>
	/// <param name="nIn">入力層のユニット数を指定します。</param>
	HiddenLayerCollection(std::mt19937::result_type rngSeed, unsigned int nIn) : rng(rngSeed), nextLayerInputUnits(nIn), frozen(false) { }

	/// <summary>隠れ層の計算に使用される乱数生成器を取得します。</summary>
	std::mt19937& GetRandomNumberGenerator() { return rng; }

	/// <summary>指定された層の入力ベクトルを計算します。層が指定されない場合、このメソッドは出力層の入力ベクトルを計算します。</summary>
	/// <param name="input">最初の隠れ層に与える入力を指定します。</param>
	/// <param name="stopLayer">入力ベクトルを計算する層を指定します。この引数は省略可能です。</param>
	/// <returns>指定された層の入力ベクトル。層が指定されなかった場合は出力層の入力ベクトルを返します。</returns>
	ReferableVector Compute(const VectorType& input, const HiddenLayer* stopLayer) const
	{
		ReferableVector result(input);
		for (unsigned int i = 0; i < items.size() && items[i].get() != stopLayer; i++)
			result = items[i]->Compute(result);
		return std::move(result);
	}

	/// <summary>指定されたインデックスの隠れ層のニューロン数を変更します。このメソッドは隠れ層を追加することもできます。</summary>
	/// <param name="index">ニューロン数を変更する隠れ層のインデックスを指定します。</param>
	/// <param name="neurons">指定された隠れ層の新しいニューロン数を指定します。</param>
	void Set(size_t index, unsigned int neurons)
	{
		if (frozen)
			throw std::domain_error("freezed collection cannot be changed");
		if (index > items.size())
			throw std::out_of_range("index less than or equal to Count()");
		if (index == items.size())
		{
			items.push_back(std::unique_ptr<HiddenLayer>(new HiddenLayer(nextLayerInputUnits, neurons, ActivationFunction::LogisticSigmoid(), this)));
			nextLayerInputUnits = neurons;
			return;
		}
		items[index] = std::unique_ptr<HiddenLayer>(new HiddenLayer(items[index]->nIn, neurons, ActivationFunction::LogisticSigmoid(), this));
		if (index < items.size() - 1)
			items[index + 1] = std::unique_ptr<HiddenLayer>(new HiddenLayer(neurons, items[index + 1]->nOut, ActivationFunction::LogisticSigmoid(), this));
	}

	/// <summary>このコレクションを固定して変更不可能にします。</summary>
	void Freeze() { frozen = true; }

	/// <summary>このコレクション内の指定されたインデックスにある隠れ層への参照を取得します。</summary>
	/// <param name="index">隠れ層を取得するインデックスを指定します。</param>
	/// <returns>取得された隠れ層への参照。これは変更可能な参照です。</returns>
	HiddenLayer& operator[](size_t index) { return *items[index]; }

	/// <summary>このコレクション内に含まれている隠れ層の個数を指定します。</summary>
	/// <returns>コレクションに含まれている隠れ層の個数。</returns>
	size_t Count() const { return items.size(); }

private:
	bool frozen;
	unsigned int nextLayerInputUnits;
	std::vector<std::unique_ptr<HiddenLayer>> items;
	std::mt19937 rng;
};

/// <summary>
/// 多クラスロジスティック回帰を行う出力層を表します。
/// ロジスティック回帰は重み行列 W とバイアスベクトル b によって完全に記述されます。
/// 分類はデータ点を超平面へ投影することによってなされます。
/// </summary>
class LogisticRegressionLayer : public Layer
{
public:
	/// <summary>ロジスティック回帰のパラメータを初期化します。</summary>
	/// <param name="nIn">入力素子の数 (データ点が存在する空間の次元) を指定します。</param>
	/// <param name="nOut">出力素子の数 (ラベルが存在する空間の次元) を指定します。</param>
	LogisticRegressionLayer(unsigned int nIn, unsigned int nOut) : Layer(nIn, nOut, ActivationFunction::SoftMax) { }

	/// <summary>確率が最大となるクラスを推定します。</summary>
	/// <param name="input">層に入力するベクトルを指定します。</param>
	/// <returns>推定された確率最大のクラスのインデックス。</returns>
	unsigned int Predict(const VectorType& input) const
	{
		auto computed = Compute(input);
		unsigned int maxIndex = 0;
		for (unsigned int i = 1; i < nOut; i++)
		{
			if (computed[i] > computed[maxIndex])
				maxIndex = i;
		}
		return maxIndex;
	}

protected:
	/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) の要素を計算します。</summary>
	/// <param name="output">この層からの出力を示すベクトルの要素を指定します。</param>
	/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
	/// <returns>勾配ベクトルの要素。</returns>
	ValueType GetDelta(ValueType output, ValueType upperInfo) const { return output - upperInfo; }
};

#pragma warning (pop)