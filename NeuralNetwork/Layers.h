#pragma once

#include "Matrix.h"
#include "Functions.h"
#include "LearningSet.h"

template <class T> class ReferableVector final
{
public:
	ReferableVector() : reference_target_(nullptr), target_() { }
	ReferableVector(const std::valarray<T>& reference) : reference_target_(&reference) { }
	ReferableVector(std::valarray<T>&& instance) : reference_target_(nullptr), target_(std::move(instance)) { }
	ReferableVector(ReferableVector&& right) : reference_target_(right.reference_target_), target_(std::move(right.target_)) { }

	ReferableVector& operator=(const std::valarray<T>& reference)
	{
		reference_target_ = &reference;
		target_.resize(0);
		return *this;
	}
	ReferableVector& operator=(std::valarray<T>&& instance)
	{
		reference_target_ = nullptr;
		target_ = std::move(instance);
		return *this;
	}
	ReferableVector& operator=(ReferableVector&& right)
	{
		reference_target_ = right.reference_target;
		target_ = std::move(right.target);
		return *this;
	}

	const std::valarray<T>& target() const { return reference_target_ ? *reference_target_ : target_; }
	operator const std::valarray<T>&() const { return target(); }
	const T& operator[](size_t index) const { return target()[index]; }

private:
	const std::valarray<T>* reference_target_;
	std::valarray<T> target_;
};

/// <summary>指定された層の学習を行い、下位層の学習に必要な情報を返します。</summary>
/// <param name="layer">学習を行う層を指定します。</param>
/// <param name="input"><paramref name="layer"/> への入力を示すベクトルを指定します。</param>
/// <param name="output"><paramref name="layer"/> からの出力を示すベクトルを指定します。</param>
/// <param name="upperInfo">上位層から得られた学習に必要な情報を指定します。<paramref name="layer"/> が出力層の場合、これは教師信号になります。</param>
/// <param name="learningRate">結合重みとバイアスをどれほど更新するかを示す値を指定します。</param>
/// <returns>下位層の学習に必要な情報。</returns>
template <class TLayer, class TUpperInfo, class TValue> std::valarray<TValue> LearnLayer(TLayer& layer, const std::valarray<TValue>& input, const std::valarray<TValue>& output, const TUpperInfo& upperInfo, TValue learningRate)
{
	std::valarray<TValue> lowerInfo(static_cast<TValue>(0), layer.Weight.Column());
	for (size_t i = 0; i < layer.Weight.Row(); i++)
	{
		auto deltaI = TLayer::GetDelta(output[i], upperInfo[i]);
		for (size_t j = 0; j < layer.Weight.Column(); j++)
		{
			lowerInfo[j] += layer.Weight(i, j) * deltaI;
			layer.Weight(i, j) -= learningRate * (deltaI * input[j]);
		}
		layer.Bias[i] -= learningRate * deltaI;
	}
	return std::move(lowerInfo);
}

template <class TMatrix, class TVector> class NeuronComputer final : private boost::noncopyable
{
public:
	NeuronComputer(const TMatrix& weight, const TVector& bias, const TVector& input) : weight(&weight), bias(&bias), input(&input) { }
	typename TVector::value_type operator[](size_t index) const
	{
		auto ret = (*bias)[index];
		for (size_t k = 0; k < input->size(); k++)
			ret += (*input)[k] * (*weight)(index, k);
		return ret;
	}
	size_t size() const { return weight->Row(); }

private:
	const TMatrix* weight;
	const TVector* bias;
	const TVector* input;
};

template <class TValue> class HiddenLayer;

/// <summary>隠れ層のコレクションに対する基本クラスを表します。</summary>
template <class TValue> class HiddenLayerCollectionBase
{
public:
	/// <summary>指定された範囲で一様な乱数を返します。</summary>
	/// <param name="min">乱数の包括的下限値を指定します。</param>
	/// <param name="max">乱数の排他的上限値を指定します。</param>
	/// <returns><paramref name="min"/> 以上 <paramref name="max"/> 未満の一様乱数。</returns>
	template <class T> T GenerateUniformRandomNumber(T min, T max) { return std::uniform_real_distribution<T>(min, max)(rng); }

	/// <summary>指定された層の入力ベクトルを計算します。層が指定されない場合、このメソッドは出力層の入力ベクトルを計算します。</summary>
	/// <param name="input">最初の隠れ層に与える入力を指定します。</param>
	/// <param name="stopLayer">入力ベクトルを計算する層を指定します。この引数は省略可能です。</param>
	/// <returns>指定された層の入力ベクトル。層が指定されなかった場合は出力層の入力ベクトルを返します。</returns>
	virtual ReferableVector<TValue> Compute(const std::valarray<TValue>& input, const HiddenLayer<TValue>* stopLayer) const = 0;

protected:
	HiddenLayerCollectionBase(std::mt19937::result_type rngSeed) : rng(rngSeed) { }
	virtual ~HiddenLayerCollectionBase() { }

private:
	std::mt19937 rng;
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
template <class TValue> class HiddenLayer final : private boost::noncopyable
{
public:
	/// <summary><see cref="HiddenLayer"/> クラスを入出力の次元数、活性化関数および下層を使用して初期化します。</summary>
	/// <param name="nIn">入力の次元数を指定します。</param>
	/// <param name="nOut">隠れ素子の数を指定します。</param>
	/// <param name="hiddenLayers">この隠れ層が所属している Stacked Denoising Auto-Encoder のすべての隠れ層を表すリストを指定します。</param>
	HiddenLayer(size_t nIn, size_t nOut, HiddenLayerCollectionBase<TValue>* hiddenLayers) : Weight(nOut, nIn), Bias(static_cast<TValue>(0), nOut), VisibleBias(static_cast<TValue>(0), nIn), hiddenLayers(hiddenLayers)
	{
		if (!hiddenLayers)
			throw std::invalid_argument("hiddenLayers must not be null pointer");
		auto randMax = nextafter(static_cast<TValue>(1), (std::numeric_limits<TValue>::max)());
		for (unsigned int j = 0; j < nOut; j++)
		{
			for (unsigned int i = 0; i < nIn; i++)
			{
				Weight(j, i) = (2 * hiddenLayers->GenerateUniformRandomNumber<TValue>(0, randMax) - 1) * sqrt(static_cast<TValue>(6.0) / (nIn + nOut));
				Weight(j, i) *= 4;
			}
		}
	}

	/// <summary>この層の結合重みを示します。</summary>
	Matrix<TValue> Weight;
	/// <summary>この層のバイアスを示します。</summary>
	std::valarray<TValue> Bias;
	/// <summary>この層から構成された Denoising Auto-Encoder の出力層のバイアスを示します。</summary>
	std::valarray<TValue> VisibleBias;

	/// <summary>この層の入力に対する出力を計算します。</summary>
	/// <param name="input">層に入力するベクトルを指定します。</param>
	/// <returns>この層の出力を示すベクトル。</returns>
	std::valarray<TValue> Compute(const std::valarray<TValue>& input) const { return std::move(ActivationFunction::LogisticSigmoid(NeuronComputer<Matrix<TValue>, std::valarray<TValue>>(Weight, Bias, input))); }

	/// <summary>この層から雑音除去自己符号化器を構成し、指定されたデータセットを使用して訓練した結果のコストを返します。</summary>
	/// <param name="dataset">訓練に使用するデータセットを指定します。</param>
	/// <param name="learningRate">学習率を指定します。</param>
	/// <param name="noise">構成された雑音除去自己符号化器の入力を生成する際のデータの欠損率を指定します。</param>
	/// <returns>構成された雑音除去自己符号化器の入力に対するコスト。</returns>
	template <class TNoise> TValue Train(const DataSet<TValue>& dataset, TValue learningRate, TNoise noise)
	{
		return ComputeCost(dataset, noise, [&](const std::valarray<TValue>& image, const std::valarray<TValue>& corrupted, const std::valarray<TValue>& latent, const std::valarray<TValue>& reconstructed)
		{
			std::valarray<TValue> delta(Weight.Row());

#pragma omp parallel for
			for (int i = 0; i < static_cast<int>(Bias.size()); i++)
			{
				delta[static_cast<size_t>(i)] = 0;
				for (size_t j = 0; j < Weight.Column(); j++)
					delta[static_cast<size_t>(i)] += (reconstructed[j] - image[j]) * Weight(static_cast<size_t>(i), j);
				delta[static_cast<size_t>(i)] *= ActivationFunction::LogisticSigmoidDifferentiated(latent[static_cast<size_t>(i)]);
				Bias[static_cast<size_t>(i)] -= learningRate * delta[static_cast<size_t>(i)];
			}

#pragma omp parallel for
			for (int j = 0; j < static_cast<int>(VisibleBias.size()); j++)
			{
				for (size_t i = 0; i < delta.size(); i++)
					Weight(i, static_cast<size_t>(j)) -= learningRate * ((reconstructed[static_cast<size_t>(j)] - image[static_cast<size_t>(j)]) * latent[i] + delta[i] * corrupted[static_cast<size_t>(j)]);
				VisibleBias[static_cast<size_t>(j)] -= learningRate * (reconstructed[static_cast<size_t>(j)] - image[static_cast<size_t>(j)]);
			}
		});
	}

	/// <summary>この層から雑音除去自己符号化器を構成し、指定されたデータセットのコストを計算します。</summary>
	/// <param name="dataset">コストを計算するデータセットを指定します。</param>
	/// <param name="noise">構成された雑音除去自己符号化器の入力を生成する際のデータの欠損率を指定します。</param>
	/// <returns>構成された雑音除去自己符号化器の入力に対するコスト。</returns>
	template <class TNoise> TValue ComputeCost(const DataSet<TValue>& dataset, TNoise noise) const { return ComputeCost(dataset, noise, [](const std::valarray<TValue>&, const std::valarray<TValue>&, const std::valarray<TValue>&, const std::valarray<TValue>&) { }); }

	/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) の要素を計算します。</summary>
	/// <param name="output">この層からの出力を示すベクトルの要素を指定します。</param>
	/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
	/// <returns>勾配ベクトルの要素。</returns>
	static TValue GetDelta(TValue output, TValue upperInfo) { return upperInfo * ActivationFunction::LogisticSigmoidDifferentiated(output); }

private:
	HiddenLayerCollectionBase<TValue>* const hiddenLayers;

	template <class T, class TNoise> TValue ComputeCost(const DataSet<TValue>& dataset, TNoise noise, T update) const
	{
		TValue cost = 0;
		for (size_t n = 0; n < dataset.Images().size(); n++)
		{
			auto image = hiddenLayers->Compute(dataset.Images()[n], this);
			std::valarray<TValue> corrupted(Weight.Column());
			for (size_t i = 0; i < Weight.Column(); i++)
				corrupted[i] = hiddenLayers->GenerateUniformRandomNumber<TNoise>(0, 1) < noise ? 0 : image[i];
			auto latent = ActivationFunction::LogisticSigmoid(NeuronComputer<Matrix<TValue>, std::valarray<TValue>>(Weight, Bias, corrupted));
			auto reconstructed = ActivationFunction::LogisticSigmoid(NeuronComputer<TransposedMatrixView<TValue>, std::valarray<TValue>>(TransposedMatrixView<TValue>::From(Weight), VisibleBias, latent));
			update(image, corrupted, latent, reconstructed);
			cost += CostFunction::BiClassCrossEntropy(image.target(), reconstructed);
		}
		return cost / dataset.Images().size();
	}
};

/// <summary>隠れ層のコレクションを表します。</summary>
template <class TValue> class HiddenLayerCollection final : public HiddenLayerCollectionBase<TValue>, private boost::noncopyable
{
public:
	/// <summary>乱数生成器のシード値と入力層のユニット数を指定して、<see cref="HiddenLayerCollection"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="rngSeed">隠れ層の計算に使用される乱数生成器のシード値を指定します。</param>
	/// <param name="nIn">入力層のユニット数を指定します。</param>
	HiddenLayerCollection(std::mt19937::result_type rngSeed, size_t nIn) : HiddenLayerCollectionBase(rngSeed), nextLayerInputUnits(nIn), frozen(false) { }

	/// <summary>指定された層の入力ベクトルを計算します。層が指定されない場合、このメソッドは出力層の入力ベクトルを計算します。</summary>
	/// <param name="input">最初の隠れ層に与える入力を指定します。</param>
	/// <param name="stopLayer">入力ベクトルを計算する層を指定します。この引数は省略可能です。</param>
	/// <returns>指定された層の入力ベクトル。層が指定されなかった場合は出力層の入力ベクトルを返します。</returns>
	ReferableVector<TValue> Compute(const std::valarray<TValue>& input, const HiddenLayer<TValue>* stopLayer) const
	{
		ReferableVector<TValue> result(input);
		for (size_t i = 0; i < items.size() && items[i].get() != stopLayer; i++)
			result = items[i]->Compute(result);
		return std::move(result);
	}

	/// <summary>指定されたインデックスの隠れ層のニューロン数を変更します。このメソッドは隠れ層を追加することもできます。</summary>
	/// <param name="index">ニューロン数を変更する隠れ層のインデックスを指定します。</param>
	/// <param name="neurons">指定された隠れ層の新しいニューロン数を指定します。</param>
	void Set(size_t index, size_t neurons)
	{
		if (frozen)
			throw std::domain_error("freezed collection cannot be changed");
		if (index > items.size())
			throw std::out_of_range("index less than or equal to Count()");
		if (index == items.size())
		{
			items.push_back(std::unique_ptr<HiddenLayer<TValue>>(new HiddenLayer<TValue>(nextLayerInputUnits, neurons, this)));
			nextLayerInputUnits = neurons;
			return;
		}
		items[index] = std::unique_ptr<HiddenLayer<TValue>>(new HiddenLayer<TValue>(items[index]->Weight.Column(), neurons, this));
		if (index < items.size() - 1)
			items[index + 1] = std::unique_ptr<HiddenLayer<TValue>>(new HiddenLayer<TValue>(neurons, items[index + 1]->Weight.Row(), this));
	}

	/// <summary>このコレクションを固定して変更不可能にします。</summary>
	void Freeze() { frozen = true; }

	/// <summary>このコレクション内の指定されたインデックスにある隠れ層への参照を取得します。</summary>
	/// <param name="index">隠れ層を取得するインデックスを指定します。</param>
	/// <returns>取得された隠れ層への参照。これは変更可能な参照です。</returns>
	HiddenLayer<TValue>& operator[](size_t index) { return *items[index]; }

	/// <summary>このコレクション内に含まれている隠れ層の個数を指定します。</summary>
	/// <returns>コレクションに含まれている隠れ層の個数。</returns>
	size_t Count() const { return items.size(); }

private:
	bool frozen;
	size_t nextLayerInputUnits;
	std::vector<std::unique_ptr<HiddenLayer<TValue>>> items;
};

/// <summary>
/// 多クラスロジスティック回帰を行う出力層を表します。
/// ロジスティック回帰は重み行列 W とバイアスベクトル b によって完全に記述されます。
/// 分類はデータ点を超平面へ投影することによってなされます。
/// </summary>
template <class TValue> class LogisticRegressionLayer final : private boost::noncopyable
{
public:
	/// <summary>ロジスティック回帰のパラメータを初期化します。</summary>
	/// <param name="nIn">入力素子の数 (データ点が存在する空間の次元) を指定します。</param>
	/// <param name="nOut">出力素子の数 (ラベルが存在する空間の次元) を指定します。</param>
	LogisticRegressionLayer(size_t nIn, size_t nOut) : Weight(nOut, nIn), Bias(static_cast<TValue>(0), nOut) { }

	/// <summary>この層の結合重みを示します。</summary>
	Matrix<TValue> Weight;
	/// <summary>この層のバイアスを示します。</summary>
	std::valarray<TValue> Bias;

	/// <summary>この層の入力に対する出力を計算します。</summary>
	/// <param name="input">層に入力するベクトルを指定します。</param>
	/// <returns>この層の出力を示すベクトル。</returns>
	std::valarray<TValue> Compute(const std::valarray<TValue>& input) const { return std::move(ActivationFunction::SoftMax(NeuronComputer<Matrix<TValue>, std::valarray<TValue>>(Weight, Bias, input))); }

	/// <summary>確率が最大となるクラスを推定します。</summary>
	/// <param name="input">層に入力するベクトルを指定します。</param>
	/// <returns>推定された確率最大のクラスのインデックス。</returns>
	unsigned int Predict(const std::valarray<TValue>& input) const
	{
		auto computed = Compute(input);
		unsigned int maxIndex = 0;
		for (unsigned int i = 1; i < Weight.Row(); i++)
		{
			if (computed[i] > computed[maxIndex])
				maxIndex = i;
		}
		return maxIndex;
	}

	/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) の要素を計算します。</summary>
	/// <param name="output">この層からの出力を示すベクトルの要素を指定します。</param>
	/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
	/// <returns>勾配ベクトルの要素。</returns>
	static TValue GetDelta(TValue output, TValue upperInfo) { return output - upperInfo; }
};
