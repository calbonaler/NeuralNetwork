#include "Functions.h"
#include "LearningSet.h"
#include <vector>
#include <memory>
#include <random>

/// <summary>ニューラルネットワークの層を表す抽象クラスです。</summary>
class Layer
{
public:
	FORCE_UNCOPYABLE(Layer);

	/// <summary>この層を破棄します。</summary>
	virtual ~Layer();

	/// <summary>この層の結合重みを示します。</summary>
	double** const Weight;
	/// <summary>この層のバイアスを示します。</summary>
	double* const Bias;
	/// <summary>この層の入力ユニット数を示します。</summary>
	unsigned int const nIn;
	/// <summary>この層の出力ユニット数を示します。</summary>
	unsigned int const nOut;

	/// <summary>この層の入力に対する出力を計算します。</summary>
	/// <param name="input">層に入力するベクトルを指定します。</param>
	/// <returns>この層の出力を示すベクトル。</returns>
	std::unique_ptr<double[]> Compute(const double* input) const;

	/// <summary>この層の学習を行い、下位層の学習に必要な情報を返します。</summary>
	/// <param name="input">この層への入力を示すベクトルを指定します。</param>
	/// <param name="output">この層からの出力を示すベクトルを指定します。</param>
	/// <param name="upperInfo">上位層から得られた学習に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
	/// <param name="learningRate">結合重みとバイアスをどれほど更新するかを示す値を指定します。</param>
	/// <returns>下位層の学習に必要な情報。</returns>
	std::unique_ptr<double[]> Learn(const double* input, const double* output, Indexer upperInfo, double learningRate);

protected:
	/// <summary>入力されるニューロン数、この層のニューロン数、活性化関数を使用して、<see cref="Layer"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="nIn">この層に入力される層のニューロン数を指定します。</param>
	/// <param name="nOut">この層のニューロン数を指定します。</param>
	/// <param name="activation">この層に適用する活性化関数を指定します。</param>
	Layer(unsigned int nIn, unsigned int nOut, const ActivationFunction::NormalForm& activation);

	/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) の要素を計算します。</summary>
	/// <param name="output">この層からの出力を示すベクトルの要素を指定します。</param>
	/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
	/// <returns>勾配ベクトルの要素。</returns>
	virtual double GetDelta(double output, double upperInfo) const = 0;

private:
	const ActivationFunction::NormalForm activation;
};

class HiddenLayerCollection;

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
	FORCE_UNCOPYABLE(HiddenLayer);

	/// <summary><see cref="HiddenLayer"/> クラスを入出力の次元数、活性化関数および下層を使用して初期化します。</summary>
	/// <param name="nIn">入力の次元数を指定します。</param>
	/// <param name="nOut">隠れ素子の数を指定します。</param>
	/// <param name="activation">隠れ層に適用される活性化関数を指定します。</param>
	/// <param name="hiddenLayers">この隠れ層が所属している Stacked Denoising Auto-Encoder のすべての隠れ層を表すリストを指定します。</param>
	HiddenLayer(unsigned int nIn, unsigned int nOut, const ActivationFunction* activation, HiddenLayerCollection* hiddenLayers);

	/// <summary>この層を破棄します。</summary>
	~HiddenLayer();

	/// <summary>この層から雑音除去自己符号化器を構成し、指定されたデータセットを使用して訓練した結果のコストを返します。</summary>
	/// <param name="dataset">訓練に使用するデータセットを指定します。</param>
	/// <param name="learningRate">学習率を指定します。</param>
	/// <param name="noise">構成された雑音除去自己符号化器の入力を生成する際のデータの欠損率を指定します。</param>
	/// <returns>構成された雑音除去自己符号化器の入力に対するコスト。</returns>
	double Train(const DataSet& dataset, double learningRate, double noise);

	/// <summary>この層から雑音除去自己符号化器を構成し、指定されたデータセットのコストを計算します。</summary>
	/// <param name="dataset">コストを計算するデータセットを指定します。</param>
	/// <param name="noise">構成された雑音除去自己符号化器の入力を生成する際のデータの欠損率を指定します。</param>
	/// <returns>構成された雑音除去自己符号化器の入力に対するコスト。</returns>
	double ComputeCost(const DataSet& dataset, double noise) const;

protected:
	/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) の要素を計算します。</summary>
	/// <param name="output">この層からの出力を示すベクトルの要素を指定します。</param>
	/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
	/// <returns>勾配ベクトルの要素。</returns>
	double GetDelta(double output, double upperInfo) const { return upperInfo * differentiatedActivation(output); }

private:
	const ActivationFunction::DifferentiatedForm differentiatedActivation;
	HiddenLayerCollection* const hiddenLayers;
	double* const visibleBias;
};

/// <summary>隠れ層のコレクションを表します。</summary>
class HiddenLayerCollection
{
public:
	FORCE_UNCOPYABLE(HiddenLayerCollection);

	/// <summary>乱数生成器のシード値と入力層のユニット数を指定して、<see cref="HiddenLayerCollection"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="rngSeed">隠れ層の計算に使用される乱数生成器のシード値を指定します。</param>
	/// <param name="nIn">入力層のユニット数を指定します。</param>
	HiddenLayerCollection(std::mt19937::result_type rngSeed, unsigned int nIn) : RandomNumberGenerator(rngSeed), nextLayerInputUnits(nIn), frozen(false) { }

	/// <summary>隠れ層の計算に使用される乱数生成器を示します。</summary>
	std::mt19937 RandomNumberGenerator;

	/// <summary>指定された層の入力ベクトルを計算します。層が指定されない場合、このメソッドは出力層の入力ベクトルを計算します。</summary>
	/// <param name="input">最初の隠れ層に与える入力を指定します。</param>
	/// <param name="stopLayer">入力ベクトルを計算する層を指定します。この引数は省略可能です。</param>
	/// <returns>指定された層の入力ベクトル。層が指定されなかった場合は出力層の入力ベクトルを返します。</returns>
	unique_or_raw_array<double> Compute(const double* input, const HiddenLayer* stopLayer) const;

	/// <summary>指定されたインデックスの隠れ層のニューロン数を変更します。このメソッドは隠れ層を追加することもできます。</summary>
	/// <param name="index">ニューロン数を変更する隠れ層のインデックスを指定します。</param>
	/// <param name="neurons">指定された隠れ層の新しいニューロン数を指定します。</param>
	void Set(unsigned int index, unsigned int neurons);

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
};

/// <summary>
/// 多クラスロジスティック回帰を行う出力層を表します。
/// ロジスティック回帰は重み行列 W とバイアスベクトル b によって完全に記述されます。
/// 分類はデータ点を超平面へ投影することによってなされます。
/// </summary>
class LogisticRegressionLayer : public Layer
{
public:
	FORCE_UNCOPYABLE(LogisticRegressionLayer);

	/// <summary>ロジスティック回帰のパラメータを初期化します。</summary>
	/// <param name="nIn">入力素子の数 (データ点が存在する空間の次元) を指定します。</param>
	/// <param name="nOut">出力素子の数 (ラベルが存在する空間の次元) を指定します。</param>
	LogisticRegressionLayer(unsigned int nIn, unsigned int nOut) : Layer(nIn, nOut, ActivationFunction::SoftMax) { }

	/// <summary>確率が最大となるクラスを推定します。</summary>
	/// <param name="input">層に入力するベクトルを指定します。</param>
	/// <returns>推定された確率最大のクラスのインデックス。</returns>
	unsigned int Predict(const double* input) const;

protected:
	/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) の要素を計算します。</summary>
	/// <param name="output">この層からの出力を示すベクトルの要素を指定します。</param>
	/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
	/// <returns>勾配ベクトルの要素。</returns>
	double GetDelta(double output, double upperInfo) const { return output - upperInfo; }
};
