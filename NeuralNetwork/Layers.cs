using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	/// <summary>ニューラルネットワークの層を表す抽象クラスです。</summary>
	public abstract class Layer
	{
		/// <summary>入力されるニューロン数、この層のニューロン数、活性化関数を使用して、<see cref="Layer"/> クラスの新しいインスタンスを初期化します。</summary>
		/// <param name="nIn">この層に入力される層のニューロン数を指定します。</param>
		/// <param name="nOut">この層のニューロン数を指定します。</param>
		/// <param name="activation">この層に適用する活性化関数を指定します。</param>
		protected Layer(int nIn, int nOut, Action<Func<int, double>, double[]> activation)
		{
			Weight = new double[nOut, nIn];
			Bias = new double[nOut];
			_activation = activation;
		}

		readonly Action<Func<int, double>, double[]> _activation;
		
		/// <summary>この層の結合重みを示します。</summary>
		public readonly double[,] Weight;
		
		/// <summary>この層のバイアスを示します。</summary>
		public readonly double[] Bias;

		/// <summary>この層の入力に対する出力を計算します。</summary>
		/// <param name="input">層に入力するベクトルを指定します。</param>
		/// <returns>この層の出力を示すベクトル。</returns>
		public double[] Compute(IEnumerable<double> input)
		{
			var output = new double[Weight.GetLength(0)];
			_activation(j => input.Select((x, k) => x * Weight[j, k]).Sum() + Bias[j], output);
			return output;
		}
		
		/// <summary>この層の学習を行い、下位層の学習に必要な情報を返します。</summary>
		/// <param name="input">この層への入力を示すベクトルを指定します。</param>
		/// <param name="output">この層からの出力を示すベクトルを指定します。</param>
		/// <param name="upperInfo">上位層から得られた学習に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
		/// <param name="learningRate">結合重みとバイアスをどれほど更新するかを示す値を指定します。</param>
		/// <returns>下位層の学習に必要な情報を示すデリゲート。</returns>
		public Func<int, double> Learn(double[] input, double[] output, Func<int, double> upperInfo, double learningRate)
		{
			double[,] lowerInfo = new double[Weight.GetLength(0), Weight.GetLength(1)];
			Parallel.For(0, Weight.GetLength(0), i =>
			{
				var deltaI = GetDelta(output[i], upperInfo(i));
				for (int j = 0; j < Weight.GetLength(1); j++)
				{
					lowerInfo[i, j] = Weight[i, j] * deltaI;
					Weight[i, j] -= learningRate * (deltaI * input[j]);
				}
				Bias[i] -= learningRate * deltaI;
			});
			return j => Enumerable.Range(0, lowerInfo.GetLength(0)).Sum(i => lowerInfo[i, j]);
		}

		/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) の要素を計算します。</summary>
		/// <param name="output">この層からの出力を示すベクトルの要素を指定します。</param>
		/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
		/// <returns>勾配ベクトルの要素。</returns>
		protected abstract double GetDelta(double output, double upperInfo);
	}

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
	///		\tilde{x} ~ q_D(\tilde{x}|x)                                     (1)
	///		y = s(W \tilde{x} + b)                                           (2)
	///		x = s(W' y  + b')                                                (3)
	///		L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log(1-z_k)]       (4)
	/// </remarks>
	public sealed class HiddenLayer : Layer
	{
		/// <summary><see cref="HiddenLayer"/> クラスを乱数生成器、入出力の次元数、活性化関数および下層を使用して初期化します。</summary>
		/// <param name="rng">重みの初期化に使用される乱数生成器を指定します。</param>
		/// <param name="nIn">入力の次元数を指定します。</param>
		/// <param name="nOut">隠れ素子の数を指定します。</param>
		/// <param name="activation">隠れ層に適用される活性化関数を指定します。</param>
		/// <param name="hiddenLayers">この隠れ層が所属している Stacked Denoising Auto-Encoder のすべての隠れ層を表すリストを指定します。</param>
		public HiddenLayer(MersenneTwister rng, int nIn, int nOut, ActivationFunction activation, HiddenLayerCollection hiddenLayers) : base(nIn, nOut, activation.Normal)
		{
			// `W` is initialized with `W_values` which is uniformely sampled from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden)) for tanh activation function
			// the output of uniform if converted using asarray to dtype theano.config.floatX so that the code is runable on GPU
			// Note : optimal initialization of weights is dependent on the activation function used (among other things).
			//        For example, results presented in [Xavier10] suggest that you should use 4 times larger initial weights for sigmoid compared to tanh
			//        We have no info for other function, so we use the same as tanh.
			for (int j = 0; j < nOut; j++)
			{
				for (int i = 0; i < nIn; i++)
				{
					Weight[j, i] = (2 * rng.NextDoubleFull() - 1) * Math.Sqrt(6.0 / (nIn + nOut));
					if (activation == ActivationFunction.Sigmoid)
						Weight[j, i] *= 4;
				}
			}
			_differentiatedActivation = activation.Differentiated;

			_rng = rng;
			_visibleBias = new double[Weight.GetLength(1)];
			_hiddenLayers = hiddenLayers;
		}

		readonly Func<double, double> _differentiatedActivation;
		// 以下、DAとして使用するための変数
		readonly HiddenLayerCollection _hiddenLayers;
		readonly MersenneTwister _rng;
		readonly double[] _visibleBias;

		/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) の要素を計算します。</summary>
		/// <param name="output">この層からの出力を示すベクトルの要素を指定します。</param>
		/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
		/// <returns>勾配ベクトルの要素。</returns>
		protected override double GetDelta(double output, double upperInfo) { return upperInfo * _differentiatedActivation(output); }

		/// <summary>この層から雑音除去自己符号化器を構成し、指定されたデータセットを使用して訓練した結果のコストを返します。</summary>
		/// <param name="dataset">訓練に使用するデータセットを指定します。</param>
		/// <param name="learningRate">学習率を指定します。</param>
		/// <param name="noise">構成された雑音除去自己符号化器の入力を生成する際のデータの欠損率を指定します。</param>
		/// <returns>構成された雑音除去自己符号化器の入力に対するコスト。</returns>
		public double Train(Pattern[] dataset, double learningRate, double noise)
		{
			var latent = new double[Bias.Length];
			var reconstructed = new double[_visibleBias.Length];
			var delta = new double[Weight.GetLength(0)];
			double cost = 0;
			for (int n = 0; n < dataset.Length; n++)
			{
				var image = _hiddenLayers.Compute(dataset[n].Image, this);
				var corrupted = image.Select(x => _rng.NextDouble() < noise ? 0 : x).ToArray();
				ActivationFunction.Sigmoid.Normal(j => corrupted.Select((y, i) => y * Weight[j, i]).Sum() + Bias[j], latent);
				ActivationFunction.Sigmoid.Normal(j => latent.Select((y, i) => y * Weight[i, j]).Sum() + _visibleBias[j], reconstructed);
				cost += ErrorFunction.BiClassCrossEntropy(image, reconstructed);
				Parallel.For(0, Weight.GetLength(0), i =>
				{
					delta[i] = 0;
					for (int j = 0; j < Weight.GetLength(1); j++)
						delta[i] += (reconstructed[j] - image[j]) * Weight[i, j];
					delta[i] *= ActivationFunction.Sigmoid.Differentiated(latent[i]);
					Bias[i] -= learningRate * delta[i];
				});
				Parallel.For(0, Weight.GetLength(1), j =>
				{
					for (int i = 0; i < Weight.GetLength(0); i++)
						Weight[i, j] -= learningRate * ((reconstructed[j] - image[j]) * latent[i] + delta[i] * corrupted[j]);
					_visibleBias[j] -= learningRate * (reconstructed[j] - image[j]);
				});
			}
			return cost / dataset.Length;
		}

		/// <summary>この層から雑音除去自己符号化器を構成し、指定されたデータセットのコストを計算します。</summary>
		/// <param name="dataset">コストを計算するデータセットを指定します。</param>
		/// <param name="noise">構成された雑音除去自己符号化器の入力を生成する際のデータの欠損率を指定します。</param>
		/// <returns>構成された雑音除去自己符号化器の入力に対するコスト。</returns>
		public double ComputeCost(Pattern[] dataset, double noise)
		{
			var latent = new double[Bias.Length];
			var reconstructed = new double[_visibleBias.Length];
			double cost = 0;
			for (int n = 0; n < dataset.Length; n++)
			{
				var image = _hiddenLayers.Compute(dataset[n].Image, this);
				var corrupted = image.Select(x => _rng.NextDouble() < noise ? 0 : x).ToArray();
				ActivationFunction.Sigmoid.Normal(j => corrupted.Select((y, i) => y * Weight[j, i]).Sum() + Bias[j], latent);
				ActivationFunction.Sigmoid.Normal(j => latent.Select((y, i) => y * Weight[i, j]).Sum() + _visibleBias[j], reconstructed);
				cost += ErrorFunction.BiClassCrossEntropy(image, reconstructed);
			}
			return cost / dataset.Length;
		}
	}

	public sealed class HiddenLayerCollection : IReadOnlyList<HiddenLayer>
	{
		public HiddenLayerCollection(MersenneTwister rng, int nIn)
		{
			_rng = rng;
			_nextLayerInputUnits = nIn;
		}

		readonly MersenneTwister _rng;
		int _nextLayerInputUnits;
		List<HiddenLayer> _items = new List<HiddenLayer>();

		public double[] Compute(double[] input, HiddenLayer stopLayer)
		{
			for (int i = 0; i < _items.Count && _items[i] != stopLayer; i++)
				input = _items[i].Compute(input);
			return input;
		}

		public void Set(int index, int neurons)
		{
			if (_nextLayerInputUnits < 0)
				throw new InvalidOperationException("固定されたコレクションの隠れ層を設定することはできません。");
			if (index > _items.Count)
				throw new ArgumentOutOfRangeException("index");
			if (index == _items.Count)
			{
				_items.Add(new HiddenLayer(_rng, _nextLayerInputUnits, neurons, ActivationFunction.Sigmoid, this));
				_nextLayerInputUnits = neurons;
				return;
			}
			_items[index] = new HiddenLayer(_rng, _items[index].Weight.GetLength(1), neurons, ActivationFunction.Sigmoid, this);
			if (index < _items.Count - 1)
				_items[index + 1] = new HiddenLayer(_rng, neurons, _items[index + 1].Weight.GetLength(0), ActivationFunction.Sigmoid, this);
		}

		public void Freeze() { _nextLayerInputUnits = -1; }

		public HiddenLayer this[int index] { get { return _items[index]; } }

		public int Count { get { return _items.Count; } }

		public IEnumerator<HiddenLayer> GetEnumerator() { return _items.GetEnumerator(); }

		System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() { return GetEnumerator(); }
	}

	/// <summary>
	/// 多クラスロジスティック回帰を行う出力層を表します。
	/// ロジスティック回帰は重み行列 W とバイアスベクトル b によって完全に記述されます。
	/// 分類はデータ点を超平面へ投影することによってなされます。
	/// </summary>
	public sealed class LogisticRegressionLayer : Layer
	{
		/// <summary>ロジスティック回帰のパラメータを初期化します。</summary>
		/// <param name="nIn">入力素子の数 (データ点が存在する空間の次元) を指定します。</param>
		/// <param name="nOut">出力素子の数 (ラベルが存在する空間の次元) を指定します。</param>
		public LogisticRegressionLayer(int nIn, int nOut) : base(nIn, nOut, ActivationFunction.SoftMax) { }

		/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) の要素を計算します。</summary>
		/// <param name="output">この層からの出力を示すベクトルの要素を指定します。</param>
		/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
		/// <returns>勾配ベクトルの要素。</returns>
		protected override double GetDelta(double output, double upperInfo) { return output - upperInfo; }

		/// <summary>確率が最大となるクラスを推定します。</summary>
		/// <param name="input">層に入力するベクトルを指定します。</param>
		/// <returns>推定された確率最大のクラスのインデックス。</returns>
		public int Predict(IEnumerable<double> input) { return Compute(input).Select((x, i) => new KeyValuePair<int, double>(i, x)).Aggregate((x, y) => x.Value > y.Value ? x : y).Key; }
	}
}
