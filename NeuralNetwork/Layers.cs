using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	/// <summary>ニューラルネットワークの層を表す抽象クラスです。</summary>
	public abstract class Layer
	{
		/// <summary>結合重みと活性化関数を使用して、<see cref="Layer"/> クラスの新しいインスタンスを初期化します。</summary>
		/// <param name="weight">前の層とこの層の間の結合重みを表す 2 次元配列を指定します。</param>
		/// <param name="activation">この層に適用する活性化関数を指定します。</param>
		protected Layer(double[,] weight, Action<Func<int, double>, double[]> activation)
		{
			Weight = weight;
			Bias = new double[weight.GetLength(0)];
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
		public Func<int, double> Learn(IReadOnlyList<double> input, IReadOnlyList<double> output, Func<int, double> upperInfo, double learningRate)
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

		/// <summary>隠れ層の既定の重みを計算します。</summary>
		/// <param name="rng">重みの初期化に使用する乱数生成器を指定します。</param>
		/// <param name="nIn">隠れ層の入力ユニット数を指定します。</param>
		/// <param name="nOut">隠れ層の出力ユニット数を指定します。</param>
		/// <param name="activation">隠れ層の活性化関数を指定します。</param>
		/// <returns>初期化された重みが格納された二次元配列。</returns>
		public static double[,] GetDefaultWeight(Random rng, int nIn, int nOut, ActivationFunction activation)
		{
			// `W` is initialized with `W_values` which is uniformely sampled from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden)) for tanh activation function
			// the output of uniform if converted using asarray to dtype theano.config.floatX so that the code is runable on GPU
			// Note : optimal initialization of weights is dependent on the activation function used (among other things).
			//        For example, results presented in [Xavier10] suggest that you should use 4 times larger initial weights for sigmoid compared to tanh
			//        We have no info for other function, so we use the same as tanh.
			var weight = new double[nOut, nIn];
			var buffer = new byte[4];
			for (int j = 0; j < nOut; j++)
			{
				for (int i = 0; i < nIn; i++)
				{
					rng.NextBytes(buffer);
					weight[j, i] = (2 * BitConverter.ToUInt32(buffer, 0) / (double)uint.MaxValue - 1) * Math.Sqrt(6.0 / (nIn + nOut));
					if (activation == ActivationFunction.Sigmoid)
						weight[j, i] *= 4;
				}
			}
			return weight;
		}
	}

	/// <summary>
	/// 多層パーセプトロンの典型的な隠れ層を表します。
	/// ユニット間は全結合されており、活性化関数を適用できます。
	/// </summary>
	public class HiddenLayer : Layer
	{
		/// <summary><see cref="HiddenLayer"/> クラスを乱数生成器と入出力の次元数、および活性化関数を使用して初期化します。</summary>
		/// <param name="rng">重みの初期化に使用される乱数生成器を指定します。</param>
		/// <param name="nIn">入力の次元数を指定します。</param>
		/// <param name="nOut">隠れ素子の数を指定します。</param>
		/// <param name="activation">隠れ層に適用される活性化関数を指定します。</param>
		public HiddenLayer(Random rng, int nIn, int nOut, ActivationFunction activation) : base(GetDefaultWeight(rng, nIn, nOut, activation), activation.Normal) { _differentiatedActivation = activation.Differentiated; }

		readonly Func<double, double> _differentiatedActivation;

		/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) の要素を計算します。</summary>
		/// <param name="output">この層からの出力を示すベクトルの要素を指定します。</param>
		/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
		/// <returns>勾配ベクトルの要素。</returns>
		protected override double GetDelta(double output, double upperInfo) { return upperInfo * _differentiatedActivation(output); }
	}

	/// <summary>
	/// 多クラスロジスティック回帰を行う出力層を表します。
	/// ロジスティック回帰は重み行列 W とバイアスベクトル b によって完全に記述されます。
	/// 分類はデータ点を超平面へ投影することによってなされます。
	/// </summary>
	public class LogisticRegressionLayer : Layer
	{
		/// <summary>ロジスティック回帰のパラメータを初期化します。</summary>
		/// <param name="nIn">入力素子の数 (データ点が存在する空間の次元) を指定します。</param>
		/// <param name="nOut">出力素子の数 (ラベルが存在する空間の次元) を指定します。</param>
		public LogisticRegressionLayer(int nIn, int nOut) : base(new double[nOut, nIn], ActivationFunction.SoftMax) { }

		/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) の要素を計算します。</summary>
		/// <param name="output">この層からの出力を示すベクトルの要素を指定します。</param>
		/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
		/// <returns>勾配ベクトルの要素。</returns>
		protected override double GetDelta(double output, double upperInfo) { return output - upperInfo; }

		/// <summary>指定された計算結果に基づいて、確率が最大となるクラスを推定します。</summary>
		/// <param name="computed">このロジスティック回帰の計算結果を指定します。</param>
		/// <returns>推定された確率最大のクラスのインデックス。</returns>
		public static int Predict(IEnumerable<double> computed) { return computed.Select((x, i) => new KeyValuePair<int, double>(i, x)).Aggregate((x, y) => x.Value > y.Value ? x : y).Key; }
	}
}
