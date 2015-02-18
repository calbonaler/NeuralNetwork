﻿using System;
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
			Activation = activation;
		}

		/// <summary>この層の活性化関数を示します。</summary>
		protected readonly Action<Func<int, double>, double[]> Activation;
		
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
			Activation(j => input.Select((x, k) => x * Weight[j, k]).Sum() + Bias[j], output);
			return output;
		}
		
		/// <summary>この層の結合重みとバイアスに対するニューラルネットワークのコストの勾配ベクトルを計算し、下位層の勾配計算に必要な情報を返します。</summary>
		/// <param name="input">この層への入力を示すベクトルを指定します。</param>
		/// <param name="output">この層からの出力を示すベクトルを指定します。</param>
		/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
		/// <param name="learningRate">結合重みとバイアスをどれほど更新するかを示す値を指定します。</param>
		/// <param name="gradients">計算された勾配が格納される <see cref="ParameterGradients"/> インターフェイスを指定します。</param>
		/// <returns>下位層の勾配計算に必要な情報を示すデリゲート。</returns>
		public Func<int, double> GetParameterGradients(IReadOnlyList<double> input, IReadOnlyList<double> output, Func<int, double> upperInfo, double learningRate, ParameterGradients gradients)
		{
			double[] delta = new double[Weight.GetLength(0)];
			Parallel.For(0, Weight.GetLength(0), i =>
			{
				delta[i] = GetDelta(i, output, upperInfo);
				for (int j = 0; j < Weight.GetLength(1); j++)
					gradients.Weight[i, j] -= learningRate * (delta[i] * input[j]);
				gradients.Bias[i] -= learningRate * delta[i];
			});
			return j => Enumerable.Range(0, gradients.UnmodifiedWeight.GetLength(0)).Select(i => gradients.UnmodifiedWeight[i, j] * delta[i]).Sum();
		}

		/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) を計算します。</summary>
		/// <param name="index">勾配ベクトル中の計算する値の位置を示すインデックスを指定します。</param>
		/// <param name="output">この層からの出力を示すベクトルを指定します。</param>
		/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
		/// <returns><paramref name="index"/> で示された勾配ベクトル中の値。</returns>
		protected abstract double GetDelta(int index, IReadOnlyList<double> output, Func<int, double> upperInfo);

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

		readonly Func<IReadOnlyList<double>, int, double> _differentiatedActivation;

		/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) を計算します。</summary>
		/// <param name="index">勾配ベクトル中の計算する値の位置を示すインデックスを指定します。</param>
		/// <param name="output">この層からの出力を示すベクトルを指定します。</param>
		/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
		/// <returns><paramref name="index"/> で示された勾配ベクトル中の値。</returns>
		protected override double GetDelta(int index, IReadOnlyList<double> output, Func<int, double> upperInfo) { return upperInfo(index) * _differentiatedActivation(output, index); }
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

		/// <summary>この層の線形計算の結果に対するニューラルネットワークのコストの勾配ベクトル (Delta) を計算します。</summary>
		/// <param name="index">勾配ベクトル中の計算する値の位置を示すインデックスを指定します。</param>
		/// <param name="output">この層からの出力を示すベクトルを指定します。</param>
		/// <param name="upperInfo">上位層から得られた勾配計算に必要な情報を指定します。この層が出力層の場合、これは教師信号になります。</param>
		/// <returns><paramref name="index"/> で示された勾配ベクトル中の値。</returns>
		protected override double GetDelta(int index, IReadOnlyList<double> output, Func<int, double> upperInfo) { return output[index] - upperInfo(index); }

		/// <summary>指定された計算結果に基づいて、確率が最大となるクラスを推定します。</summary>
		/// <param name="computed">このロジスティック回帰の計算結果を指定します。</param>
		/// <returns>推定された確率最大のクラスのインデックス。</returns>
		public static int Predict(IEnumerable<double> computed) { return computed.Select((x, i) => new KeyValuePair<int, double>(i, x)).Aggregate((x, y) => x.Value > y.Value ? x : y).Key; }
	}

	public sealed class ParameterGradients
	{
		ParameterGradients(double[,] weight, double[] bias, double[,] unmodifiedWeight, double[] unmodifiedBias)
		{
			Weight = weight;
			Bias = bias;
			UnmodifiedWeight = unmodifiedWeight;
			_unmodifiedBias = unmodifiedBias;
		}

		internal readonly double[,] Weight;

		internal readonly double[] Bias;

		internal readonly double[,] UnmodifiedWeight;

		readonly double[] _unmodifiedBias;

		public static ParameterGradients ForBatch(double[,] weight, double[] bias) { return new ParameterGradients(new double[weight.GetLength(0), weight.GetLength(1)], new double[bias.Length], weight, bias); }

		public static ParameterGradients ForOnline(double[,] weight, double[] bias) { return new ParameterGradients(weight, bias, (double[,])weight.Clone(), null); }

		public void UpdateParameters(int updates)
		{
			Parallel.For(0, Weight.GetLength(0), i =>
			{
				for (int j = 0; j < Weight.GetLength(1); j++)
					UnmodifiedWeight[i, j] += Weight[i, j] / updates;
				if (_unmodifiedBias != null)
					_unmodifiedBias[i] += Bias[i] / updates;
			});
		}
	}
}
