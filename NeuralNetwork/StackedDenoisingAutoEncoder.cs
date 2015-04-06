using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	/// <summary>
	/// 積層雑音除去自己符号化器を表します。
	/// 
	/// 積層雑音除去自己符号化器モデルはいくつかの雑音除去自己符号化器を積み重ねることにより得られます。
	/// 第 i 層目の雑音除去自己符号化器の隠れ層は第 i + 1 層目の雑音除去自己符号化器の入力になります。
	/// 最初の層の雑音除去自己符号化器は入力として積層雑音除去自己符号化器の入力を受け取り、最後の層の雑音除去自己符号化器の隠れ層は出力を表します。
	/// 注釈: 事前学習後、積層雑音除去自己符号化器は通常の多層パーセプトロンとして扱われます。雑音除去自己符号化器は重みの初期化にのみ使用されます。
	/// </summary>
	public sealed class StackedDenoisingAutoEncoder
	{
		/// <summary><see cref="StackedDenoisingAutoEncoder"/> クラスを乱数生成器と入力次元数を使用して初期化します。</summary>
		/// <param name="rng">重みの初期化と雑音除去自己符号化器の雑音生成に使用される乱数生成器を指定します。</param>
		/// <param name="nIn">このネットワークの入力次元数を指定します。</param>
		public StackedDenoisingAutoEncoder(MersenneTwister rng, int nIn)
		{
			// 積層雑音除去自己符号化器は多層パーセプトロンで、これはすべての中間層の重みが別の雑音除去自己符号化器と共有されています。
			// ここでは最初に積層雑音除去自己符号化器を深い多層パーセプトロンとして構築し、それぞれのシグモイド層の構築時にその層と重みを共有する雑音除去自己符号化器も構築します。
			// 事前学習では、(多層パーセプトロンの重みを更新することにもなる) これらの自己符号化器の訓練を行います。
			// ファインチューニングでは、多層パーセプトロン上で確率的勾配降下法を実行することにより、積層雑音除去自己符号化器の訓練を完了させます。
			HiddenLayers = new HiddenLayerCollection(rng, nIn);
		}

		LogisticRegressionLayer _outputLayer;

		public readonly HiddenLayerCollection HiddenLayers;

		public void SetLogisticRegressionLayer(int neurons)
		{
			_outputLayer = new LogisticRegressionLayer(HiddenLayers[HiddenLayers.Count - 1].Bias.Length, neurons);
			HiddenLayers.Freeze();
		}

		/// <summary>指定されたデータセットに対してファインチューニングを実行します。</summary>
		/// <param name="dataset">ファインチューニングに使用されるデータセットを指定します。このデータにはデータ点とラベルが含まれます。</param>
		/// <param name="learningRate">ファインチューニング段階で使用される学習率を指定します。</param>
		public void FineTune(Pattern[] dataset, double learningRate)
		{
			double[][] inputs = new double[HiddenLayers.Count + 2][];
			for (int d = 0; d < dataset.Length; d++)
			{
				inputs[0] = dataset[d].Image;
				int n = 0;
				for (; n < HiddenLayers.Count; n++)
					inputs[n + 1] = HiddenLayers[n].Compute(inputs[n]);
				inputs[n + 1] = _outputLayer.Compute(inputs[n]);
				Func<int, double> upperInfo = i => i == dataset[d].Label ? 1.0 : 0.0;
				upperInfo = _outputLayer.Learn(inputs[n], inputs[n + 1], upperInfo, learningRate);
				while (--n >= 0)
					upperInfo = HiddenLayers[n].Learn(inputs[n], inputs[n + 1], upperInfo, learningRate);
			}
		}

		/// <summary>指定されたデータセットのバッチ全体に対して誤り率を計算します。</summary>
		/// <param name="dataset">誤り率の計算対象となるデータセットを指定します。このデータセットにはデータ点とラベルが含まれます。</param>
		/// <returns>データセット全体に対して計算された誤り率。</returns>
		public double ComputeErrorRates(IEnumerable<Pattern> dataset) { return dataset.Select(d => _outputLayer.Predict(HiddenLayers.Aggregate(d.Image, (x, y) => y.Compute(x))) != d.Label ? 1 : 0).Average(); }
	}
}
