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
	public class StackedDenoisingAutoEncoder
	{
		/// <summary><see cref="StackedDenoisingAutoEncoder"/> クラスを乱数生成器と各層の次元数を使用して初期化します。</summary>
		/// <param name="rng">重みの初期化と雑音除去自己符号化器の雑音生成に使用される乱数生成器を指定します。</param>
		/// <param name="layerSizes">各層の次元数を入力から出力に向かって指定します。少なくとも 2 つの要素を指定する必要があります。</param>
		public StackedDenoisingAutoEncoder(Random rng, params int[] layerSizes)
		{
			// 積層雑音除去自己符号化器は多層パーセプトロンで、これはすべての中間層の重みが別の雑音除去自己符号化器と共有されています。
			// ここでは最初に積層雑音除去自己符号化器を深い多層パーセプトロンとして構築し、それぞれのシグモイド層の構築時にその層と重みを共有する雑音除去自己符号化器も構築します。
			// 事前学習では、(多層パーセプトロンの重みを更新することにもなる) これらの自己符号化器の訓練を行います。
			// ファインチューニングでは、多層パーセプトロン上で確率的勾配降下法を実行することにより、積層雑音除去自己符号化器の訓練を完了させます。

			if (layerSizes.Length < 2)
				throw new ArgumentException("layerSizes の長さは少なくとも 2 以上である必要があります。", "layerSizes");

			// シグモイド層とロジスティック層の構築
			_layers = layerSizes.Zip(layerSizes.Skip(1), Tuple.Create).Select((x, i) =>
				i < layerSizes.Length - 2 ?
					(Layer)new HiddenLayer(rng, x.Item1, x.Item2, ActivationFunction.Sigmoid) :
					new LogisticRegressionLayer(x.Item1, x.Item2)
			).ToArray();
			
			// 対応する層と重みを共有する雑音除去自己符号化器を構築
			DenoisingAutoEncoders = _layers.Take(_layers.Count - 1).Select((x, i) => new DenoisingAutoEncoder(rng, x.Weight, x.Bias, _layers.Take(i))).ToArray();
		}

		readonly IReadOnlyList<Layer> _layers;

		public readonly IReadOnlyList<DenoisingAutoEncoder> DenoisingAutoEncoders;

		/// <summary>指定されたバッチに対してファインチューニングを実行します。</summary>
		/// <param name="dataset">ファインチューニングに使用されるデータセットを指定します。このデータにはデータ点とラベルが含まれます。</param>
		/// <param name="batchSize">ファインチューニングで使用されるバッチの大きさを指定します。</param>
		/// <param name="learningRate">ファインチューニング段階で使用される学習率を指定します。</param>
		public void FineTune(IEnumerable<Pattern> dataset, int batchSize, double learningRate)
		{
			double[][] inputs = new double[_layers.Count + 1][];
			ParameterGradients[] gradients = new ParameterGradients[_layers.Count];
			foreach (var batch in dataset.Partition(batchSize))
			{
				if (batchSize > 1)
					Parallel.For(0, gradients.Length, i => gradients[i] = ParameterGradients.ForBatch(_layers[i].Weight, _layers[i].Bias));
				else
					Parallel.For(0, gradients.Length, i => gradients[i] = ParameterGradients.ForOnline(_layers[i].Weight, _layers[i].Bias));
				foreach (var data in batch)
				{
					inputs[0] = data.Image;
					for (int n = 0; n < _layers.Count; n++)
						inputs[n + 1] = _layers[n].Compute(inputs[n]);
					Func<int, double> upperInfo = i => i == data.Label ? 1.0 : 0.0;
					for (int n = _layers.Count - 1; n >= 0; n--)
						upperInfo = _layers[n].GetParameterGradients(inputs[n], inputs[n + 1], upperInfo, learningRate, gradients[n]);
				}
				if (batchSize > 1)
				{
					for (int n = 0; n < _layers.Count; n++)
						gradients[n].UpdateParameters(batchSize);
				}
			}
		}

		/// <summary>指定されたデータセットのバッチ全体に対して誤り率を計算します。</summary>
		/// <param name="dataset">誤り率の計算対象となるデータセットを指定します。このデータセットにはデータ点とラベルが含まれます。</param>
		/// <returns>データセット全体に対して計算された誤り率。</returns>
		public double ComputeErrorRates(IEnumerable<Pattern> dataset) { return dataset.Select(d => LogisticRegressionLayer.Predict(_layers.Aggregate(d.Image, (x, y) => y.Compute(x))) != d.Label ? 1 : 0).Average(); }
	}
}
