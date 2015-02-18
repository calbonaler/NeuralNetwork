using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	/// <summary>
	/// 雑音除去自己符号化器を表します。
	/// 雑音除去自己符号化器は破壊された入力から、入力をまず隠れ空間に投影しその後入力空間に再投影することで、もとの入力の復元を試みます。
	/// 詳しい情報が必要な場合は Vincent et al., 2008 を参照してください。
	/// </summary>
	/// <remarks>
	/// x を入力とすると、式(1)は確率的写像 q_D の手段によって部分的に破壊された入力を計算します。
	/// 式(2)は入力から隠れ空間に対する投影を計算します。
	/// 式(3)は入力の再構築を行い、式(4)が再構築誤差を計算します。
	///		\tilde{x} ~ q_D(\tilde{x}|x)                                     (1)
	///		y = s(W \tilde{x} + b)                                           (2)
	///		x = s(W' y  + b')                                                (3)
	///		L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log(1-z_k)]       (4)
	/// </remarks>
	public class DenoisingAutoEncoder
	{
		/// <summary><see cref="DenoisingAutoEncoder"/> クラスを指定された結合重み、バイアス、下層によって初期化します。</summary>
		/// <param name="rng">乱数生成器を指定します。</param>
		/// <param name="weight">重みを指定します。</param>
		/// <param name="hiddenBias">隠れ層のバイアスを指定します。</param>
		/// <param name="beforeLayers">この雑音除去自己符号化器が対象とする層よりも下にある層を指定します。</param>
		public DenoisingAutoEncoder(Random rng, double[,] weight, double[] hiddenBias, IEnumerable<Layer> beforeLayers)
		{
			_rng = rng;
			_weight = weight;
			_hiddenBias = hiddenBias ?? new double[_weight.GetLength(0)];
			_visibleBias = new double[_weight.GetLength(1)];
			_beforeLayers = beforeLayers;
		}

		readonly IEnumerable<Layer> _beforeLayers;
		readonly Random _rng;
		readonly double[,] _weight;
		readonly double[] _hiddenBias;
		readonly double[] _visibleBias;

		/// <summary>この雑音除去自己符号化器を指定されたバッチを使用して訓練し、コストを返します。</summary>
		/// <param name="dataset">訓練に使用するデータセットを指定します。</param>
		/// <param name="batchSize">訓練に使用するバッチの大きさを指定します。</param>
		/// <param name="learningRate">学習率を指定します。</param>
		/// <param name="noise">この雑音除去自己符号化器の入力を生成する際のデータの欠損率を指定します。</param>
		/// <returns>訓練後のこの雑音除去自己符号化器の入力に対するコスト。</returns>
		public double Train(IReadOnlyCollection<Pattern> dataset, int batchSize, double learningRate, double noise)
		{
			var weightGrad = batchSize > 1 ? new double[_weight.GetLength(0), _weight.GetLength(1)] : _weight;
			var hiddenBiasGrad = batchSize > 1 ? new double[_hiddenBias.Length] : _hiddenBias;
			var visibleBiasGrad = batchSize > 1 ? new double[_visibleBias.Length] : _visibleBias;
			var latent = new double[_hiddenBias.Length];
			var reconstructed = new double[_visibleBias.Length];
			var delta = new double[_weight.GetLength(0)];
			double cost = 0;
			foreach (var batch in dataset.Partition(batchSize))
			{
				foreach (var input in batch)
				{
					var image = input.Image;
					foreach (var layer in _beforeLayers)
						image = layer.Compute(image);

					var corrupted = image.Select(x => _rng.NextDouble() < noise ? 0 : x).ToArray();
					ActivationFunction.Sigmoid.Normal(j => corrupted.Select((y, i) => y * _weight[j, i]).Sum() + _hiddenBias[j], latent);
					ActivationFunction.Sigmoid.Normal(j => latent.Select((y, i) => y * _weight[i, j]).Sum() + _visibleBias[j], reconstructed);
					Parallel.For(0, _weight.GetLength(0), i =>
					{
						delta[i] = 0;
						for (int j = 0; j < _weight.GetLength(1); j++)
							delta[i] += (reconstructed[j] - image[j]) * _weight[i, j];
						delta[i] *= ActivationFunction.Sigmoid.Differentiated(latent, i);
						hiddenBiasGrad[i] -= learningRate * delta[i];
					});
					Parallel.For(0, _weight.GetLength(1), j =>
					{
						for (int i = 0; i < _weight.GetLength(0); i++)
							weightGrad[i, j] -= learningRate * ((reconstructed[j] - image[j]) * latent[i] + delta[i] * corrupted[j]);
						visibleBiasGrad[j] -= learningRate * (reconstructed[j] - image[j]);
					});
					cost += ErrorFunction.BiClassCrossEntropy(image, reconstructed);
				}

				if (batchSize > 1)
				{
					Parallel.For(0, _weight.GetLength(0), i =>
					{
						for (int j = 0; j < _weight.GetLength(1); j++)
						{
							_weight[i, j] += weightGrad[i, j] / batchSize;
							weightGrad[i, j] = 0;
						}
						_hiddenBias[i] += hiddenBiasGrad[i] / batchSize;
						hiddenBiasGrad[i] = 0;
					});
					Parallel.For(0, _weight.GetLength(1), i =>
					{
						_visibleBias[i] += visibleBiasGrad[i] / batchSize;
						visibleBiasGrad[i] = 0;
					});
				}
			}
			return cost / dataset.Count;
		}
	}
}