using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	public sealed class ActivationFunction
	{
		ActivationFunction(Action<Func<int, double>, double[]> normal, Func<double, double> differentiated)
		{
			Normal = normal;
			Differentiated = differentiated;
		}

		public static readonly ActivationFunction Sigmoid = new ActivationFunction(
			(x, res) => Parallel.For(0, res.Length, i => res[i] = 1 / (1 + Math.Exp(-x(i)))),
			y => y * (1 - y)
		);

		public readonly Action<Func<int, double>, double[]> Normal;

		public readonly Func<double, double> Differentiated;

		public static void Identity(Func<int, double> input, double[] result) { Parallel.For(0, result.Length, i => result[i] = input(i)); }

		public static void SoftMax(Func<int, double> input, double[] result)
		{
			Parallel.For(0, result.Length, i => result[i] = input(i));
			var max = result.AsParallel().Max();
			Parallel.For(0, result.Length, i => result[i] = Math.Exp(result[i] - max));
			var sum = result.AsParallel().Sum();
			Parallel.For(0, result.Length, i => result[i] /= sum);
		}
	}

	public static class ErrorFunction
	{
		public static double BiClassCrossEntropy(IEnumerable<double> source, IEnumerable<double> target) { return -source.AsParallel().Zip(target.AsParallel(), (y, t) => t * Math.Log(y + 1e-10) + (1 - t) * Math.Log(1 - y + 1e-10)).Sum(); }

		public static double MultiClassCrossEntropy(IEnumerable<double> source, IEnumerable<double> target) { return -source.AsParallel().Zip(target.AsParallel(), (y, t) => t * Math.Log(y + 1e-10)).Sum(); }

		public static double LeastSquaresMethod(IEnumerable<double> source, IEnumerable<double> target) { return source.AsParallel().Zip(target.AsParallel(), (y, t) => y - t).Sum(x => x * x) / 2; }
	}
}
