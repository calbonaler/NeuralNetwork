using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
	static class Program
	{
		static void Main(string[] args) { TestSdA(MnistSet.Load("MNIST").Subset(5000, 1000)); }

		static void TestSdA(LearningSet datasets)
		{
			var sda = new StackedDenoisingAutoEncoder(new MersenneTwister(89677), datasets.Row * datasets.Column, 45, 45, 45, datasets.ClassCount);

			Console.WriteLine("... pre-training the model");
			var startTime = DateTime.Now;
			PreTrain(sda, datasets.TrainingData);
			Console.Error.WriteLine("The pretraining code ran for {0}", DateTime.Now - startTime);

			Console.WriteLine("... finetunning the model");
			startTime = DateTime.Now;
			var result = FineTune(sda, datasets);
			Console.Error.WriteLine("The training code ran for {0}", DateTime.Now - startTime);
			Console.WriteLine("Optimization complete with best test score of {0} %, on epoch {1}", result.Item1 * 100.0, result.Item2);
		}

		static void PreTrain(StackedDenoisingAutoEncoder sda, IReadOnlyList<Pattern> dataset, double[] corruptionLevels = null, int epochs = 15, double learningRate = 0.001)
		{
			if (corruptionLevels == null)
				corruptionLevels = new[] { 0.1, 0.2, 0.3 };
			int i = 0;
			foreach (var da in sda.DenoisingAutoEncoders)
			{
				for (var epoch = 1; epoch <= epochs; epoch++)
				{
					var cost = da.Train(dataset, learningRate, corruptionLevels[i]);
					Console.WriteLine("Pre-training layer {0}, epoch {1}, cost {2}", i, epoch, cost);
				}
				i++;
			}
		}

		static Tuple<double, int> FineTune(StackedDenoisingAutoEncoder sda, LearningSet datasets, int epochs = 100, double learningRate = 0.1)
		{
			var testScore = double.PositiveInfinity;
			var bestEpoch = 0;

			for (int epoch = 1; epoch < epochs; epoch++)
			{
				sda.FineTune(datasets.TrainingData, learningRate);
				var thisTestLoss = sda.ComputeErrorRates(datasets.TestData);
				Console.WriteLine("epoch {0}, test error {1} %", epoch, thisTestLoss * 100.0);
				if (thisTestLoss < testScore)
				{
					testScore = thisTestLoss;
					bestEpoch = epoch;
				}
			}
			return new Tuple<double, int>(testScore, bestEpoch);
		}
	}
}
