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
			var sda = new StackedDenoisingAutoEncoder(new MersenneTwister(89677), datasets.Row * datasets.Column);
			sda.AddHiddenLayer(45);
			sda.AddHiddenLayer(45);
			sda.AddHiddenLayer(45);
			sda.SetLogisticRegressionLayer(datasets.ClassCount);

			Console.WriteLine("... pre-training the model");
			var startTime = DateTime.Now;
			PreTrain(sda.HiddenLayers, datasets.TrainingData);
			Console.Error.WriteLine("The pretraining code ran for {0}", DateTime.Now - startTime);

			Console.WriteLine("The pretraining complete.");
			Console.WriteLine("Test score of the current model is {0} %", sda.ComputeErrorRates(datasets.TestData) * 100.0);

			Console.WriteLine("... finetunning the model");
			startTime = DateTime.Now;
			var result = FineTune(sda, datasets);
			Console.Error.WriteLine("The training code ran for {0}", DateTime.Now - startTime);
			Console.WriteLine("Optimization complete with best test score of {0} %, on epoch {1}", result.Item1 * 100.0, result.Item2);
		}

		static void PreTrain(IReadOnlyList<HiddenLayer> hiddenLayers, Pattern[] dataset, double[] corruptionLevels = null, int epochs = 15, double learningRate = 0.001)
		{
			if (corruptionLevels == null)
				corruptionLevels = new[] { 0.1, 0.2, 0.3 };
			for (int i = 0; i < hiddenLayers.Count;  i++)
			{
				for (var epoch = 1; epoch <= epochs; epoch++)
				{
					var cost = hiddenLayers[i].Train(dataset, learningRate, corruptionLevels[i]);
					Console.WriteLine("Pre-training layer {0}, epoch {1}, cost {2}", i, epoch, cost);
				}
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
				Console.WriteLine("epoch {0}, test score {1} %", epoch, thisTestLoss * 100.0);
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
