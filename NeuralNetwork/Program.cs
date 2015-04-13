using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
	static class Program
	{
		static void Main(string[] args) { TestSdA(MnistSet.Load("MNIST").Subset(5000, 1000)); }

		const int PreTrainingEpochs = 15;
		const double PreTrainingLearningRate = 0.001;
		static readonly IReadOnlyList<double> PreTrainingCorruptionLevels = new double[] { 0.1, 0.2, 0.3 };

		const int FineTuningEpochs = 100;
		const double FineTuningLearningRate = 0.01;

		static void TestSdA(LearningSet datasets)
		{
			var sda = new StackedDenoisingAutoEncoder(89677, datasets.Row * datasets.Column);

			using (StreamWriter writer = new StreamWriter("Experiments (Variable Neurons).txt"))
			{
				writer.AutoFlush = true;

				for (int i = 0; i < 1; i++)
				{
					for (int neurons = 100; neurons <= 10000; neurons += 100)
					{
						sda.HiddenLayers.Set(i, neurons);
						Console.WriteLine("Number of neurons of layer {0} is {1}", i, neurons);
						Console.WriteLine("... pre-training the model");
						double costTrain = 0;
						for (var epoch = 1; epoch <= PreTrainingEpochs; epoch++)
						{
							costTrain = sda.HiddenLayers[i].Train(datasets.TrainingData, PreTrainingLearningRate, PreTrainingCorruptionLevels[i]);
							Console.WriteLine("Pre-training layer {0}, epoch {1}, cost {2}", i, epoch, costTrain);
						}
						var costTest = sda.HiddenLayers[i].ComputeCost(datasets.TestData, PreTrainingCorruptionLevels[i]);
						Console.WriteLine("Pre-training layer {0} complete with training cost {1}, test cost {2}", i, costTrain, costTest);
						writer.WriteLine("{0}, {1}, {2}, {3}", i, neurons, costTrain, costTest);
					}
				}

				//sda.SetLogisticRegressionLayer(datasets.ClassCount);
				//Console.WriteLine("... finetunning the model");
				//var testScore = double.PositiveInfinity;
				//var bestEpoch = 0;
				//for (int epoch = 1; epoch < FineTuningEpochs; epoch++)
				//{
				//	sda.FineTune(datasets.TrainingData, FineTuningLearningRate);
				//	var thisTestLoss = sda.ComputeErrorRates(datasets.TestData);
				//	Console.WriteLine("epoch {0}, test score {1} %", epoch, thisTestLoss * 100.0);
				//	if (thisTestLoss < testScore)
				//	{
				//		testScore = thisTestLoss;
				//		bestEpoch = epoch;
				//	}
				//}
				//Console.WriteLine("Optimization complete with best test score of {0} %, on epoch {1}", testScore * 100.0, bestEpoch);
			}
		}
	}
}
