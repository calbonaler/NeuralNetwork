using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
	static class Program
	{
		static void Main(string[] args)
		{
			var set = MnistSet.Load("MNIST");
			//var set = PatternRecognitionSet.Load("PR");
			using (StreamWriter writer = new StreamWriter("Experiment" + DateTime.Now.ToString("(yyyy-MM-dd HH-mm-ss)") + ".txt"))
			{
				Console.SetOut(writer);
				TestSdA(set);
			}
		}

		/// <summary>
		/// Demonstrates how to train and test a stochastic denoising autoencoder.
		/// This is demonstrated on MNIST.
		/// </summary>
		/// <param name="datasets">dataset</param>
		/// <param name="finetuneLr">learning rate used in the finetune stage (factor for the stochastic gradient)</param>
		/// <param name="pretrainingEpochs">number of epoch to do pretraining</param>
		/// <param name="pretrainLr">learning rate to be used during pre-training</param>
		/// <param name="trainingEpochs"></param>
		/// <param name="batchSize"></param>
		static void TestSdA(ILearningSet datasets, int batchSize = 1)
		{
			for (int neurons = 10; neurons <= 1000; neurons += 10)
			{
				var sda = new StackedDenoisingAutoEncoder(new MersenneTwister(89677), datasets.Row * datasets.Column, neurons, 45, 45, 10);

				Console.WriteLine("... pre-training the model");
				var startTime = DateTime.Now;
				PreTrain(sda, datasets.TrainingData, batchSize);
				Console.Error.WriteLine("The pretraining code ran for {0}", DateTime.Now - startTime);

				Console.WriteLine("... finetunning the model");
				startTime = DateTime.Now;
				var result = FineTune(sda, datasets, batchSize);
				Console.Error.WriteLine("The training code ran for {0}", DateTime.Now - startTime);
				Console.WriteLine("Optimization complete with best validation score of {0} %, on epoch {1}, with test performance {2} %", result.Item1 * 100.0, result.Item3, result.Item2 * 100.0);
			}
		}

		static void PreTrain(StackedDenoisingAutoEncoder sda, IReadOnlyList<Pattern> dataset, int batchSize, double[] corruptionLevels = null, int epochs = 15, double learningRate = 0.001)
		{
			if (corruptionLevels == null)
				corruptionLevels = new[] { 0.1, 0.2, 0.3 };
			int i = 0;
			foreach (var da in sda.DenoisingAutoEncoders)
			{
				for (var epoch = 1; epoch <= epochs; epoch++)
				{
					var cost = da.Train(dataset, batchSize, learningRate, corruptionLevels[i]);
					Console.WriteLine("Pre-training layer {0}, epoch {1}, cost {2}", i, epoch, cost);
				}
				i++;
			}
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="sda"></param>
		/// <param name="datasets"></param>
		/// <param name="batchSize"></param>
		/// <param name="epochs"></param>
		/// <param name="learningRate"></param>
		/// <param name="patienceIncrease">wait this much longer when a new best is found</param>
		/// <param name="improvementThreshold">a relative improvement of this much is considered significant</param>
		static Tuple<double, double, int> FineTune(StackedDenoisingAutoEncoder sda, ILearningSet datasets, int batchSize, double learningRate = 0.1, int patienceIncrease = 2, double improvementThreshold = 0.995)
		{
			var nTrainBatches = datasets.TrainingData.Count / batchSize;
			var patience = 10 * nTrainBatches; // look as this many examples regardless

			var bestValidationLoss = double.PositiveInfinity;
			var testScore = 0.0;
			var bestEpoch = 0;

			for (int epoch = 1, iter = -1; iter < patience; epoch++)
			{
				sda.FineTune(datasets.TrainingData, batchSize, learningRate);
				var thisValidationLoss = sda.ComputeErrorRates(datasets.ValidationData);
				Console.WriteLine("epoch {0}, validation error {1} %", epoch, thisValidationLoss * 100.0);

				iter = nTrainBatches * epoch - 1;

				// if we got the best validation score until now
				if (thisValidationLoss < bestValidationLoss)
				{
					//improve patience if loss improvement is good enough
					if (thisValidationLoss < bestValidationLoss * improvementThreshold)
						patience = Math.Max(patience, iter * patienceIncrease);

					// save best validation score and iteration number
					bestValidationLoss = thisValidationLoss;
					bestEpoch = epoch;

					// test it on the test set
					testScore = sda.ComputeErrorRates(datasets.TestData);
					Console.WriteLine("     epoch {0}, test error of best model {1} %", epoch, testScore * 100.0);
				}
			}
			return new Tuple<double, double, int>(bestValidationLoss, testScore, bestEpoch);
		}
	}
}
