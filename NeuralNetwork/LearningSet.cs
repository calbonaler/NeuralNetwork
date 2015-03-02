using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
	public class LearningSet
	{
		public LearningSet(IEnumerable<Pattern> trainingData, IEnumerable<Pattern> testData, int row, int column, int classCount)
		{
			TrainingData = trainingData.ToArray();
			TestData = testData.ToArray();
			Row = row;
			Column = column;
			ClassCount = classCount;
		}
		public readonly IReadOnlyList<Pattern> TrainingData;
		public readonly IReadOnlyList<Pattern> TestData;
		public readonly int Row;
		public readonly int Column;
		public readonly int ClassCount;
		public LearningSet Subset(int trainingDataCount, int testDataCount) { return new LearningSet(TrainingData.Take(trainingDataCount), TestData.Take(testDataCount), Row, Column, ClassCount); }
	}

	public class Pattern
	{
		public Pattern(int label, double[] image)
		{
			Label = label;
			Image = image;
		}
		public readonly int Label;
		public readonly double[] Image;
		public override string ToString() { return "Image: " + Image + ", Label: " + Label; }
	}

	public static class MnistSet
	{
		static uint BigEndianToLittleEndian(uint value)
		{
			return
				(value & 0x000000FF) << 24 |
				(value & 0x0000FF00) << 8 |
				(value & 0x00FF0000) >> 8 |
				(value & 0xFF000000) >> 24;
		}
		
		public static LearningSet Load(string directoryName)
		{
			var trainSet = LoadInternal(directoryName + "/train");
			var testSet = LoadInternal(directoryName + "/t10k");
			if (trainSet.Item2 != testSet.Item2 || trainSet.Item3 != testSet.Item3)
				return null;
			return new LearningSet(trainSet.Item1, testSet.Item1, trainSet.Item2, trainSet.Item3, 10);
		}

		static Tuple<Pattern[], int, int> LoadInternal(string fileName)
		{
			using (FileStream labelFile = new FileStream(fileName + "-labels.idx1-ubyte", FileMode.Open))
			using (BinaryReader labelReader = new BinaryReader(labelFile))
			using (FileStream imageFile = new FileStream(fileName + "-images.idx3-ubyte", FileMode.Open))
			using (BinaryReader imageReader = new BinaryReader(imageFile))
			{
				if (BigEndianToLittleEndian(labelReader.ReadUInt32()) != 0x801)
					return null;
				if (BigEndianToLittleEndian(imageReader.ReadUInt32()) != 0x803)
					return null;
				var length = (int)BigEndianToLittleEndian(labelReader.ReadUInt32());
				if (length != BigEndianToLittleEndian(imageReader.ReadUInt32()))
					return null;
				var row = (int)BigEndianToLittleEndian(imageReader.ReadUInt32());
				var column = (int)BigEndianToLittleEndian(imageReader.ReadUInt32());
				Pattern[] data = new Pattern[length];
				for (int i = 0; i < data.Length; i++)
				{
					var label = labelReader.ReadByte();
					var image = new double[row * column];
					for (int j = 0; j < image.Length; j++)
						image[j] = (double)imageReader.ReadByte() / byte.MaxValue;
					data[i] = new Pattern(label, image);
				}
				return new Tuple<Pattern[], int, int>(data, row, column);
			}
		}
	}

	public static class PatternRecognitionSet
	{
		public static LearningSet Load(string directoryName)
		{
			return new LearningSet(
				LoadPatterns(directoryName + "/pattern2learn.dat"),
				LoadPatterns(directoryName + "/pattern2recog.dat"),
				7, 5, 10
			);
		}

		static IEnumerable<Pattern> LoadPatterns(string fileName)
		{
			using (StreamReader reader = new StreamReader(fileName))
			{
				string line;
				while ((line = reader.ReadLine()) != null)
				{
					var elems = line.Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
					yield return new Pattern(int.Parse(elems[0]), elems.Skip(1).Take(elems.Length - 1).Select(x => double.Parse(x)).ToArray());
				}
			}
		}
	}
}
