using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
	public interface ILearningSet
	{
		IReadOnlyList<Pattern> TrainingData { get; }
		IReadOnlyList<Pattern> ValidationData { get; }
		IReadOnlyList<Pattern> TestData { get; }
		int Row { get; }
		int Column { get; }
	}

	public class Pattern
	{
		public Pattern(int label, double[] image)
		{
			Label = label;
			Image = image;
		}
		public int Label { get; private set; }
		public double[] Image { get; private set; }
		public override string ToString() { return "Image: " + Image + ", Label: " + Label; }
	}

	public class MnistSet : ILearningSet
	{
		MnistSet(IEnumerable<Pattern> trainingData, IEnumerable<Pattern> validationData, IEnumerable<Pattern> testData, int row, int column)
		{
			TrainingData = trainingData.Take(5000).ToArray();
			ValidationData = validationData.Take(1000).ToArray();
			TestData = testData.Take(1000).ToArray();
			Row = row;
			Column = column;
		}

		public IReadOnlyList<Pattern> TrainingData { get; private set; }

		public IReadOnlyList<Pattern> ValidationData { get; private set; }

		public IReadOnlyList<Pattern> TestData { get; private set; }

		public int Row { get; private set; }

		public int Column { get; private set; }

		static uint BigEndianToLittleEndian(uint value)
		{
			return
				(value & 0x000000FF) << 24 |
				(value & 0x0000FF00) << 8 |
				(value & 0x00FF0000) >> 8 |
				(value & 0xFF000000) >> 24;
		}
		
		public static MnistSet Load(string directoryName)
		{
			var trainSet = LoadInternal(directoryName + "/train");
			var testSet = LoadInternal(directoryName + "/t10k");
			if (trainSet.Item2 != testSet.Item2 || trainSet.Item3 != testSet.Item3)
				return null;
			return new MnistSet(
				trainSet.Item1.Take(trainSet.Item1.Length - testSet.Item1.Length),
				trainSet.Item1.Skip(trainSet.Item1.Length - testSet.Item1.Length),
				testSet.Item1,
				trainSet.Item2,
				trainSet.Item3
			);
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

	public class PatternRecognitionSet : ILearningSet
	{
		PatternRecognitionSet(IEnumerable<Pattern> trainingData, IEnumerable<Pattern> testData)
		{
			TrainingData = trainingData.ToArray();
			TestData = testData.ToArray();
			ValidationData = TestData;
		}

		public IReadOnlyList<Pattern> TrainingData { get; private set; }

		public IReadOnlyList<Pattern> ValidationData { get; private set; }

		public IReadOnlyList<Pattern> TestData { get; private set; }

		public int Row { get { return 7; } }

		public int Column { get { return 5; } }

		public static PatternRecognitionSet Load(string directoryName)
		{
			return new PatternRecognitionSet(
				LoadPatterns(directoryName + "/pattern2learn.dat"),
				LoadPatterns(directoryName + "/pattern2recog.dat")
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
