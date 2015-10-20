#pragma once

/// <summary>学習および識別に使用されるデータセットを表します。</summary>
template <class TValue> class DataSet final
{
public:
	/// <summary><see cref="DataSet"/> クラスの新しいインスタンスを初期化します。</summary>
	DataSet() : row(0), column(0) { }

	/// <summary>指定されたデータセットのデータをコピーして、<see cref="DataSet"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="dataset">コピー元のデータセットを指定します。</param>
	DataSet(const DataSet& dataset) : row(0), column(0) { From(dataset, dataset.labels.size()); }

	/// <summary>指定されたデータセットのデータを移動して、<see cref="DataSet"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="dataset">移動元のデータセットを指定します。</param>
	DataSet(DataSet&& dataset) : row(0), column(0) { *this = std::move(dataset); }

	/// <summary>指定されたデータセットのデータの一部を使用して、<see cref="DataSet"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="dataset">基になるデータセットを指定します。</param>
	/// <param name = "index"><paramref name="dataset"/> 内のコピーが開始される位置を指定します。</param>
	/// <param name="count"><paramref name="dataset"/> からこのデータセットにコピーされるデータ数を指定します。</param>
	DataSet(const DataSet& dataset, size_t index, size_t count) : row(0), column(0) { CopyFrom(dataset, index, count); }

	/// <summary>指定されたデータセットのデータの一部を使用して、<see cref="DataSet"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="dataset">基になるデータセットを指定します。</param>
	/// <param name = "index"><paramref name="dataset"/> 内の移動が開始される位置を指定します。</param>
	/// <param name="count"><paramref name="dataset"/> からこのデータセットに移動されるデータ数を指定します。</param>
	DataSet(DataSet&& dataset, size_t index, size_t count) : row(0), column(0) { From(dataset, index, count); }

	/// <summary>指定されたデータセットからこのデータセットにデータをコピーします。</summary>
	/// <param name="dataset">データのコピー元のデータセットを指定します。</param>
	/// <returns>このデータセットへの参照。</returns>
	DataSet& operator=(const DataSet& dataset)
	{
		CopyFrom(dataset, dataset.labels.size());
		return *this;
	}

	/// <summary>指定されたデータセットからこのデータセットにデータを移動します。</summary>
	/// <param name="dataset">データの移動元のデータセットを指定します。</param>
	/// <returns>このデータセットへの参照。</returns>
	DataSet& operator=(DataSet&& dataset)
	{
		if (this != &dataset)
		{
			labels = std::move(dataset.labels);
			images = std::move(dataset.images);
			row = dataset.row;
			column = dataset.column;

			dataset.row = 0;
			dataset.column = 0;
		}
		return *this;
	}

	/// <summary>指定されたデータセットの一部をこのデータセットにコピーします。</summary>
	/// <param name="dataset">基になるデータセットを指定します。</param>
	/// <param name="index"><paramref name="dataset"/> 内のコピーが開始される位置を指定します。</param>
	/// <param name="count"><paramref name="dataset"/> からこのデータセットにコピーされるデータ数を指定します。</param>
	void From(const DataSet& dataset, size_t index, size_t count)
	{
		if (count < 0)
			throw std::invalid_argument("count must not be negative.");
		if (index < 0 || index > dataset.labels.size() - count)
			throw std::invalid_argument("index must be in range [0, dataset.Labels().size() - count]");
		Allocate(count, dataset.row, dataset.column);
		for (unsigned int i = 0; i < count; i++)
		{
			labels[i] = dataset.labels[i + index];
			images[i] = dataset.images[i + index];
		}
	}

	/// <summary>指定されたデータセットの一部をこのデータセットに移動します。</summary>
	/// <param name="dataset">基になるデータセットを指定します。</param>
	/// <param name="index"><paramref name="dataset"/> 内の移動が開始される位置を指定します。</param>
	/// <param name="count"><paramref name="dataset"/> からこのデータセットに移動されるデータ数を指定します。</param>
	void From(DataSet&& dataset, size_t index, size_t count)
	{
		if (count < 0)
			throw std::invalid_argument("count must not be negative.");
		if (index < 0 || index > dataset.labels.size() - count)
			throw std::invalid_argument("index must be in range [0, dataset.Labels().size() - count]");
		Allocate(count, dataset.row, dataset.column);
		for (unsigned int i = 0; i < count; i++)
		{
			labels[i] = std::move(dataset.labels[i + index]);
			images[i] = std::move(dataset.images[i + index]);
		}
	}

	/// <summary>画像およびラベルの保存領域を確保します。</summary>
	/// <param name="length">総パターン数を指定します。</param>
	/// <param name="newRow">画像の垂直方向の長さを指定します。</param>
	/// <param name="newColumn">画像の水平方向の長さを指定します。</param>
	void Allocate(size_t length, unsigned int newRow, unsigned int newColumn)
	{
		if (newRow <= 0 || newColumn <= 0)
			throw std::invalid_argument("newRow and newColumn must not be 0");
		labels.resize(length);
		images.resize(length);
		for (size_t i = 0; i < length; i++)
			images[i].resize(newRow * newColumn);
		row = newRow;
		column = newColumn;
	}

	/// <summary>画像の垂直および水平方向の長さを設定します。</summary>
	/// <param name="newRow">画像の垂直方向の長さを指定します。</param>
	/// <param name="newColumn">画像の水平方向の長さを指定します。</param>
	void SetDimension(unsigned int newRow, unsigned int newColumn)
	{
		row = newRow;
		column = newColumn;
	}

	/// <summary>画像の垂直方向の長さを取得します。</summary>
	unsigned int Row() const { return row; }

	/// <summary>画像の水平方向の長さを取得します。</summary>
	unsigned int Column() const { return column; }

	/// <summary>確保されたラベルの保存領域へのポインタを返します。</summary>
	std::vector<unsigned int>& Labels() { return labels; }

	/// <summary>確保されたラベルの保存領域へのポインタを返します。</summary>
	const std::vector<unsigned int>& Labels() const { return labels; }

	/// <summary>確保された画像の保存領域へのポインタを返します。</summary>
	std::vector<std::valarray<TValue>>& Images() { return images; }

	/// <summary>確保された画像の保存領域へのポインタを返します。</summary>
	const std::vector<std::valarray<TValue>>& Images() const { return images; }

private:
	unsigned int row;
	unsigned int column;
	std::vector<unsigned int> labels;
	std::vector<std::valarray<TValue>> images;
};

/// <summary>学習データおよび識別データを格納するセットを表します。</summary>
template <class TValue> class LearningSet final
{
public:
	/// <summary><see cref="LearningSet"/> クラスの新しいインスタンスを初期化します。</summary>
	LearningSet() { }

	/// <summary>指定された <see cref="LearningSet"/> のデータをコピーして、<see cref="LearningSet"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="learningSet">コピー元の <see cref="LearningSet"/> を指定します。</param>
	LearningSet(const LearningSet& learningSet) { *this = learningSet; }

	/// <summary>指定された <see cref="LearningSet"/> のデータを移動して、<see cref="LearningSet"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="learningSet">移動元の <see cref="LearningSet"/> を指定します。</param>
	LearningSet(LearningSet&& learningSet) { *this = std::move(learningSet); }

	/// <summary>指定された <see cref="LearningSet"/> のデータをこの <see cref="LearningSet"/> にコピーします。</summary>
	/// <param name="learningSet">データのコピー元の <see cref="LearningSet"/> を指定します。</param>
	/// <returns>この <see cref="LearningSet"/> への参照。</returns>
	LearningSet& operator=(const LearningSet& learningSet)
	{
		trainingData = learningSet.trainingData;
		validationData = learningSet.validationData;
		testData = learningSet.testData;
		ClassCount = learningSet.ClassCount;
		return *this;
	}

	/// <summary>指定された <see cref="LearningSet"/> のデータをこの <see cref="LearningSet"/> に移動します。</summary>
	/// <param name="learningSet">データの移動元の <see cref="LearningSet"/> を指定します。</param>
	/// <returns>この <see cref="LearningSet"/> への参照。</returns>
	LearningSet& operator=(LearningSet&& learningSet)
	{
		if (this != &learningSet)
		{
			trainingData = std::move(learningSet.trainingData);
			validationData = std::move(learningSet.validationData);
			testData = std::move(learningSet.testData);
			ClassCount = learningSet.ClassCount;
		}
		return *this;
	}

	/// <summary>学習データを取得します。</summary>
	DataSet<TValue>& TrainingData() { return trainingData; }

	/// <summary>学習データを取得します。</summary>
	const DataSet<TValue>& TrainingData() const { return trainingData; }

	/// <summary>検証データを取得します。</summary>
	DataSet<TValue>& ValidationData() { return validationData; }
	
	/// <summary>検証データを取得します。</summary>
	const DataSet<TValue>& ValidationData() const { return validationData; }

	/// <summary>テストデータを取得します。</summary>
	DataSet<TValue>& TestData() { return testData; }

	/// <summary>テストデータを取得します。</summary>
	const DataSet<TValue>& TestData() const { return testData; }

	/// <summary>このセットに格納されているパターンのクラス数を示します。</summary>
	unsigned int ClassCount;

private:
	DataSet<TValue> trainingData;
	DataSet<TValue> validationData;
	DataSet<TValue> testData;
};

template <class T> inline T* pointer_cast(void* pointer) { return static_cast<T*>(pointer); }

/// <summary>学習セットのローダーを表します。</summary>
template <class TValue> class LearningSetLoader
{
public:
	virtual ~LearningSetLoader() { }

	/// <summary>学習セットをロードします。</summary>
	/// <param name="directoryName">データが存在するディレクトリの場所を指定します。</param>
	LearningSet<TValue> Load(const std::string& path)
	{
		LearningSet<TValue> set;
		LoadDataSet(set.TrainingData(), GetTrainingPath(path));
		LoadDataSet(set.TestData(), GetTestPath(path));
		auto validationPath = GetValidationPath(path);
		if (!validationPath.empty())
			LoadDataSet(set.ValidationData(), validationPath);
		set.ClassCount = ClassCount();
		return set;
	}

protected:
	virtual void LoadDataSet(DataSet<TValue>& dataset, const std::string& path) = 0;
	virtual std::string GetTrainingPath(const std::string& path) = 0;
	virtual std::string GetValidationPath(const std::string&) { return ""; }
	virtual std::string GetTestPath(const std::string& path) = 0;
	virtual unsigned int ClassCount() { return 10; }
};

template <class TValue> class MnistLoader final : public LearningSetLoader<TValue>
{
protected:
	virtual void LoadDataSet(DataSet<TValue>& dataset, const std::string& path)
	{
		std::ifstream labelFile(path + "-labels.idx1-ubyte", std::ios::binary | std::ios::in);
		std::ifstream imageFile(path + "-images.idx3-ubyte", std::ios::binary | std::ios::in);
		if (ReadInt32BigEndian(labelFile) != 0x801)
			return;
		if (ReadInt32BigEndian(imageFile) != 0x803)
			return;
		auto length = ReadInt32BigEndian(labelFile);
		if (length != ReadInt32BigEndian(imageFile))
			return;
		auto row = ReadInt32BigEndian(imageFile);
		auto column = ReadInt32BigEndian(imageFile);
		auto imageLength = row * column;
		dataset.Allocate(length, row, column);
		for (uint32_t i = 0; i < length; i++)
		{
			dataset.Labels()[i] = ReadByte(labelFile);
			for (uint32_t j = 0; j < imageLength; j++)
				dataset.Images()[i][j] = static_cast<TValue>(ReadByte(imageFile)) / (std::numeric_limits<unsigned char>::max)();
		}
	}

	virtual std::string GetTrainingPath(const std::string& path) { return path + "/train"; }

	virtual std::string GetTestPath(const std::string& path) { return path + "/t10k"; }

private:
	uint32_t ReadInt32BigEndian(std::ifstream& stream)
	{
		uint32_t temp;
		stream.read(pointer_cast<char>(&temp), sizeof(temp));
		return
			(temp & 0x000000FF) << 24 |
			(temp & 0x0000FF00) << 8 |
			(temp & 0x00FF0000) >> 8 |
			(temp & 0xFF000000) >> 24;
	}

	uint8_t ReadByte(std::ifstream& stream)
	{
		uint8_t temp;
		stream.read(pointer_cast<char>(&temp), sizeof(temp));
		return temp;
	}
};

template <class TValue> class Cifar10Loader final : public LearningSetLoader<TValue>
{
protected:
	virtual void LoadDataSet(DataSet<TValue>& dataset, const std::string& path)
	{
		if (!LoadSingleFile(dataset, path))
		{
			unsigned int i = 1;
			while (LoadSingleFile(dataset, path + "_" + std::to_string(i)))
				i++;
		}
		dataset.Images().shrink_to_fit();
		dataset.Labels().shrink_to_fit();
	}

	virtual std::string GetTrainingPath(const std::string& path) { return path + "/data_batch"; }

	virtual std::string GetTestPath(const std::string& path) { return path + "/test_batch"; }

private:
	bool LoadSingleFile(DataSet<TValue>& dataset, const std::string& path)
	{
		std::ifstream file(path + ".bin", std::ios::binary | std::ios::in);
		if (!file)
			return false;
		dataset.SetDimension(32, 32);
		for (size_t i = 0; i < 10000; i++)
		{
			dataset.Labels().push_back(ReadByte(file));
			std::valarray<uint8_t> reds(dataset.Row() * dataset.Column());
			for (size_t j = 0; j < reds.size(); j++)
				reds[j] = ReadByte(file);
			std::valarray<uint8_t> greens(dataset.Row() * dataset.Column());
			for (size_t j = 0; j < greens.size(); j++)
				greens[j] = ReadByte(file);
			std::valarray<TValue> image(dataset.Row() * dataset.Column());
			for (size_t j = 0; j < image.size(); j++)
			{
				auto blue = ReadByte(file);
				image[j] = static_cast<TValue>(0.299 * reds[j] + 0.587 * greens[j] + 0.114 * blue) / (std::numeric_limits<uint8_t>::max)();
			}
			dataset.Images().push_back(std::move(image));
		}
		return true;
	}

	uint8_t ReadByte(std::ifstream& stream)
	{
		uint8_t temp;
		stream.read(pointer_cast<char>(&temp), sizeof(temp));
		return temp;
	}
};

template <class TValue> class Caltech101SilhouettesLoader final : public LearningSetLoader<TValue>
{
protected:
	virtual void LoadDataSet(DataSet<TValue>& dataset, const std::string& path)
	{
		std::ifstream labelFile(path + "_labels.bin", std::ios::binary | std::ios::in);
		std::ifstream imageFile(path + "_images.bin", std::ios::binary | std::ios::in);
		auto length = ReadInt32(labelFile);
		if (length != ReadInt32(imageFile))
			return;
		auto imageLength = ReadInt32(imageFile);
		auto oneSide = static_cast<unsigned int>(sqrt(imageLength));
		dataset.Allocate(length, oneSide, oneSide);
		for (uint32_t i = 0; i < length; i++)
		{
			dataset.Labels()[i] = ReadByte(labelFile);
			for (uint32_t j = 0; j < imageLength; j++)
				dataset.Images()[i][j] = static_cast<TValue>(ReadByte(imageFile)); // value is either 0 or 1
		}
	}

	virtual std::string GetTrainingPath(const std::string& path) { return path + "/train"; }

	virtual std::string GetValidationPath(const std::string& path) { return path + "/valid"; }

	virtual std::string GetTestPath(const std::string& path) { return path + "/test"; }

	virtual unsigned int ClassCount() { return 101; }

private:
	uint32_t ReadInt32(std::ifstream& stream)
	{
		uint32_t temp;
		stream.read(pointer_cast<char>(&temp), sizeof(temp));
		return temp;
	}

	uint8_t ReadByte(std::ifstream& stream)
	{
		uint8_t temp;
		stream.read(pointer_cast<char>(&temp), sizeof(temp));
		return temp;
	}
};

template <class TValue> class PatternRecognitionLoader final : public LearningSetLoader<TValue>
{
protected:
	virtual void LoadDataSet(DataSet<TValue>& dataset, const std::string& path)
	{
		dataset.SetDimension(7, 5);
		std::ifstream file(path, std::ios::in);
		std::string line;
		while (std::getline(file, line))
		{
			std::istringstream ss(line);
			std::string item;
			if (!std::getline(ss, item, ','))
				continue;
			dataset.Labels().push_back(static_cast<unsigned int>(std::stoul(item)));
			std::valarray<TValue> image(dataset.Row() * dataset.Column());
			size_t i = 0;
			while (std::getline(ss, item, ','))
				image[i++] = std::stod(item);
			dataset.Images().push_back(image);
		}
		dataset.Images().shrink_to_fit();
		dataset.Labels().shrink_to_fit();
	}

	virtual std::string GetTrainingPath(const std::string& path) { return path + "/pattern2learn.dat"; }

	virtual std::string GetTestPath(const std::string& path) { return path + "/pattern2recog.dat"; }
};