#pragma once

#include "Utility.h"

/// <summary>学習および識別に使用されるデータセットを表します。</summary>
class DataSet
{
public:
	/// <summary><see cref="DataSet"/> クラスの新しいインスタンスを初期化します。</summary>
	DataSet() : row(0), column(0) { }

	/// <summary>指定されたデータセットのデータをコピーして、<see cref="DataSet"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="dataset">コピー元のデータセットを指定します。</param>
	DataSet(const DataSet& dataset) : row(0), column(0) { CopyFrom(dataset, dataset.Count()); }

	/// <summary>指定されたデータセットのデータを移動して、<see cref="DataSet"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="dataset">移動元のデータセットを指定します。</param>
	DataSet(DataSet&& dataset) : row(0), column(0) { *this = std::move(dataset); }

	/// <summary>指定されたデータセットのデータの一部を使用して、<see cref="DataSet"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="dataset">基になるデータセットを指定します。</param>
	/// <param name="count"><paramref name="dataset"/> からこのデータセットにコピーされるデータ数を指定します。データは先頭からコピーされます。</param>
	DataSet(const DataSet& dataset, size_t count) : row(0), column(0) { CopyFrom(dataset, count); }

	/// <summary>指定されたデータセットからこのデータセットにデータをコピーします。</summary>
	/// <param name="dataset">データのコピー元のデータセットを指定します。</param>
	/// <returns>このデータセットへの参照。</returns>
	DataSet& operator=(const DataSet& dataset)
	{
		CopyFrom(dataset, dataset.Count());
		return *this;
	}

	/// <summary>指定されたデータセットからこのデータセットにデータを移動します。</summary>
	/// <param name="dataset">データの移動元のデータセットを指定します。</param>
	/// <returns>このデータセットへの参照。</returns>
	DataSet& operator=(DataSet&& dataset);

	/// <summary>指定されたデータセットの一部をこのデータセットにコピーします。</summary>
	/// <param name="dataset">基になるデータセットを指定します。</param>
	/// <param name="count"><paramref name="dataset"/> からこのデータセットにコピーされるデータ数を指定します。データは先頭からコピーされます。</param>
	void CopyFrom(const DataSet& dataset, size_t count);

	/// <summary>画像およびラベルの保存領域を確保します。</summary>
	/// <param name="length">総パターン数を指定します。</param>
	/// <param name="newRow">画像の垂直方向の長さを指定します。</param>
	/// <param name="newColumn">画像の水平方向の長さを指定します。</param>
	void Allocate(size_t length, unsigned int newRow, unsigned int newColumn);

	/// <summary>総パターン数を取得します。</summary>
	size_t Count() const { return labels.size(); }

	/// <summary>画像の垂直方向の長さを取得します。</summary>
	unsigned int Row() const { return row; }

	/// <summary>画像の水平方向の長さを取得します。</summary>
	unsigned int Column() const { return column; }

	/// <summary>確保されたラベルの保存領域へのポインタを返します。</summary>
	std::vector<unsigned int>& Labels() { return labels; }

	/// <summary>確保されたラベルの保存領域へのポインタを返します。</summary>
	const std::vector<unsigned int>& Labels() const { return labels; }

	/// <summary>確保された画像の保存領域へのポインタを返します。</summary>
	std::vector<VectorType>& Images() { return images; }

	/// <summary>確保された画像の保存領域へのポインタを返します。</summary>
	const std::vector<VectorType>& Images() const { return images; }

private:
	unsigned int row;
	unsigned int column;
	std::vector<unsigned int> labels;
	std::vector<VectorType> images;
};

/// <summary>学習データおよび識別データを格納するセットを表します。</summary>
class LearningSet
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

	/// <summary>指定された <see cref="LearningSet"/> の一部を使用して、<see cref="LearningSet"/> クラスの新しいインスタンスを初期化します。</summary>
	/// <param name="learningSet">基になる <see cref="LearningSet"/> を指定します。</param>
	/// <param name="trainingDataCount"><paramref name="learningSet"/> からこの <see cref="LearningSet"/> にコピーされる学習データ数を指定します。</param>
	/// <param name="testDataCount"><paramref name="learningSet"/> からこの <see cref="LearningSet"/> にコピーされるテストデータ数を指定します。</param>
	LearningSet(const LearningSet& learningSet, unsigned int trainingDataCount, unsigned int testDataCount) : trainingData(learningSet.trainingData, trainingDataCount), testData(learningSet.testData, testDataCount) { ClassCount = learningSet.ClassCount; }

	/// <summary>指定された <see cref="LearningSet"/> のデータをこの <see cref="LearningSet"/> にコピーします。</summary>
	/// <param name="learningSet">データのコピー元の <see cref="LearningSet"/> を指定します。</param>
	/// <returns>この <see cref="LearningSet"/> への参照。</returns>
	LearningSet& operator=(const LearningSet& learningSet)
	{
		trainingData = learningSet.trainingData;
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
			testData = std::move(learningSet.testData);
			ClassCount = learningSet.ClassCount;
		}
		return *this;
	}

	/// <summary>学習データを取得します。</summary>
	DataSet& TrainingData() { return trainingData; }

	/// <summary>学習データを取得します。</summary>
	const DataSet& TrainingData() const { return trainingData; }

	/// <summary>テストデータを取得します。</summary>
	DataSet& TestData() { return testData; }

	/// <summary>テストデータを取得します。</summary>
	const DataSet& TestData() const { return testData; }

	/// <summary>このセットに格納されているパターンのクラス数を示します。</summary>
	unsigned int ClassCount;

private:
	DataSet trainingData;
	DataSet testData;
};

/// <summary>MNIST をロードします。</summary>
/// <param name="directoryName">MNIST データが存在するディレクトリの場所を指定します。</param>
LearningSet LoadMnistSet(const std::string& directoryName);
