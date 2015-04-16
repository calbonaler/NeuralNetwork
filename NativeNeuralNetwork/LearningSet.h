#pragma once

#include <string>
#include "Utility.h"

/// <summary>�w�K����ю��ʂɎg�p�����f�[�^�Z�b�g��\���܂��B</summary>
class DataSet
{
public:
	/// <summary><see cref="DataSet"/> �N���X�̐V�����C���X�^���X�����������܂��B</summary>
	DataSet() : labels(nullptr), images(nullptr), count(0), row(0), column(0) { }

	/// <summary>�w�肳�ꂽ�f�[�^�Z�b�g�̃f�[�^���R�s�[���āA<see cref="DataSet"/> �N���X�̐V�����C���X�^���X�����������܂��B</summary>
	/// <param name="dataset">�R�s�[���̃f�[�^�Z�b�g���w�肵�܂��B</param>
	DataSet(const DataSet& dataset) : labels(nullptr), images(nullptr), count(0), row(0), column(0) { CopyFrom(dataset, dataset.count); }

	/// <summary>�w�肳�ꂽ�f�[�^�Z�b�g�̃f�[�^���ړ����āA<see cref="DataSet"/> �N���X�̐V�����C���X�^���X�����������܂��B</summary>
	/// <param name="dataset">�ړ����̃f�[�^�Z�b�g���w�肵�܂��B</param>
	DataSet(DataSet&& dataset) : labels(nullptr), images(nullptr), count(0), row(0), column(0) { *this = std::move(dataset); }

	/// <summary>�w�肳�ꂽ�f�[�^�Z�b�g�̃f�[�^�̈ꕔ���g�p���āA<see cref="DataSet"/> �N���X�̐V�����C���X�^���X�����������܂��B</summary>
	/// <param name="dataset">��ɂȂ�f�[�^�Z�b�g���w�肵�܂��B</param>
	/// <param name="count"><paramref name="dataset"/> ���炱�̃f�[�^�Z�b�g�ɃR�s�[�����f�[�^�����w�肵�܂��B�f�[�^�͐擪����R�s�[����܂��B</param>
	DataSet(const DataSet& dataset, unsigned int count) : labels(nullptr), images(nullptr), count(0), row(0), column(0) { CopyFrom(dataset, count); }

	/// <summary>���̃f�[�^�Z�b�g��j�����܂��B</summary>
	~DataSet() { Deallocate(); }

	/// <summary>�w�肳�ꂽ�f�[�^�Z�b�g���炱�̃f�[�^�Z�b�g�Ƀf�[�^���R�s�[���܂��B</summary>
	/// <param name="dataset">�f�[�^�̃R�s�[���̃f�[�^�Z�b�g���w�肵�܂��B</param>
	/// <returns>���̃f�[�^�Z�b�g�ւ̎Q�ƁB</returns>
	DataSet& operator=(const DataSet& dataset)
	{
		CopyFrom(dataset, dataset.count);
		return *this;
	}

	/// <summary>�w�肳�ꂽ�f�[�^�Z�b�g���炱�̃f�[�^�Z�b�g�Ƀf�[�^���ړ����܂��B</summary>
	/// <param name="dataset">�f�[�^�̈ړ����̃f�[�^�Z�b�g���w�肵�܂��B</param>
	/// <returns>���̃f�[�^�Z�b�g�ւ̎Q�ƁB</returns>
	DataSet& operator=(DataSet&& dataset);

	/// <summary>�w�肳�ꂽ�f�[�^�Z�b�g�̈ꕔ�����̃f�[�^�Z�b�g�ɃR�s�[���܂��B</summary>
	/// <param name="dataset">��ɂȂ�f�[�^�Z�b�g���w�肵�܂��B</param>
	/// <param name="count"><paramref name="dataset"/> ���炱�̃f�[�^�Z�b�g�ɃR�s�[�����f�[�^�����w�肵�܂��B�f�[�^�͐擪����R�s�[����܂��B</param>
	void CopyFrom(const DataSet& dataset, unsigned int count);

	/// <summary>�摜����у��x���̕ۑ��̈���m�ۂ��܂��B</summary>
	/// <param name="length">���p�^�[�������w�肵�܂��B</param>
	/// <param name="newRow">�摜�̐��������̒������w�肵�܂��B</param>
	/// <param name="newColumn">�摜�̐��������̒������w�肵�܂��B</param>
	void Allocate(unsigned int length, unsigned int newRow, unsigned int newColumn);

	/// <summary>���p�^�[�������擾���܂��B</summary>
	unsigned int Count() const { return count; }

	/// <summary>�摜�̐��������̒������擾���܂��B</summary>
	unsigned int Row() const { return row; }

	/// <summary>�摜�̐��������̒������擾���܂��B</summary>
	unsigned int Column() const { return column; }

	/// <summary>�m�ۂ��ꂽ���x���̕ۑ��̈�ւ̃|�C���^��Ԃ��܂��B</summary>
	unsigned int* Labels() { return labels; }

	/// <summary>�m�ۂ��ꂽ���x���̕ۑ��̈�ւ̃|�C���^��Ԃ��܂��B</summary>
	const unsigned int* Labels() const { return labels; }

	/// <summary>�m�ۂ��ꂽ�摜�̕ۑ��̈�ւ̃|�C���^��Ԃ��܂��B</summary>
	double** Images() { return images; }

	/// <summary>�m�ۂ��ꂽ�摜�̕ۑ��̈�ւ̃|�C���^��Ԃ��܂��B</summary>
	double** const Images() const { return images; }

private:
	unsigned int row;
	unsigned int column;
	unsigned int count;
	unsigned int* labels;
	double** images;

	void Deallocate();
};

/// <summary>�w�K�f�[�^����ю��ʃf�[�^���i�[����Z�b�g��\���܂��B</summary>
class LearningSet
{
public:
	/// <summary><see cref="LearningSet"/> �N���X�̐V�����C���X�^���X�����������܂��B</summary>
	LearningSet() { }

	/// <summary>�w�肳�ꂽ <see cref="LearningSet"/> �̃f�[�^���R�s�[���āA<see cref="LearningSet"/> �N���X�̐V�����C���X�^���X�����������܂��B</summary>
	/// <param name="learningSet">�R�s�[���� <see cref="LearningSet"/> ���w�肵�܂��B</param>
	LearningSet(const LearningSet& learningSet) { *this = learningSet; }

	/// <summary>�w�肳�ꂽ <see cref="LearningSet"/> �̃f�[�^���ړ����āA<see cref="LearningSet"/> �N���X�̐V�����C���X�^���X�����������܂��B</summary>
	/// <param name="learningSet">�ړ����� <see cref="LearningSet"/> ���w�肵�܂��B</param>
	LearningSet(LearningSet&& learningSet) { *this = std::move(learningSet); }

	/// <summary>�w�肳�ꂽ <see cref="LearningSet"/> �̈ꕔ���g�p���āA<see cref="LearningSet"/> �N���X�̐V�����C���X�^���X�����������܂��B</summary>
	/// <param name="learningSet">��ɂȂ� <see cref="LearningSet"/> ���w�肵�܂��B</param>
	/// <param name="trainingDataCount"><paramref name="learningSet"/> ���炱�� <see cref="LearningSet"/> �ɃR�s�[�����w�K�f�[�^�����w�肵�܂��B</param>
	/// <param name="testDataCount"><paramref name="learningSet"/> ���炱�� <see cref="LearningSet"/> �ɃR�s�[�����e�X�g�f�[�^�����w�肵�܂��B</param>
	LearningSet(const LearningSet& learningSet, unsigned int trainingDataCount, unsigned int testDataCount) : trainingData(learningSet.trainingData, trainingDataCount), testData(learningSet.testData, testDataCount) { ClassCount = learningSet.ClassCount; }

	/// <summary>�w�肳�ꂽ <see cref="LearningSet"/> �̃f�[�^������ <see cref="LearningSet"/> �ɃR�s�[���܂��B</summary>
	/// <param name="learningSet">�f�[�^�̃R�s�[���� <see cref="LearningSet"/> ���w�肵�܂��B</param>
	/// <returns>���� <see cref="LearningSet"/> �ւ̎Q�ƁB</returns>
	LearningSet& operator=(const LearningSet& learningSet)
	{
		trainingData = learningSet.trainingData;
		testData = learningSet.testData;
		ClassCount = learningSet.ClassCount;
		return *this;
	}

	/// <summary>�w�肳�ꂽ <see cref="LearningSet"/> �̃f�[�^������ <see cref="LearningSet"/> �Ɉړ����܂��B</summary>
	/// <param name="learningSet">�f�[�^�̈ړ����� <see cref="LearningSet"/> ���w�肵�܂��B</param>
	/// <returns>���� <see cref="LearningSet"/> �ւ̎Q�ƁB</returns>
	LearningSet& operator=(LearningSet&& learningSet)
	{
		trainingData = std::move(learningSet.trainingData);
		testData = std::move(learningSet.testData);
		ClassCount = learningSet.ClassCount;
		return *this;
	}

	/// <summary>�w�K�f�[�^���擾���܂��B</summary>
	DataSet& TrainingData() { return trainingData; }

	/// <summary>�w�K�f�[�^���擾���܂��B</summary>
	const DataSet& TrainingData() const { return trainingData; }

	/// <summary>�e�X�g�f�[�^���擾���܂��B</summary>
	DataSet& TestData() { return testData; }

	/// <summary>�e�X�g�f�[�^���擾���܂��B</summary>
	const DataSet& TestData() const { return testData; }

	/// <summary>���̃Z�b�g�Ɋi�[����Ă���p�^�[���̃N���X���������܂��B</summary>
	unsigned int ClassCount;

private:
	DataSet trainingData;
	DataSet testData;
};

/// <summary>MNIST �����[�h���܂��B</summary>
/// <param name="directoryName">MNIST �f�[�^�����݂���f�B���N�g���̏ꏊ���w�肵�܂��B</param>
LearningSet LoadMnistSet(const std::string& directoryName);