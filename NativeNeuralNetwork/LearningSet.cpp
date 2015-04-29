#include <fstream>
#include <cstdint>
#include <cassert>
#include <limits>
#include "LearningSet.h"

DataSet& DataSet::operator=(DataSet&& dataset)
{
	Deallocate();
	labels = dataset.labels;
	images = dataset.images;
	row = dataset.row;
	column = dataset.column;
	count = dataset.count;

	dataset.labels = nullptr;
	dataset.images = nullptr;
	dataset.row = 0;
	dataset.column = 0;
	dataset.count = 0;
	return *this;
}

void DataSet::CopyFrom(const DataSet& dataset, unsigned int count)
{
	assert(count > 0 && count <= dataset.count);
	Allocate(count, dataset.row, dataset.column);
	for (unsigned int i = 0; i < count; i++)
	{
		labels[i] = dataset.labels[i];
		std::copy(dataset.images[i], dataset.images[i] + row * column, images[i]);
	}
}

void DataSet::Allocate(unsigned int length, int newRow, int newColumn)
{
	assert(length > 0);
	assert(newRow * newColumn > 0);
	Deallocate();
	labels = new int[length];
	images = new double*[length];
	for (unsigned int i = 0; i < length; i++)
		images[i] = new double[(unsigned)(newRow * newColumn)];
	row = newRow;
	column = newColumn;
	count = length;
}

void DataSet::Deallocate()
{
	delete[] labels;
	labels = nullptr;
	if (images)
	{
		for (unsigned int i = 0; i < count; i++)
			delete[] images[i];
		row = 0;
		column = 0;
		delete[] images;
		images = nullptr;
		count = 0;
	}
}

inline uint32_t ReadInt32BigEndian(std::ifstream& stream)
{
	uint32_t temp;
	stream.read(pointer_cast<char>(&temp), sizeof(temp));
	return
		(temp & 0x000000FF) << 24 |
		(temp & 0x0000FF00) << 8 |
		(temp & 0x00FF0000) >> 8 |
		(temp & 0xFF000000) >> 24;
}

inline uint8_t ReadByte(std::ifstream& stream)
{
	uint8_t temp;
	stream.read(pointer_cast<char>(&temp), sizeof(temp));
	return temp;
}

void LoadMnistSetInternal(DataSet& dataset, const std::string& fileName)
{
	std::ifstream labelFile((fileName + "-labels.idx1-ubyte").c_str(), std::ios::binary | std::ios::in);
	std::ifstream imageFile((fileName + "-images.idx3-ubyte").c_str(), std::ios::binary | std::ios::in);
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
	dataset.Allocate(length, (signed)row, (signed)column);
	for (uint32_t i = 0; i < length; i++)
	{
		dataset.Labels()[i] = ReadByte(labelFile);
		for (uint32_t j = 0; j < imageLength; j++)
			dataset.Images()[i][j] = static_cast<double>(ReadByte(imageFile)) / std::numeric_limits<unsigned char>::max();
	}
}

LearningSet LoadMnistSet(const std::string& directoryName)
{
	LearningSet set;
	LoadMnistSetInternal(set.TrainingData(), directoryName + "/train");
	LoadMnistSetInternal(set.TestData(), directoryName + "/t10k");
	set.ClassCount = 10;
	return set;
}
