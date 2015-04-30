#include <fstream>
#include <cstdint>
#include <cassert>
#include <limits>
#include "LearningSet.h"

DataSet& DataSet::operator=(DataSet&& dataset)
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

void DataSet::CopyFrom(const DataSet& dataset, size_t count)
{
	assert(count > 0 && count <= dataset.count);
	Allocate(count, dataset.row, dataset.column);
	for (unsigned int i = 0; i < count; i++)
	{
		labels[i] = dataset.labels[i];
		std::copy(dataset.images[i].begin(), dataset.images[i].end(), images[i].begin());
	}
}

void DataSet::Allocate(size_t length, int newRow, int newColumn)
{
	assert(newRow * newColumn > 0);
	labels.resize(length);
	images.resize(length);
	for (size_t i = 0; i < length; i++)
		images[i].resize((unsigned)(newRow * newColumn));
	row = newRow;
	column = newColumn;
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
