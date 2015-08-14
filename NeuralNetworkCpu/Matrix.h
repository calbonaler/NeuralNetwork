#pragma once

template <class T> class Matrix final
{
public:
	Matrix(size_t row, size_t column)
	{
		if (row <= 0 || column <= 0)
			throw std::invalid_argument("rows and columns must not be 0");
		Allocate(row, column);
	}

	Matrix(const Matrix& source) : data(nullptr), row(0), column(0) { *this = source; }

	Matrix(Matrix&& source) : data(nullptr), row(0), column(0) { *this = std::move(source); }

	~Matrix() { Free(); }

	Matrix& operator=(const Matrix& source)
	{
		if (this != &source)
		{
			Free();
			Allocate(source.row, source.column);
			std::copy(source.data, source.data + row * column, stdext::make_checked_array_iterator(data, row * column));
		}
		return *this;
	}

	Matrix& operator=(Matrix&& source)
	{
		Swap(source);
		return *this;
	}

	void Swap(Matrix& source)
	{
		if (this != &source)
		{
			auto tmp_data = source.data;
			auto tmp_row = source.row;
			auto tmp_column = source.column;
			source.data = data;
			source.row = row;
			source.column = column;
			data = tmp_data;
			row = tmp_row;
			column = tmp_column;
		}
	}

	size_t Row() const { return row; }

	size_t Column() const { return column; }

	T* Data() { return data; }

	const T* Data() const { return data; }

	T& Element(size_t rowIndex, size_t columnIndex) { return data[rowIndex * column + columnIndex]; }

	const T& Element(size_t rowIndex, size_t columnIndex) const { return data[rowIndex * column + columnIndex]; }

	T& operator()(size_t rowIndex, size_t columnIndex) { return Element(rowIndex, columnIndex); }

	const T& operator()(size_t rowIndex, size_t columnIndex) const { return Element(rowIndex, columnIndex); }

private:
	T* data;
	size_t row;
	size_t column;

	void Allocate(size_t row, size_t column)
	{
		data = new T[row * column]();
		this->row = row;
		this->column = column;
	}

	void Free()
	{
		delete[] data;
		data = nullptr;
		row = 0;
		column = 0;
	}
};

template <class T> class TransposedMatrixView final
{
public:
	static TransposedMatrixView<T> From(Matrix<T>& base) { return TransposedMatrixView<T>(base); }

	static const TransposedMatrixView<T> From(const Matrix<T>& base) { return TransposedMatrixView<T>(const_cast<Matrix<T>&>(base)); }

	TransposedMatrixView(const TransposedMatrixView& source) : base(source.base) { }

	TransposedMatrixView& operator=(const TransposedMatrixView& source)
	{
		base = source.base;
		return *this;
	}

	size_t Row() const { return base->Column(); }

	size_t Column() const { return base->Row(); }

	T& Element(size_t rowIndex, size_t columnIndex) { return base->Element(columnIndex, rowIndex); }

	const T& Element(size_t rowIndex, size_t columnIndex) const { return base->Element(columnIndex, rowIndex); }

	T& operator()(size_t rowIndex, size_t columnIndex) { return Element(rowIndex, columnIndex); }

	const T& operator()(size_t rowIndex, size_t columnIndex) const { return Element(rowIndex, columnIndex); }

private:
	TransposedMatrixView(Matrix<T>& base) : base(&base) { }

	Matrix<T>* base;
};