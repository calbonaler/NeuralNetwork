#pragma once

template <class T> class Matrix final
{
public:
	Matrix(size_t row, size_t column) : row_(row), column_(column)
	{
		if (row <= 0 || column <= 0)
			throw std::invalid_argument("rows and columns must not be 0");
		data_ = std::valarray<T>(static_cast<T>(0), row * column);
	}

	Matrix(const Matrix& source) : row_(0), column_(0) { *this = source; }

	Matrix(Matrix&& source) : row_(0), column_(0) { *this = std::move(source); }

	Matrix& operator=(const Matrix& source)
	{
		if (this != &source)
		{
			row_ = source.row_;
			column = source.column_;
			data_ = source.data_;
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
			auto tmp_row = source.row_;
			auto tmp_column = source.column_;
			source.row_ = row_;
			source.column_ = column_;
			row_ = tmp_row;
			column_ = tmp_column;
			data_.swap(source.data_);
		}
	}

	size_t Row() const { return row_; }

	size_t Column() const { return column_; }

	T& Element(size_t rowIndex, size_t columnIndex) { return data_[rowIndex * column_ + columnIndex]; }

	const T& Element(size_t rowIndex, size_t columnIndex) const { return data_[rowIndex * column_ + columnIndex]; }

	T& operator()(size_t rowIndex, size_t columnIndex) { return Element(rowIndex, columnIndex); }

	const T& operator()(size_t rowIndex, size_t columnIndex) const { return Element(rowIndex, columnIndex); }

private:
	std::valarray<T> data_;
	size_t row_;
	size_t column_;
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