#pragma once

template <class T, size_t N> class ShiftRegister
{
public:
	void Push(const T& item)
	{
		data[(baseIndex + count++) % N] = item;
		if (count > N)
		{
			baseIndex = (baseIndex + count - N) % N;
			count = N;
		}
	}
	void Push(T&& item)
	{
		data[(baseIndex + count++) % N] = std::move(item);
		if (count > N)
		{
			baseIndex = (baseIndex + count - N) % N;
			count = N;
		}
	}
	T& operator[](ptrdiff_t index)
	{
		auto actualIndex = (static_cast<ptrdiff_t>(baseIndex) + index) % static_cast<ptrdiff_t>(count);
		if (actualIndex < 0)
			actualIndex += count;
		return data[actualIndex];
	}
	size_t Count() const { return count; }

private:
	T data[N] { };
	size_t baseIndex = 0;
	size_t count = 0;
};