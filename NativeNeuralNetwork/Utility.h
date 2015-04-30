#pragma once

#include <memory>
#include <functional>

#define FORCE_UNCOPYABLE(T) \
	void operator=(const T&) = delete; \
	T(const T&) = delete

template <class T> inline T* pointer_cast(void* pointer) { return static_cast<T*>(pointer); }

typedef std::function<double(int)> Indexer;

template <class T> class unique_or_raw_array
{
public:
	FORCE_UNCOPYABLE(unique_or_raw_array);

	unique_or_raw_array() : unique(), raw(nullptr) { }
	unique_or_raw_array(unique_or_raw_array&& right) : unique(std::move(right.unique)), raw(right.raw) { }
	unique_or_raw_array(std::unique_ptr<T[]>&& unique) : unique(std::move(unique)), raw(nullptr) { }
	unique_or_raw_array(const T* raw) : unique(), raw(raw) { }

	const T* get() { return unique ? unique.get() : raw; }

	unique_or_raw_array& operator=(unique_or_raw_array&& right)
	{
		if (this != &right)
		{
			unique = std::move(right.unique);
			raw = right.raw;
		}
		return *this;
	}
	unique_or_raw_array& operator=(const T* raw)
	{
		unique = nullptr;
		this->raw = raw;
		return *this;
	}
	unique_or_raw_array& operator=(std::unique_ptr<T[]>&& unique)
	{
		this->unique = std::move(unique);
		raw = nullptr;
		return *this;
	}

private:
	std::unique_ptr<T[]> unique;
	const T* raw;
};