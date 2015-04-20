#pragma once

#include <memory>
#include <functional>

class Noncopyable
{
protected:
	Noncopyable() { }
	~Noncopyable() { }
	void operator=(const Noncopyable&) = delete;
	Noncopyable(const Noncopyable&) = delete;
};

template <class T> inline T* pointer_cast(void* pointer) { return static_cast<T*>(pointer); }

typedef std::function<double(unsigned int)> Indexer;

template <class T> class unique_or_raw_array : private Noncopyable
{
public:
	unique_or_raw_array() : unique(), raw(nullptr) { }
	unique_or_raw_array(const T* raw) : unique(), raw(raw) { }
	unique_or_raw_array(std::unique_ptr<T[]>&& unique) : unique(std::move(unique)), raw(nullptr) { }
	unique_or_raw_array(unique_or_raw_array&& right) : unique(std::move(right.unique)), raw(right.raw) { }

	const T* get() { return unique ? unique.get() : raw; }

	unique_or_raw_array& operator=(unique_or_raw_array&& right)
	{
		unique = std::move(right.unique);
		raw = right.raw;
		return *this;
	}

private:
	std::unique_ptr<T[]> unique;
	const T* raw;
};