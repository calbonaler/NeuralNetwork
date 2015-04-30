#pragma once

#include <memory>
#include <functional>
#include <vector>

#define FORCE_UNCOPYABLE(T) \
	void operator=(const T&) = delete; \
	T(const T&) = delete

template <class T> inline T* pointer_cast(void* pointer) { return static_cast<T*>(pointer); }

typedef std::function<double(int)> Indexer;

template <class T> class referenceable_vector
{
public:
	referenceable_vector() : reference_target(nullptr) { }
	referenceable_vector(const std::vector<T>& reference) : reference_target(&reference) { }
	referenceable_vector(std::vector<T>&& instance) : reference_target(nullptr), target(std::move(instance)) { }
	referenceable_vector(referenceable_vector&& right) : reference_target(right.reference_target), target(std::move(right.target)) { }

	referenceable_vector& operator=(const std::vector<T>& reference)
	{
		reference_target = &reference;
		std::vector<T>().swap(target);
		return *this;
	}
	referenceable_vector& operator=(std::vector<T>&& instance)
	{
		reference_target = nullptr;
		target = std::move(instance);
		return *this;
	}
	referenceable_vector& operator=(referenceable_vector&& right)
	{
		reference_target = right.reference_target;
		target = std::move(right.target);
		return *this;
	}

	const std::vector<T>& get() { return reference_target ? *reference_target : target; }

private:
	const std::vector<T>* reference_target;
	std::vector<T> target;
};