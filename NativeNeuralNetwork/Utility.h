﻿#pragma once

typedef float Floating;
typedef Floating ValueType;
typedef std::valarray<ValueType> VectorType;

class ReferableVector
{
public:
	ReferableVector() : reference_target(nullptr) { }
	ReferableVector(const VectorType& reference) : reference_target(&reference) { }
	ReferableVector(VectorType&& instance) : reference_target(nullptr), target(std::move(instance)) { }
	ReferableVector(ReferableVector&& right) : reference_target(right.reference_target), target(std::move(right.target)) { }

	ReferableVector& operator=(const VectorType& reference)
	{
		reference_target = &reference;
		target.resize(0);
		return *this;
	}
	ReferableVector& operator=(VectorType&& instance)
	{
		reference_target = nullptr;
		target = std::move(instance);
		return *this;
	}
	ReferableVector& operator=(ReferableVector&& right)
	{
		reference_target = right.reference_target;
		target = std::move(right.target);
		return *this;
	}

	const VectorType& get() const { return reference_target ? *reference_target : target; }

private:
	const VectorType* reference_target;
	VectorType target;
};