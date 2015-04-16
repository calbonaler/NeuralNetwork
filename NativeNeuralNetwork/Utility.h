#pragma once

class Noncopyable
{
protected:
	Noncopyable() { }
	~Noncopyable() { }
	void operator=(const Noncopyable&) = delete;
	Noncopyable(const Noncopyable&) = delete;
};

template <class T> inline T* pointer_cast(void* pointer) { return static_cast<T*>(pointer); }