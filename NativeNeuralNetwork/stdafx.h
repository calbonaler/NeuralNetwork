#pragma once

// Standard C++ Libraries

#include <algorithm>
#include <functional>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <valarray>
#include <vector>

// Standard C Libraries for C++

#include <cmath>
#include <cstdint>

// C++ AMP

#pragma warning (push)

#pragma warning (disable : 4365)
#pragma warning (disable : 4062)
#pragma warning (disable : 4265)
#pragma warning (disable : 4355)
#pragma warning (disable : 4571)
#pragma warning (disable : 4623)
#pragma warning (disable : 4946)
#pragma warning (disable : 4987)

#include <amp.h>
#include <amp_math.h>

#pragma warning (pop)

// Boost

#include <boost/core/noncopyable.hpp>