#include "Functions.h"

ActivationFunction ActivationFunction::_logisticSigmoid([](const Indexer& x, VectorType& res)
{
#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(res.size()); i++)
		res[static_cast<unsigned int>(i)] = 1 / (1 + exp(-x(static_cast<unsigned int>(i))));
}, [](ValueType y) { return y * (1 - y); });

ActivationFunction ActivationFunction::_tanh([](const Indexer& x, VectorType& res)
{
#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(res.size()); i++)
		res[static_cast<unsigned int>(i)] = tanh(x(static_cast<unsigned int>(i)));
}, [](ValueType y) { return 1 - y * y; });

ActivationFunction ActivationFunction::_rectifiedLinear([](const Indexer& x, VectorType& res)
{
#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(res.size()); i++)
	{
		ValueType xs = x(static_cast<unsigned int>(i));
		res[static_cast<unsigned int>(i)] = xs > 0 ? xs : 0;
	}
}, [](ValueType y) { return y > 0 ? static_cast<ValueType>(1) : static_cast<ValueType>(0); });

ActivationFunction ActivationFunction::_softplus([](const Indexer& x, VectorType& res)
{
#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(res.size()); i++)
	{
		ValueType xs = x(static_cast<unsigned int>(i));
		res[static_cast<unsigned int>(i)] = xs > 0 ? xs + log(1 + exp(-xs)) : log(1 + exp(xs));
	}
}, [](ValueType y) { return 1 - exp(-y); });

ActivationFunction ActivationFunction::_identity([](const Indexer& x, VectorType& res)
{
#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(res.size()); i++)
		res[static_cast<unsigned int>(i)] = x(static_cast<unsigned int>(i));
}, [](ValueType) { return static_cast<ValueType>(1); });