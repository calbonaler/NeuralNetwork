#include "Functions.h"

ActivationFunction ActivationFunction::_logisticSigmoid([](const Indexer& x, VectorType& res)
{
#ifdef NEURALNETWORK_USE_GPU
	concurrency::array_view<ValueType, 1> resView(res.size(), &res[0]);
	resView.discard_data();
	concurrency::parallel_for(0, static_cast<int>(res.size()), [=](int i) restrict(cpu, amp)
	{
		resView[i] = 1 / (1 + concurrency::fast_math::exp(-x(static_cast<unsigned int>(i))));
	});
	resView.synchronize();
#else
#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(res.size()); i++)
		res[static_cast<unsigned int>(i)] = 1 / (1 + exp(-x(static_cast<unsigned int>(i))));
#endif
}, [](ValueType y) { return y * (1 - y); });

ActivationFunction ActivationFunction::_tanh([](const Indexer& x, VectorType& res)
{
#ifdef NEURALNETWORK_USE_GPU
	concurrency::array_view<ValueType, 1> resView(res.size(), &res[0]);
	resView.discard_data();
	concurrency::parallel_for(0, static_cast<int>(res.size()), [=](int i) restrict(cpu, amp)
	{
		resView[i] = concurrency::fast_math::tanh(x(static_cast<unsigned int>(i)));
	});
	resView.synchronize();
#else
#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(res.size()); i++)
		res[static_cast<unsigned int>(i)] = tanh(x(static_cast<unsigned int>(i)));
#endif
}, [](ValueType y) { return 1 - y * y; });

ActivationFunction ActivationFunction::_rectifiedLinear([](const Indexer& x, VectorType& res)
{
#ifdef NEURALNETWORK_USE_GPU
	concurrency::array_view<ValueType, 1> resView(res.size(), &res[0]);
	resView.discard_data();
	concurrency::parallel_for(0, static_cast<int>(res.size()), [=](int i) restrict(cpu, amp)
	{
		ValueType xs = x(static_cast<unsigned int>(i));
		resView[i] = xs > 0 ? xs : 0;
	});
	resView.synchronize();
#else
#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(res.size()); i++)
	{
		ValueType xs = x(static_cast<unsigned int>(i));
		res[static_cast<unsigned int>(i)] = xs > 0 ? xs : 0;
	}
#endif
}, [](ValueType y) { return y > 0 ? static_cast<ValueType>(1) : static_cast<ValueType>(0); });

ActivationFunction ActivationFunction::_softplus([](const Indexer& x, VectorType& res)
{
#ifdef NEURALNETWORK_USE_GPU
	concurrency::array_view<ValueType, 1> resView(res.size(), &res[0]);
	resView.discard_data();
	concurrency::parallel_for(0, static_cast<int>(res.size()), [=](int i) restrict(cpu, amp)
	{
		ValueType xs = x(static_cast<unsigned int>(i));
		resView[i] = xs > 0 ? xs + concurrency::fast_math::log(1 + concurrency::fast_math::exp(-xs)) : concurrency::fast_math::log(1 + concurrency::fast_math::exp(xs));
	});
	resView.synchronize();
#else
#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(res.size()); i++)
	{
		ValueType xs = x(static_cast<unsigned int>(i));
		res[static_cast<unsigned int>(i)] = xs > 0 ? xs + log(1 + exp(-xs)) : log(1 + exp(xs));
	}
#endif
}, [](ValueType y) { return 1 - exp(-y); });

ActivationFunction ActivationFunction::_identity([](const Indexer& x, VectorType& res)
{
#ifdef NEURALNETWORK_USE_GPU
	concurrency::array_view<ValueType, 1> resView(res.size(), &res[0]);
	resView.discard_data();
	concurrency::parallel_for(0, static_cast<int>(res.size()), [=](int i) restrict(cpu, amp)
	{
		resView[i] = x(static_cast<unsigned int>(i));
	});
	resView.synchronize();
#else
#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(res.size()); i++)
		res[static_cast<unsigned int>(i)] = x(static_cast<unsigned int>(i));
#endif
}, [](ValueType) { return static_cast<ValueType>(1); });

void ActivationFunction::SoftMax(const Indexer& input, VectorType& result)
{
	ValueType max = -std::numeric_limits<ValueType>::infinity();
	for (unsigned int i = 0; i < result.size(); i++)
		max = (std::max)(result[i] = input(i), max);
	ValueType sum = 0;
	for (unsigned int i = 0; i < result.size(); i++)
		sum += result[i] = exp(result[i] - max);
#ifdef NEURALNETWORK_USE_GPU
	concurrency::array_view<ValueType, 1> resView(result.size(), &result[0]);
	concurrency::parallel_for(0, static_cast<int>(result.size()), [=](int i) restrict(cpu, amp)
	{
		resView[i] /= sum;
	});
	resView.synchronize();
#else
#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(result.size()); i++)
		result[static_cast<unsigned int>(i)] /= sum;
#endif
}