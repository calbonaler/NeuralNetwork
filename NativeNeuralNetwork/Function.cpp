#include "Functions.h"

ActivationFunction ActivationFunction::_logisticSigmoid([](const Indexer& x, VectorType& res)
{
	concurrency::array_view<ValueType, 1> resView(static_cast<int>(res.size()), &res[0]);
	resView.discard_data();
	concurrency::parallel_for_each(resView.extent, [=](concurrency::index<1> i) restrict(amp) { resView[i] = 1 / (1 + concurrency::fast_math::exp(-x(i))); });
	resView.synchronize();
}, [](ValueType y) { return y * (1 - y); });

ActivationFunction ActivationFunction::_tanh([](const Indexer& x, VectorType& res)
{
	concurrency::array_view<ValueType, 1> resView(static_cast<int>(res.size()), &res[0]);
	resView.discard_data();
	concurrency::parallel_for_each(resView.extent, [=](concurrency::index<1> i) restrict(amp) { resView[i] = concurrency::fast_math::tanh(x(i)); });
	resView.synchronize();
}, [](ValueType y) { return 1 - y * y; });

ActivationFunction ActivationFunction::_rectifiedLinear([](const Indexer& x, VectorType& res)
{
	concurrency::array_view<ValueType, 1> resView(static_cast<int>(res.size()), &res[0]);
	resView.discard_data();
	concurrency::parallel_for_each(resView.extent, [=](concurrency::index<1> i) restrict(amp)
	{
		ValueType xs = x(i);
		resView[i] = xs > 0 ? xs : 0;
	});
	resView.synchronize();
}, [](ValueType y) { return y > 0 ? static_cast<ValueType>(1) : static_cast<ValueType>(0); });

ActivationFunction ActivationFunction::_softplus([](const Indexer& x, VectorType& res)
{
	concurrency::array_view<ValueType, 1> resView(static_cast<int>(res.size()), &res[0]);
	resView.discard_data();
	concurrency::parallel_for_each(resView.extent, [=](concurrency::index<1> i) restrict(amp)
	{
		ValueType xs = x(i);
		resView[i] = xs > 0 ? xs + concurrency::fast_math::log(1 + concurrency::fast_math::exp(-xs)) : concurrency::fast_math::log(1 + concurrency::fast_math::exp(xs));
	});
	resView.synchronize();
}, [](ValueType y) { return 1 - exp(-y); });

ActivationFunction ActivationFunction::_identity([](const Indexer& x, VectorType& res)
{
	concurrency::array_view<ValueType, 1> resView(static_cast<int>(res.size()), &res[0]);
	resView.discard_data();
	concurrency::parallel_for_each(resView.extent, [=](concurrency::index<1> i) restrict(amp) { resView[i] = x(i); });
	resView.synchronize();
}, [](ValueType) { return static_cast<ValueType>(1); });

void ActivationFunction::SoftMax(const Indexer& input, VectorType& result)
{
	ValueType max = -std::numeric_limits<ValueType>::infinity();
	for (unsigned int i = 0; i < result.size(); i++)
		max = (std::max)(result[i] = input(concurrency::index<1>(static_cast<int>(i))), max);
	ValueType sum = 0;
	for (unsigned int i = 0; i < result.size(); i++)
		sum += result[i] = exp(result[i] - max);
	concurrency::array_view<ValueType, 1> resView(static_cast<int>(result.size()), &result[0]);
	concurrency::parallel_for_each(resView.extent, [=](concurrency::index<1> i) restrict(amp) { resView[i] /= sum; });
	resView.synchronize();
}