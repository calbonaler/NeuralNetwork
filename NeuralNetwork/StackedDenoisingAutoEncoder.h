﻿#pragma once

#include "Layers.h"

/// <summary>
/// 積層雑音除去自己符号化器を表します。
/// 
/// 積層雑音除去自己符号化器モデルはいくつかの雑音除去自己符号化器を積み重ねることにより得られます。
/// 第 i 層目の雑音除去自己符号化器の隠れ層は第 i + 1 層目の雑音除去自己符号化器の入力になります。
/// 最初の層の雑音除去自己符号化器は入力として積層雑音除去自己符号化器の入力を受け取り、最後の層の雑音除去自己符号化器の隠れ層は出力を表します。
/// 注釈: 事前学習後、積層雑音除去自己符号化器は通常の多層パーセプトロンとして扱われます。雑音除去自己符号化器は重みの初期化にのみ使用されます。
/// </summary>
template <class TValue> class StackedDenoisingAutoEncoder final : private boost::noncopyable
{
public:
	/// <summary><see cref="StackedDenoisingAutoEncoder"/> クラスを乱数生成器のシード値と入力次元数を使用して初期化します。</summary>
	/// <param name="rng">重みの初期化と雑音除去自己符号化器の雑音生成に使用される乱数生成器のシード値を指定します。</param>
	/// <param name="nIn">このネットワークの入力次元数を指定します。</param>
	StackedDenoisingAutoEncoder(std::mt19937::result_type rngSeed, unsigned int nIn) : HiddenLayers(rngSeed, nIn) { }

	/// <summary>隠れ層のコレクションを取得します。</summary>
	HiddenLayerCollection<TValue> HiddenLayers;

	/// <summary>この SDA の出力層のニューロン数を指定された値に設定します。</summary>
	/// <param name="neurons">SDA の出力層のニューロン数を指定します。</param>
	void SetLogisticRegressionLayer(unsigned int neurons)
	{
		outputLayer = std::unique_ptr<LogisticRegressionLayer<TValue>>(new LogisticRegressionLayer<TValue>(HiddenLayers.InputNeuronCount(HiddenLayers.Count()), neurons));
		HiddenLayers.Freeze();
	}

	/// <summary>指定されたデータセットに対してファインチューニングを実行します。</summary>
	/// <param name="dataset">ファインチューニングに使用されるデータセットを指定します。このデータにはデータ点とラベルが含まれます。</param>
	/// <param name="learningRate">ファインチューニング段階で使用される学習率を指定します。</param>
	void FineTune(const DataSet<TValue>& dataset, TValue learningRate)
	{
		struct equal
		{
			equal(size_t constant) : constant(constant) { }
			TValue operator[](size_t index) const { return index == constant ? static_cast<TValue>(1.0) : static_cast<TValue>(0.0); }
		private:
			size_t constant;
		};

		auto inputs = std::vector<ReferableVector<TValue>>(HiddenLayers.Count() + 2);
		for (unsigned int d = 0; d < dataset.Labels().size(); d++)
		{
			inputs[0] = dataset.Images()[d];
			size_t n = 0;
			for (; n < HiddenLayers.Count(); n++)
				inputs[n + 1] = HiddenLayers[n].Compute(inputs[n]);
			inputs[n + 1] = outputLayer->Compute(inputs[n]);
			std::valarray<TValue> lowerInfo(LearnLayer(*outputLayer, inputs[n].target(), inputs[n + 1].target(), equal(dataset.Labels()[d]), learningRate));
			while (--n <= HiddenLayers.Count())
				lowerInfo = LearnLayer(HiddenLayers[n], inputs[n].target(), inputs[n + 1].target(), lowerInfo, learningRate);
		}
	}

	/// <summary>指定されたデータセットのバッチ全体に対して誤り率を計算します。</summary>
	/// <param name="dataset">誤り率の計算対象となるデータセットを指定します。このデータセットにはデータ点とラベルが含まれます。</param>
	/// <returns>データセット全体に対して計算された誤り率。</returns>
	template <class TResult> TResult ComputeErrorRates(const DataSet<TValue>& dataset)
	{
		unsigned int sum = 0;
		for (size_t i = 0; i < dataset.Labels().size(); i++)
		{
			if (outputLayer->Predict(HiddenLayers.Compute(dataset.Images()[i], nullptr)) != dataset.Labels()[i])
				sum++;
		}
		return static_cast<TResult>(sum) / dataset.Labels().size();
	}

private:
	std::unique_ptr<LogisticRegressionLayer<TValue>> outputLayer;
};

