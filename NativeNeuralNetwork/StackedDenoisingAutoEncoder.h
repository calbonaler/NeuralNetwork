#pragma once

#include "Layers.h"

/// <summary>
/// �ϑw�G���������ȕ��������\���܂��B
/// 
/// �ϑw�G���������ȕ������탂�f���͂������̎G���������ȕ��������ςݏd�˂邱�Ƃɂ�蓾���܂��B
/// �� i �w�ڂ̎G���������ȕ�������̉B��w�͑� i + 1 �w�ڂ̎G���������ȕ�������̓��͂ɂȂ�܂��B
/// �ŏ��̑w�̎G���������ȕ�������͓��͂Ƃ��Đϑw�G���������ȕ�������̓��͂��󂯎��A�Ō�̑w�̎G���������ȕ�������̉B��w�͏o�͂�\���܂��B
/// ����: ���O�w�K��A�ϑw�G���������ȕ�������͒ʏ�̑��w�p�[�Z�v�g�����Ƃ��Ĉ����܂��B�G���������ȕ�������͏d�݂̏������ɂ̂ݎg�p����܂��B
/// </summary>
class StackedDenoisingAutoEncoder : private Noncopyable
{
public:
	/// <summary><see cref="StackedDenoisingAutoEncoder"/> �N���X�𗐐�������̃V�[�h�l�Ɠ��͎��������g�p���ď��������܂��B</summary>
	/// <param name="rng">�d�݂̏������ƎG���������ȕ�������̎G�������Ɏg�p����闐��������̃V�[�h�l���w�肵�܂��B</param>
	/// <param name="nIn">���̃l�b�g���[�N�̓��͎��������w�肵�܂��B</param>
	StackedDenoisingAutoEncoder(std::mt19937::result_type rngSeed, unsigned int nIn) : HiddenLayers(rngSeed, nIn) { }

	/// <summary>�B��w�̃R���N�V�������擾���܂��B</summary>
	HiddenLayerCollection HiddenLayers;

	/// <summary>���� SDA �̏o�͑w�̃j���[���������w�肳�ꂽ�l�ɐݒ肵�܂��B</summary>
	/// <param name="neurons">SDA �̏o�͑w�̃j���[���������w�肵�܂��B</param>
	void SetLogisticRegressionLayer(unsigned int neurons);

	/// <summary>�w�肳�ꂽ�f�[�^�Z�b�g�ɑ΂��ăt�@�C���`���[�j���O�����s���܂��B</summary>
	/// <param name="dataset">�t�@�C���`���[�j���O�Ɏg�p�����f�[�^�Z�b�g���w�肵�܂��B���̃f�[�^�ɂ̓f�[�^�_�ƃ��x�����܂܂�܂��B</param>
	/// <param name="learningRate">�t�@�C���`���[�j���O�i�K�Ŏg�p�����w�K�����w�肵�܂��B</param>
	void FineTune(const DataSet& dataset, double learningRate);

	/// <summary>�w�肳�ꂽ�f�[�^�Z�b�g�̃o�b�`�S�̂ɑ΂��Č�藦���v�Z���܂��B</summary>
	/// <param name="dataset">��藦�̌v�Z�ΏۂƂȂ�f�[�^�Z�b�g���w�肵�܂��B���̃f�[�^�Z�b�g�ɂ̓f�[�^�_�ƃ��x�����܂܂�܂��B</param>
	/// <returns>�f�[�^�Z�b�g�S�̂ɑ΂��Čv�Z���ꂽ��藦�B</returns>
	double ComputeErrorRates(const DataSet& dataset);

private:
	std::unique_ptr<LogisticRegressionLayer> outputLayer;
};

