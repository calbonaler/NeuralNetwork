#include "Functions.h"
#include "LearningSet.h"
#include <vector>
#include <memory>
#include <random>

/// <summary>�j���[�����l�b�g���[�N�̑w��\�����ۃN���X�ł��B</summary>
class Layer : private Noncopyable
{
public:
	/// <summary>���̑w��j�����܂��B</summary>
	virtual ~Layer();

	/// <summary>���̑w�̌����d�݂������܂��B</summary>
	double** const Weight;
	/// <summary>���̑w�̃o�C�A�X�������܂��B</summary>
	double* const Bias;
	/// <summary>���̑w�̓��̓��j�b�g���������܂��B</summary>
	unsigned int const nIn;
	/// <summary>���̑w�̏o�̓��j�b�g���������܂��B</summary>
	unsigned int const nOut;

	/// <summary>���̑w�̓��͂ɑ΂���o�͂��v�Z���܂��B</summary>
	/// <param name="input">�w�ɓ��͂���x�N�g�����w�肵�܂��B</param>
	/// <returns>���̑w�̏o�͂������x�N�g���B</returns>
	double* Compute(const double* input) const;

	/// <summary>���̑w�̊w�K���s���A���ʑw�̊w�K�ɕK�v�ȏ���Ԃ��܂��B</summary>
	/// <param name="input">���̑w�ւ̓��͂������x�N�g�����w�肵�܂��B</param>
	/// <param name="output">���̑w����̏o�͂������x�N�g�����w�肵�܂��B</param>
	/// <param name="upperInfo">��ʑw���瓾��ꂽ�w�K�ɕK�v�ȏ����w�肵�܂��B���̑w���o�͑w�̏ꍇ�A����͋��t�M���ɂȂ�܂��B</param>
	/// <param name="learningRate">�����d�݂ƃo�C�A�X���ǂ�قǍX�V���邩�������l���w�肵�܂��B</param>
	/// <returns>���ʑw�̊w�K�ɕK�v�ȏ��B</returns>
	double* Learn(const double* input, const double* output, Indexer upperInfo, double learningRate);

protected:
	/// <summary>���͂����j���[�������A���̑w�̃j���[�������A�������֐����g�p���āA<see cref="Layer"/> �N���X�̐V�����C���X�^���X�����������܂��B</summary>
	/// <param name="nIn">���̑w�ɓ��͂����w�̃j���[���������w�肵�܂��B</param>
	/// <param name="nOut">���̑w�̃j���[���������w�肵�܂��B</param>
	/// <param name="activation">���̑w�ɓK�p���銈�����֐����w�肵�܂��B</param>
	Layer(unsigned int nIn, unsigned int nOut, const ActivationFunction::NormalForm& activation);

	/// <summary>���̑w�̐��`�v�Z�̌��ʂɑ΂���j���[�����l�b�g���[�N�̃R�X�g�̌��z�x�N�g�� (Delta) �̗v�f���v�Z���܂��B</summary>
	/// <param name="output">���̑w����̏o�͂������x�N�g���̗v�f���w�肵�܂��B</param>
	/// <param name="upperInfo">��ʑw���瓾��ꂽ���z�v�Z�ɕK�v�ȏ����w�肵�܂��B���̑w���o�͑w�̏ꍇ�A����͋��t�M���ɂȂ�܂��B</param>
	/// <returns>���z�x�N�g���̗v�f�B</returns>
	virtual double GetDelta(double output, double upperInfo) const = 0;

private:
	const ActivationFunction::NormalForm activation;
};

class HiddenLayerCollection;

/// <summary>
/// ���w�p�[�Z�v�g�����̓T�^�I�ȉB��w��\���܂��B
/// ���j�b�g�Ԃ͑S��������Ă���A�������֐���K�p�ł��܂��B
/// ����͎G���������ȕ������������Ɋ܂݂܂��B
/// </summary>
/// <remarks>
/// �G���������ȕ�������͔j�󂳂ꂽ���͂���A���͂��܂��B���Ԃɓ��e�����̌���͋�Ԃɍē��e���邱�ƂŁA���Ƃ̓��͂̕��������݂܂��B
/// �ڂ�����񂪕K�v�ȏꍇ�� Vincent et al., 2008 ���Q�Ƃ��Ă��������B
/// 
/// x ����͂Ƃ���ƁA��(1)�͊m���I�ʑ� q_D �̎�i�ɂ���ĕ����I�ɔj�󂳂ꂽ���͂��v�Z���܂��B
/// ��(2)�͓��͂���B���Ԃɑ΂��铊�e���v�Z���܂��B
/// ��(3)�͓��͂̍č\�z���s���A��(4)���č\�z�덷���v�Z���܂��B
///		\tilde{x} ~ q_D(\tilde{x}|x)                                     (1)
///		y = s(W \tilde{x} + b)                                           (2)
///		x = s(W' y  + b')                                                (3)
///		L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log(1-z_k)]       (4)
/// </remarks>
class HiddenLayer : public Layer
{
public:
	/// <summary><see cref="HiddenLayer"/> �N���X����o�͂̎������A�������֐�����щ��w���g�p���ď��������܂��B</summary>
	/// <param name="nIn">���͂̎��������w�肵�܂��B</param>
	/// <param name="nOut">�B��f�q�̐����w�肵�܂��B</param>
	/// <param name="activation">�B��w�ɓK�p����銈�����֐����w�肵�܂��B</param>
	/// <param name="hiddenLayers">���̉B��w���������Ă��� Stacked Denoising Auto-Encoder �̂��ׂẲB��w��\�����X�g���w�肵�܂��B</param>
	HiddenLayer(unsigned int nIn, unsigned int nOut, const ActivationFunction* activation, HiddenLayerCollection* hiddenLayers);

	/// <summary>���̑w��j�����܂��B</summary>
	~HiddenLayer();

	/// <summary>���̑w����G���������ȕ���������\�����A�w�肳�ꂽ�f�[�^�Z�b�g���g�p���ČP���������ʂ̃R�X�g��Ԃ��܂��B</summary>
	/// <param name="dataset">�P���Ɏg�p����f�[�^�Z�b�g���w�肵�܂��B</param>
	/// <param name="learningRate">�w�K�����w�肵�܂��B</param>
	/// <param name="noise">�\�����ꂽ�G���������ȕ�������̓��͂𐶐�����ۂ̃f�[�^�̌��������w�肵�܂��B</param>
	/// <returns>�\�����ꂽ�G���������ȕ�������̓��͂ɑ΂���R�X�g�B</returns>
	double Train(const DataSet& dataset, double learningRate, double noise);

	/// <summary>���̑w����G���������ȕ���������\�����A�w�肳�ꂽ�f�[�^�Z�b�g�̃R�X�g���v�Z���܂��B</summary>
	/// <param name="dataset">�R�X�g���v�Z����f�[�^�Z�b�g���w�肵�܂��B</param>
	/// <param name="noise">�\�����ꂽ�G���������ȕ�������̓��͂𐶐�����ۂ̃f�[�^�̌��������w�肵�܂��B</param>
	/// <returns>�\�����ꂽ�G���������ȕ�������̓��͂ɑ΂���R�X�g�B</returns>
	double ComputeCost(const DataSet& dataset, double noise) const;

protected:
	/// <summary>���̑w�̐��`�v�Z�̌��ʂɑ΂���j���[�����l�b�g���[�N�̃R�X�g�̌��z�x�N�g�� (Delta) �̗v�f���v�Z���܂��B</summary>
	/// <param name="output">���̑w����̏o�͂������x�N�g���̗v�f���w�肵�܂��B</param>
	/// <param name="upperInfo">��ʑw���瓾��ꂽ���z�v�Z�ɕK�v�ȏ����w�肵�܂��B���̑w���o�͑w�̏ꍇ�A����͋��t�M���ɂȂ�܂��B</param>
	/// <returns>���z�x�N�g���̗v�f�B</returns>
	double GetDelta(double output, double upperInfo) const { return upperInfo * differentiatedActivation(output); }

private:
	const ActivationFunction::DifferentiatedForm differentiatedActivation;
	HiddenLayerCollection* const hiddenLayers;
	double* const visibleBias;
};

/// <summary>�B��w�̃R���N�V������\���܂��B</summary>
class HiddenLayerCollection : private Noncopyable
{
public:
	/// <summary>����������̃V�[�h�l�Ɠ��͑w�̃��j�b�g�����w�肵�āA<see cref="HiddenLayerCollection"/> �N���X�̐V�����C���X�^���X�����������܂��B</summary>
	/// <param name="rngSeed">�B��w�̌v�Z�Ɏg�p����闐��������̃V�[�h�l���w�肵�܂��B</param>
	/// <param name="nIn">���͑w�̃��j�b�g�����w�肵�܂��B</param>
	HiddenLayerCollection(std::mt19937::result_type rngSeed, unsigned int nIn) : RandomNumberGenerator(rngSeed), nextLayerInputUnits(nIn), frozen(false) { }

	/// <summary>�B��w�̌v�Z�Ɏg�p����闐��������������܂��B</summary>
	std::mt19937 RandomNumberGenerator;

	/// <summary>�w�肳�ꂽ�w�̓��̓x�N�g�����v�Z���܂��B�w���w�肳��Ȃ��ꍇ�A���̃��\�b�h�͏o�͑w�̓��̓x�N�g�����v�Z���܂��B</summary>
	/// <param name="input">�ŏ��̉B��w�ɗ^������͂��w�肵�܂��B</param>
	/// <param name="stopLayer">���̓x�N�g�����v�Z����w���w�肵�܂��B���̈����͏ȗ��\�ł��B</param>
	/// <returns>�w�肳�ꂽ�w�̓��̓x�N�g���B�w���w�肳��Ȃ������ꍇ�͏o�͑w�̓��̓x�N�g����Ԃ��܂��B</returns>
	const double* Compute(const double* input, const HiddenLayer* stopLayer) const;

	/// <summary>�w�肳�ꂽ�C���f�b�N�X�̉B��w�̃j���[��������ύX���܂��B���̃��\�b�h�͉B��w��ǉ����邱�Ƃ��ł��܂��B</summary>
	/// <param name="index">�j���[��������ύX����B��w�̃C���f�b�N�X���w�肵�܂��B</param>
	/// <param name="neurons">�w�肳�ꂽ�B��w�̐V�����j���[���������w�肵�܂��B</param>
	void Set(unsigned int index, unsigned int neurons);

	/// <summary>���̃R���N�V�������Œ肵�ĕύX�s�\�ɂ��܂��B</summary>
	void Freeze() { frozen = true; }

	/// <summary>���̃R���N�V�������̎w�肳�ꂽ�C���f�b�N�X�ɂ���B��w�ւ̎Q�Ƃ��擾���܂��B</summary>
	/// <param name="index">�B��w���擾����C���f�b�N�X���w�肵�܂��B</param>
	/// <returns>�擾���ꂽ�B��w�ւ̎Q�ƁB����͕ύX�\�ȎQ�Ƃł��B</returns>
	HiddenLayer& operator[](unsigned int index) { return *items[index]; }

	/// <summary>���̃R���N�V�������Ɋ܂܂�Ă���B��w�̌����w�肵�܂��B</summary>
	/// <returns>�R���N�V�����Ɋ܂܂�Ă���B��w�̌��B</returns>
	unsigned int Count() const { return items.size(); }

private:
	bool frozen;
	unsigned int nextLayerInputUnits;
	std::vector<std::unique_ptr<HiddenLayer>> items;
};

/// <summary>
/// ���N���X���W�X�e�B�b�N��A���s���o�͑w��\���܂��B
/// ���W�X�e�B�b�N��A�͏d�ݍs�� W �ƃo�C�A�X�x�N�g�� b �ɂ���Ċ��S�ɋL�q����܂��B
/// ���ނ̓f�[�^�_�𒴕��ʂ֓��e���邱�Ƃɂ���ĂȂ���܂��B
/// </summary>
class LogisticRegressionLayer : public Layer
{
public:
	/// <summary>���W�X�e�B�b�N��A�̃p�����[�^�����������܂��B</summary>
	/// <param name="nIn">���͑f�q�̐� (�f�[�^�_�����݂����Ԃ̎���) ���w�肵�܂��B</param>
	/// <param name="nOut">�o�͑f�q�̐� (���x�������݂����Ԃ̎���) ���w�肵�܂��B</param>
	LogisticRegressionLayer(unsigned int nIn, unsigned int nOut) : Layer(nIn, nOut, ActivationFunction::SoftMax) { }

	/// <summary>�m�����ő�ƂȂ�N���X�𐄒肵�܂��B</summary>
	/// <param name="input">�w�ɓ��͂���x�N�g�����w�肵�܂��B</param>
	/// <returns>���肳�ꂽ�m���ő�̃N���X�̃C���f�b�N�X�B</returns>
	unsigned int Predict(const double* input) const;

protected:
	/// <summary>���̑w�̐��`�v�Z�̌��ʂɑ΂���j���[�����l�b�g���[�N�̃R�X�g�̌��z�x�N�g�� (Delta) �̗v�f���v�Z���܂��B</summary>
	/// <param name="output">���̑w����̏o�͂������x�N�g���̗v�f���w�肵�܂��B</param>
	/// <param name="upperInfo">��ʑw���瓾��ꂽ���z�v�Z�ɕK�v�ȏ����w�肵�܂��B���̑w���o�͑w�̏ꍇ�A����͋��t�M���ɂȂ�܂��B</param>
	/// <returns>���z�x�N�g���̗v�f�B</returns>
	double GetDelta(double output, double upperInfo) const { return output - upperInfo; }
};