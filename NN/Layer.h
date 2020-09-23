#ifndef LAYER_H
#define LAYER_H
#include<vector>
using namespace std;
class layer {
private:
	// var
	const double epsilon;                        // �w�K��
	const int num_rows;                          // �p�[�Z�v�g�����̐�
	vector<vector<double>> weights;              // �d��[�ρ[���Ճg�����̐�][����+1]�@+1�̓o�C�A�X����
	vector<double> inputs;                       // ���́@�����ɂ̓o�C�A�X���܂܂��
	const int num_inputs;                        // ���͂̐� �o�C�A�X�͊܂܂Ȃ�
	vector<double> outputs;                      // �o�́@���̓p�[�Z�v�g�����̐��Ɠ���
	vector<double> dL_dx;                        // ���̑w���X�V�Ɏg�����߂̌�̑w���玝���Ă������
	vector<double> dL_dx_for_before;             // �O�̑w���X�V�Ɏg����� 
	vector<vector<double>> sum_errors_for_patch; // �p�b�`�w�K�̂Ƃ��Ƀf�[�^�S���̌덷�����߂Ƃ����
	// func
	void init_weight();                          // �����ŏ�����[0, 1]
	double sigmoid(double x, double gain = 1);
public:

	layer(int num_rows, int num_inputs, double epsilon); // �����̓p�[�Z�v�g�����̐��Ɠ��͂̐��C�w�K��
	virtual ~layer() {};
	void set_inputs(const vector<double>& inputs);       // ���͂��󂯎��p
	void calc_outputs();                                 // �o�͂��v�Z����
	vector<double> get_outputs();                        // �o�͌��ʂ������ ��̑w���g��
	void set_dL_dx(const vector<double>& dL_dx);         // ��납��Ƃ��Ă����덷����荞�ޗp
	void calc_dL_dx_for_before();                                   // �O�̑w���g���덷dL/dY�����@�d�ݍX�V�̑O�Ɏg���Ă�
	void update_weights();                               // �o�C�A�X�ɑ΂�����̂��܂ޏd�݂̍X�V�@
	vector<double> get_dL_dx_for_before();               // �O�̑w���g���덷dL/dY��n���Ƃ��p
	void pool_errors();                                  // �덷��~�ς���
	void update_weights_for_patch(int data_size);        // �p�b�`�w�K��sum_errors_for_patch���g���ďd�݂��X�V����
	void reset_weights_variation();                      // �~�ς��ꂽ�덷�����Z�b�g����B
	void print_weight();                                 // �f�o�b�N�p�֐�
};

inline double layer::sigmoid(double x, double gain)
{
	return 1.0 / (1.0 + exp(-gain * x));
}
inline void layer::set_inputs(const vector<double>& inputs) {
	this->inputs = inputs;
	this->inputs.insert(this->inputs.begin(), 1);// �o�C�A�X�����ŏ��ɒǉ�
}
inline vector<double> layer::get_outputs() {
	return outputs;
}
// ��납��Ƃ��Ă����̂�˂����ނ悤
inline void layer::set_dL_dx(const vector<double>& dL_dx) {
	this->dL_dx = dL_dx;
}
inline vector<double> layer::get_dL_dx_for_before() {
	return dL_dx_for_before;
}
inline void layer::reset_weights_variation() {
	sum_errors_for_patch = vector<vector<double>>(num_rows, vector<double>(num_inputs + 1, 0));// ������
}
#endif LAYER_H
