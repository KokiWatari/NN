#include<random>
#include<iostream>
#include"Layer.h"
layer::layer(int num_rows, int num_inputs, double epsilon)// ���낢�돉�������Ă��邾��
	: num_rows(num_rows)
	, num_inputs(num_inputs)
	, epsilon(epsilon)
{
	weights = vector<vector<double>>(num_rows, vector<double>(num_inputs + 1, 0));
	sum_errors_for_patch = vector<vector<double>>(num_rows, vector<double>(num_inputs + 1, 0));
	outputs = vector<double>(num_rows);
	init_weight();
	dL_dx = vector<double>(num_rows);
	dL_dx_for_before = vector<double>(num_rows);
}

void layer::init_weight() {
	std::random_device rnd;     // �񌈒�I�ȗ���������𐶐�
	std::mt19937 mt(rnd());     //  
	std::uniform_real_distribution<> rand01(0, 1.0);    // [0, 1.0] �͈͂̈�l����
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 0; input < num_inputs + 1; ++input) {
			weights[row][input] = rand01(mt);
		}
	}
}
/**
* �����߃|�C���g
* calc_output()
* ���̊֐��ł͑w�̏o�͂��Z�o����B
* �Z�o�����o�͂�outputs[]�ɑ������B�o�͂̐���num_rows�Œ�`����Ă���B
* Hint
* �w�ɑ΂�����͂�inputs[]�ɓ����Ă���B���͂̐���num_inputs + 1��(�o�C�A�X��)
* �d�݂�weights[][]�ɓ����Ă���B�Y������weights[�o��][����]�̏��ł���B
* �V�O���C�h�֐���sigmoid()�Œ�`����Ă���B
*/

void layer::calc_outputs() {
	vector<double> u(num_rows);
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 0; input < num_inputs + 1; ++input) {
			u[row] += inputs[input] * weights[row][input];
		}
		outputs[row] = sigmoid(u[row]);
	}
}

/**
* �����߃|�C���g
* calc_dL_dx_for_before()
* ���̊֐��ł͑O�̑w���g�p����dL/dx���o�C�A�X�����������͂ɑ΂��ĎZ�o����B
* dL_dx_for_before[]�ɎZ�o����dL/dx��������
* Hint
* �o�C�A�X��inputs[0]�ɓ����Ă���B
* �w�̏o�͂�outputs[]�ɓ����Ă���B�o�͂̐���num_rows�Œ�`����Ă���B
* �d�݂�weights[][]�ɓ����Ă���B�Y������weights[�o��][����]�̏��ł���B
* �O�̑w�ɂ���Čv�Z���ꂽdL/dx��dL_dx[]�ɓ����Ă���BdL_dx�̃T�C�Y�͑w�̏o�͂Ɠ����ł���B
*/
void layer::calc_dL_dx_for_before() {
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 1; input < num_inputs + 1; ++input) {
				dL_dx_for_before[row] += weights[row][input] * outputs[row] * (1 - outputs[row]) * dL_dx[row];
		}
	}
}

/**
* �����߃|�C���g
* update_weights()
* ���̊֐��ł͑w�̏d�݂̍X�V���s���B
* weights[][]���X�V����B
* Hint
* �w�K����epsilon�ɓ����Ă���B
* �w�ɑ΂�����͂�input[]�ɓ����Ă���B���͂̐���num_inputs + 1��(�o�C�A�X��)
* �w�̏o�͂�outputs[]�ɓ����Ă���B�o�͂̐���num_rows�Œ�`����Ă���B
* �d�݂�weights[][]�ɓ����Ă���B�Y������weights[�o��][����]�̏��ł���B
* �O�̑w�ɂ���Čv�Z���ꂽdL/dx��dL_dx[]�ɓ����Ă���BdL_dx�̃T�C�Y�͑w�̏o�͂Ɠ����ł���B
*/
void layer::update_weights() {
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 1; input < num_inputs + 1; ++input) {
			weights[row][input] -= epsilon * inputs[input] * outputs[row] * (1 - outputs[row]) * dL_dx[row];
		}
	}
}

/**
* �����߃|�C���g
* pool_errors()
* ���̊֐��ł͈ꊇ�w�K�̂��߂̏d�݂ɑ΂���X�V�ʂ̍��v���v�Z����B
* �d�݂̍X�V�ʂ̍��v��sum_errors_for_patch[][]�ɓ��͂���B
* Hint
* �Y������sum_errors_for_patch[�o��][����]�̏��ł���B
* �w�K����epsilon�ɓ����Ă���B
* �w�ɑ΂�����͂�input[]�ɓ����Ă���B���͂̐���num_inputs + 1��(�o�C�A�X��)
* �w�̏o�͂�outputs[]�ɓ����Ă���B�o�͂̐���num_rows�Œ�`����Ă���B
* �O�̑w�ɂ���Čv�Z���ꂽdL/dx��dL_dx[]�ɓ����Ă���BdL_dx�̃T�C�Y�͑w�̏o�͂Ɠ����ł���B
*/
void layer::pool_errors() {
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 0; input < num_inputs + 1; ++input) {
			sum_errors_for_patch[row][input] = epsilon * dL_dx[row] / (num_inputs + 1);
		}
	}
}

/**
* �����߃|�C���g
* update_weights_for_patc()
* ���̊֐��ł͈ꊇ�w�K�̎��ɏd�݂̍X�V���s���B
* �d�݂̍X�V�ʂ̍��v�ł���sum_errors_for_patch[][]��p���ďd��weight()�̍X�V���s���B
* Hint
* �Y������sum_errors_for_patch[�o��][����]�̏��ł���B
* �d�݂�weights[][]�ɓ����Ă���B�Y������weights[�o��][����]�̏��ł���B
* 
*/
void layer::update_weights_for_patch(int data_size) {
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 1; input < num_inputs + 1; ++input) {
			weights[row][input] -= sum_errors_for_patch[row][input];
		}
	}
}

void layer::print_weight() {
	for (int row=0; row < num_rows; ++row) {
		for (int input = 0; input < num_inputs + 1; ++input) {
			cout << "row:" << row << ", input:" << input << ", weight:" << weights[row][input] << endl;
		}
	}
}
