#include<cmath>
#include<iostream>
#include<numeric>
#include "Layer.h"
#include"ManageLayer.h"
ManageLayer::ManageLayer(int num_layer,int num_rows,int num_input,int num_output,double epsilon)
	: num_layer(num_layer)
	, num_rows(num_rows)
	, num_input(num_input)
	, num_output(num_output)
	, epsilon(epsilon)
{
	for (int l = 0; l < num_layer + 1; ++l) {
		// �ŏ��̒��ԑw�̓��͂̐��́C���̂܂܃f�[�^�̓��͂̐�
		// ����ȍ~�̒��ԑw�̓��͂̐��͒��ԑw�̑f�q�̐�(layer�N���X�̈����̓��͂̐��̓o�C�A�X���܂܂Ȃ�����)
		// �Ō�̑w�͏o�͑w�Ȃ̂ŏo�͂̐��̓��x���f�[�^�̐�
		if (l == 0) {
			middle_layers.push_back(layer(num_rows, num_input, epsilon));
		}
		else if (l == num_layer) {
			middle_layers.push_back(layer(num_output, num_rows, epsilon));
		}
		else {
			middle_layers.push_back(layer(num_rows, num_rows, epsilon));
		}
	}
}

/**
* �����߃|�C���g
* 
*/


vector<double> ManageLayer::forword(const vector<double>& inputs) {
	vector <double> out(num_output);
	for (int l = 0; l < num_layer + 1; ++l) {
		// �ŏ��̒��ԑw�������͂�����������炤
		// ����ȍ~�͈�O�̒��ԑw�̏o�͂���͂Ƃ��Ă�����Ă���
		if (l==0) {
			middle_layers[l].set_inputs(inputs);
			middle_layers[l].calc_outputs();
		}
		else if (l==num_layer) {
			middle_layers[l].set_inputs(middle_layers[l - 1].get_outputs());
			middle_layers[l].calc_outputs();
			out = middle_layers[l].get_outputs();
		}
		else {
			//std::cout << "l= " << l << std::endl;
			middle_layers[l].set_inputs(middle_layers[l-1].get_outputs());
			middle_layers[l].calc_outputs();
		}
		
	}
	// �o�͑w�̏o�͂�Ԃ�l�ŕԂ��Ă���
	return out;
}
/**
* �����߃|�C���g
*
*/
void ManageLayer::back_online(vector<double> &error) {
	for (int l = num_layer; l >= 0; --l) {
			// �o�͑w�ł͐^�l�Əo�͑w�̏o�͂̍����덷�Ƃ��čX�V�pdL_dx�ɃZ�b�g����
			// ����ȊO�̒��ԑw�ł͏d�݂̍X�V�p��dL_dx����̑w���璲�B����
		// �X�V�pdL_dx���g���đO�̑w�ɓn��dL_dx���Z�o����@�d�ݍX�V�̑O�ɂ�邱��
		// �X�V�pdL_dx���g���ďd�݂��X�V
		if (l == num_layer) {
			middle_layers[l].set_dL_dx(error);
		}
		else {
			middle_layers[l].set_dL_dx(middle_layers[l + 1].get_dL_dx_for_before());
		}
		middle_layers[l].calc_dL_dx_for_before();
		middle_layers[l].update_weights();
	}
}
/**
* �����߃|�C���g
*
*/
// �قڏ��back_online�Ɠ���
// ������������͏d�݂̍X�V�����Ȃ��ŁC�덷�����߂�
void ManageLayer::pool_errors_patch(const vector<double>& error) {
	for (int l = num_layer; l >= 0; --l) {
		if (l == num_layer) {
			middle_layers[l].set_dL_dx(error);
		}
		else {
			middle_layers[l].set_dL_dx(middle_layers[l + 1].get_dL_dx_for_before());
		}
		middle_layers[l].calc_dL_dx_for_before();
		middle_layers[l].pool_errors();
	}
}
/**
* �����߃|�C���g
*
*/
// �S�Ă̑w�ł��߂��덷���g���ďd�݂��X�V����
// ���̃f�[�^�̃G�|�b�N�̂��߂ɂ��߂��덷�����Z�b�g���邱��
void ManageLayer::back_patch(int data_size) {
	for (int l = num_layer - 1; l >= 0; --l) {
		middle_layers[l].update_weights_for_patch(data_size);
		middle_layers[l].reset_weights_variation();
	}
}

void ManageLayer::online(const vector<vector<double>> &input_data, vector<vector<double>> &output_data) {
	double loss =100;
	for (int times = 0; times < 10001 && loss / input_data.size() > 0.01; ++times) {
		loss = 0;
		for (int d = 0; d < input_data.size(); d++) {
			// �����ŏ������Ƌt�`��������Ă���
			vector<double> error(output_data[0].size());
			vector<double> result = forword(input_data[d]);
			for (int out = 0; out < output_data[0].size(); ++out) {
				error[out] = result[out] - output_data[d][out];
			}
			back_online(error);
			// �덷�̎Z�o�@����͂����Ɠ����Ă邩���ׂ���̂ŕʂɍX�V�Ɏg���Ă�킯����Ȃ��D
			double square_error = 0;
			for (int i = 0; i < error.size(); i++) {
				square_error += pow(error[i], 2);
			}
			loss += square_error;
			/*if (times % 100 == 0) {
				cout << "times:" << times << ", " << "gosa:" << gosa << endl;
			}*/
		}
		if (times % 10 == 0) {
			cout << "times:" << times << ", loss:" << loss / input_data.size() << endl;
		}	
	}
	/*while(true) {
		cout << "input(push Enter each number):";
		vector<double> tmp_input;
		for (int i = 0; i < input_data[0].size(); ++i) {
			double ttmp;
			cin >> ttmp;
			tmp_input.push_back(ttmp);
		}
		vector<double> ans = forword(tmp_input);
		cout << "ans:[";
		for (int i = 0; i < ans.size(); ++i) {
			cout << ans[i] << " ";
		}
		cout << "]" << endl;
		cout << "continue?(Y/n):";
		string str;
		cin >> str;
		if (str == "n")
			break;
	}*/
}
/**
* �����߃|�C���g
*
*/
// ���online�Ɠ���
void ManageLayer::patch(const vector<vector<double>>& input_data, vector<vector<double>>& output_data) {
	double loss = 100;
	for (int times = 0; times < 100001 && loss / input_data.size() > 0.01; ++times) {
		loss = 0;
		for (int d = 0; d < input_data.size(); d++) {
			vector<double> error(output_data[0].size());
			vector<double> result = forword(input_data[d]);
			for (int out = 0; out < output_data[0].size(); ++out) {
				error[out] = result[out] - output_data[d][out];
			}
			pool_errors_patch(error);
			double square_error = 0;
			for (int i = 0; i < error.size(); i++) {
				square_error += pow(error[i], 2);
			}
			loss += square_error;
		}
		// �����ł��߂��덷���g���Ĉ�C�ɋt�`�d�������Ă���
		back_patch(input_data.size());
		if (times % 10 == 0) {
			cout << "times:" << times << ", loss:" << loss / input_data.size() << endl;
		}

	}
	/*while (true) {
		cout << "input(push Enter each number):";
		vector<double> tmp_input;
		for (int i = 0; i < input_data[0].size(); ++i) {
			double ttmp;
			cin >> ttmp;
			tmp_input.push_back(ttmp);
		}
		vector<double> ans = forword(tmp_input);
		cout << "ans:[";
		for (int i = 0; i < ans.size(); ++i) {
			cout << ans[i] << " ";
		}
		cout << "]" << endl;
		cout << "continue?(Y/n):";
		string str;
		cin >> str;
		if (str == "n")
			break;
	}*/
}
void ManageLayer::loss(const vector<vector<double>>& test_input_data, vector<vector<double>>& test_output_data) {
	double loss = 0;
	for (int d = 0; d < test_input_data.size(); d++) {
		vector<double> error(test_output_data[0].size());
		vector<double> result = forword(test_input_data[d]);
		for (int out = 0; out < test_output_data[0].size(); ++out) {
			error[out] = result[out] - test_output_data[d][out];
		}
		double square_error = 0;
		for (int i = 0; i < error.size(); i++) {
			square_error += pow(error[i], 2);
		}
		loss += square_error;
	}
	cout << "test loss:" << loss / test_input_data.size() << endl;
}
void ManageLayer::print_weight() {
	for (int l = 0; l < middle_layers.size(); ++l) {
		middle_layers[l].print_weight();
	}
}