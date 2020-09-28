
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include"ManageLayer.h"
#include"Layer.h"
using namespace std;
vector<vector<double>> get_vector_from_file(string filename);
int main() {
	// ���[�U���͕�
	int soshi,sou,is_online;
	cout << "�f�q��:";
	cin >> soshi;
	cout << "�w��:";
	cin >> sou;
	cout << "�ꊇ�w�K:0, �����w�K:1" << endl <<":";
	cin >> is_online;
	// �f�[�^�ǂݎ�蕔
	vector<vector<double>> input_data = get_vector_from_file("data.csv");
	vector<vector<double>> output_data = get_vector_from_file("data_T.csv");
	vector<vector<double>> test_input_data = get_vector_from_file("dis_sig.csv");
	vector<vector<double>> test_output_data = get_vector_from_file("dis_T_sig.csv");
	// ��������NN�{��
	// ManageLayer �N���X�̑�4�������w�K��������ς�������΂�������
	ManageLayer ml(sou, soshi, input_data[0].size(), output_data[0].size() , 0.01);
	if (is_online) {
		ml.online(input_data,output_data);
	}
	else {
		ml.patch(input_data,output_data);
	}
	/**
	* �����߃|�C���g
	* forword()���g���ďo�͌��ʂ��o���B
	* ���t�f�[�^�Ɩ��w�K�f�[�^�̏o�͌��ʂ��o�͂���B
	*/



	return 0;
}
vector<vector<double>> get_vector_from_file(string filename) {
	ifstream ifs(filename);
	if (ifs.fail()) {
		std::cerr << "���s" << std::endl;
	}
	string str, str1;
	vector<vector<double>> data;
	while (getline(ifs, str)) {
		stringstream ss{ str };
		vector<double> tmp;
		while (getline(ss, str1, ',')) {
			tmp.push_back(stod(str1));
		}
		data.push_back(tmp);
	}
	return data;
}