#ifndef MANAGELAYER_H
#define MANAGELAYER_H
#include<vector>
using namespace std;
class layer;
// ������Layer�N���X���Ǘ����Ċw�K�Ǝ��ʂ��s���N���X�B
class ManageLayer {
private:
	// variable
	const int num_layer;         // ���C���[�̐�
	const int num_rows;          // �f�q�̐�
	const int num_input;         // ���͂̐�
	const int num_output;        // �o�͂̐�
	const double epsilon;        // �w�K��
	vector<layer> middle_layers; // ���ԑw
	// function
	/** back_online, back_patch
	* ���͂͋��t�f�[�^�Əo�͂̌덷�B
	* back_online�͒����w�K�ŏd�݂��X�V�Aback_patch�͈ꊇ�w�K�ŏd�݂��X�V
	*/
	void back_online(vector<double> &error);
	void back_patch(int data_size);
	/** pool_errors_patch
	* ���͂͋��t�f�[�^�Əo�͂̌덷�B
	* back_patch�ŗp����덷��~�ς�����
	*/
	void pool_errors_patch(const vector<double> &error);// patch�̋t�����Ō덷��~�ς�����
public:
	ManageLayer(int num_layer,int num_rows,int num_input,int num_output,double epsilon);
	virtual ~ManageLayer() {};
	/** online, patch
	* �����͋��t�f�[�^�̓��͂Əo��
	* online�͒����w�K, patch�͈ꊇ�w�K���s���B
	*/
	void online(const vector<vector<double>> &input_data, vector<vector<double>> &output_data);
	void patch(const vector<vector<double>>& input_data, vector<vector<double>>& output_data);
	/** forword
	* �����͓���
	* �Ԃ�l�̓��f���̏o��
	* ���̓f�[�^�ɑ΂��鏇�������Z���s��
	*/
	vector<double> forword(const vector<double>& input);
	/** loss
	* �����̓e�X�g�f�[�^�̓��͂Əo��
	* �e�X�g�f�[�^����͂���Ǝ��ʗ����Z�o���Ă����
	*/
	void loss(const vector<vector<double>>& test_input_data, vector<vector<double>>& test_output_data);
	void print_weight();// �f�o�b�N�p�̊֐�
};
#endif MANAGELAYER_H
