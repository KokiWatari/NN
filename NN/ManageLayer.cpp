#include<cmath>
#include<iostream>
#include<numeric>
#include "Layer.h"
#include"ManageLayer.h"
#include<random>
#include <time.h>
ManageLayer::ManageLayer(int num_layer,int num_rows,int num_input,int num_output,double epsilon)
	: num_layer(num_layer)
	, num_rows(num_rows)
	, num_input(num_input)
	, num_output(num_output)
	, epsilon(epsilon)
{
	for (int l = 0; l < num_layer + 1; ++l) {
		// 最初の中間層の入力の数は，そのままデータの入力の数
		// それ以降の中間層の入力の数は中間層の素子の数(layerクラスの引数の入力の数はバイアスを含まないため)
		// 最後の層は出力層なので出力の数はラベルデータの数
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
* 穴埋めポイント
* 
*/

vector<double> ManageLayer::forword(const vector<double>& inputs) {
	vector <double> out(num_output, 0);
	for (int l = 0; l < num_layer + 1; ++l) {
		// 最初の中間層だけ入力を引数からもらう
		// それ以降は一つ前の中間層の出力を入力としてもらっている
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
			middle_layers[l].set_inputs(middle_layers[l - 1].get_outputs());
			middle_layers[l].calc_outputs();
		}
		
	}
	// 出力層の出力を返り値で返している
	return out;
}



/**
* 穴埋めポイント
*
*/
void ManageLayer::back_online(vector<double> &error) {
	for (int l = num_layer; l >= 0; --l) {
			// 出力層では真値と出力層の出力の差を誤差として更新用dL_dxにセットする
			// それ以外の中間層では重みの更新用のdL_dxを後の層から調達する
		// 更新用dL_dxを使って前の層に渡すdL_dxを算出する　重み更新の前にやること
		// 更新用dL_dxを使って重みを更新
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
* 穴埋めポイント
*
*/
// ほぼ上のback_onlineと同じ
// ただしこちらは重みの更新をしないで，誤差をためる
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
* 穴埋めポイント
*
*/
// 全ての層でためた誤差を使って重みを更新する
// 次のデータのエポックのためにためた誤差をリセットすること
void ManageLayer::back_patch(int data_size) {
	for (int l = num_layer ; l >= 0; --l) {
		middle_layers[l].update_weights_for_patch(data_size);
		middle_layers[l].reset_weights_variation();
	}
}


void ManageLayer::online(const vector<vector<double>>& input_data, vector<vector<double>>& output_data) {
	// 本当は誤差でループの終了をコントロールしなければならないが面倒なので一定回数で終わるようにした
	double ave_gosa = 100;
	for (int times = 0; times < 10001 && ave_gosa / input_data.size() > 0.01; ++times) {
		ave_gosa = 0;
		for (int d = 0; d < input_data.size(); d++) {
			// ここで順方向と逆伝搬をやっている
			vector<double> error(output_data[0].size());
			vector<double> result = forword(input_data[d]);
			for (int out = 0; out < output_data[0].size(); ++out) {
				error[out] = result[out] - output_data[d][out];
			}
			back_online(error);
			// 誤差の算出　これはちゃんと動いてるか調べるもので別に更新に使ってるわけじゃない．
			double gosa = 0;
			for (int i = 0; i < error.size(); i++) {
				gosa += pow(error[i], 2);
			}
			ave_gosa += gosa;
			/*if (times % 100 == 0) {
				cout << "times:" << times << ", " << "gosa:" << gosa << endl;
			}*/
		}
		if (times % 10 == 0) {
			cout << "times:" << times << ", ave_gosa:" << ave_gosa / input_data.size() << endl;
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
// 大体onlineと同じ
void ManageLayer::patch(const vector<vector<double>>& input_data, vector<vector<double>>& output_data) {
	double ave_gosa = 100;
	for (int times = 0; times < 100001 && ave_gosa / input_data.size() > 0.01; ++times) {
		ave_gosa = 0;
		for (int d = 0; d < input_data.size(); d++) {
			vector<double> error(output_data[0].size());
			vector<double> result = forword(input_data[d]);
			for (int out = 0; out < output_data[0].size(); ++out) {
				error[out] = result[out] - output_data[d][out];
			}
			pool_errors_patch(error);//
			double gosa = 0;
			for (int i = 0; i < error.size(); i++) {
				gosa += pow(error[i], 2);
			}
			ave_gosa += gosa;
		}
		// ここでためた誤差を使って一気に逆伝播をかけている
		back_patch(input_data.size());
		if (times % 10 == 0) {
			cout << "times:" << times << ", ave_gosa:" << ave_gosa / input_data.size() << endl;
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
	double ave_gosa = 0;
	for (int d = 0; d < test_input_data.size(); d++) {
		vector<double> error(test_output_data[0].size());
		vector<double> result = forword(test_input_data[d]);
		for (int out = 0; out < test_output_data[0].size(); ++out) {
			error[out] = result[out] - test_output_data[d][out];
		}
		double gosa = 0;
		for (int i = 0; i < error.size(); i++) {
			gosa += pow(error[i], 2);
		}
		ave_gosa += gosa;
	}
	cout << "test_ave_gosa:" << ave_gosa / test_input_data.size() << endl;
}
void ManageLayer::print_weight() {
	for (int l = 0; l < num_layer + 1; ++l) {
		middle_layers[l].print_weight();
	}
}
