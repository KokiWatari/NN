#include<random>
#include<iostream>
#include"Layer.h"
layer::layer(int num_rows, int num_inputs, double epsilon)// いろいろ初期化しているだけ
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
	std::random_device rnd;     // 非決定的な乱数生成器を生成
	std::mt19937 mt(rnd());     //  
	std::uniform_real_distribution<> rand01(0, 1.0);    // [0, 1.0] 範囲の一様乱数
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 0; input < num_inputs + 1; ++input) {
			weights[row][input] = rand01(mt);
		}
	}
}
/**
* 穴埋めポイント
* calc_output()
* この関数では層の出力を算出する。
* 算出した出力はoutputs[]に代入する。出力の数はnum_rowsで定義されている。
* Hint
* 層に対する入力はinputs[]に入っている。入力の数はnum_inputs + 1個(バイアス分)
* 重みはweights[][]に入っている。添え字はweights[出力][入力]の順である。
* シグモイド関数はsigmoid()で定義されている。
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
* 穴埋めポイント
* calc_dL_dx_for_before()
* この関数では前の層が使用するdL/dxをバイアスを除いた入力に対して算出する。
* dL_dx_for_before[]に算出したdL/dxを代入する
* Hint
* バイアスはinputs[0]に入っている。
* 層の出力はoutputs[]に入っている。出力の数はnum_rowsで定義されている。
* 重みはweights[][]に入っている。添え字はweights[出力][入力]の順である。
* 前の層によって計算されたdL/dxはdL_dx[]に入っている。dL_dxのサイズは層の出力と同じである。
*/
void layer::calc_dL_dx_for_before() {
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 1; input < num_inputs + 1; ++input) {
				dL_dx_for_before[row] += weights[row][input] * outputs[row] * (1 - outputs[row]) * dL_dx[row];
		}
	}
}

/**
* 穴埋めポイント
* update_weights()
* この関数では層の重みの更新を行う。
* weights[][]を更新する。
* Hint
* 学習率はepsilonに入っている。
* 層に対する入力はinput[]に入っている。入力の数はnum_inputs + 1個(バイアス分)
* 層の出力はoutputs[]に入っている。出力の数はnum_rowsで定義されている。
* 重みはweights[][]に入っている。添え字はweights[出力][入力]の順である。
* 前の層によって計算されたdL/dxはdL_dx[]に入っている。dL_dxのサイズは層の出力と同じである。
*/
void layer::update_weights() {
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 1; input < num_inputs + 1; ++input) {
			weights[row][input] -= epsilon * inputs[input] * outputs[row] * (1 - outputs[row]) * dL_dx[row];
		}
	}
}

/**
* 穴埋めポイント
* pool_errors()
* この関数では一括学習のための重みに対する更新量の合計を計算する。
* 重みの更新量の合計をsum_errors_for_patch[][]に入力する。
* Hint
* 添え字はsum_errors_for_patch[出力][入力]の順である。
* 学習率はepsilonに入っている。
* 層に対する入力はinput[]に入っている。入力の数はnum_inputs + 1個(バイアス分)
* 層の出力はoutputs[]に入っている。出力の数はnum_rowsで定義されている。
* 前の層によって計算されたdL/dxはdL_dx[]に入っている。dL_dxのサイズは層の出力と同じである。
*/
void layer::pool_errors() {
	for (int row = 0; row < num_rows; ++row) {
		for (int input = 0; input < num_inputs + 1; ++input) {
			sum_errors_for_patch[row][input] = epsilon * dL_dx[row] / (num_inputs + 1);
		}
	}
}

/**
* 穴埋めポイント
* update_weights_for_patc()
* この関数では一括学習の時に重みの更新を行う。
* 重みの更新量の合計であるsum_errors_for_patch[][]を用いて重みweight()の更新を行う。
* Hint
* 添え字はsum_errors_for_patch[出力][入力]の順である。
* 重みはweights[][]に入っている。添え字はweights[出力][入力]の順である。
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
