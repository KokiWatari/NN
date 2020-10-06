
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
	// ユーザ入力部
	int soshi,sou,is_online;
	cout << "素子数:";
	cin >> soshi;
	cout << "層数:";
	cin >> sou;
	cout << "一括学習:0, 逐次学習:1" << endl <<":";
	cin >> is_online;
	// データ読み取り部
	vector<vector<double>> input_data = get_vector_from_file("data.csv");
	vector<vector<double>> output_data = get_vector_from_file("data_T.csv");
	vector<vector<double>> test_input_data = get_vector_from_file("data_test.csv");
	vector<vector<double>> test_output_data = get_vector_from_file("data_test_T.csv");

	// ここからNN本体
	// ManageLayer クラスの第4引数が学習率だから変えたければいじって
	ManageLayer ml(sou, soshi, input_data[0].size(), output_data[0].size() , 0.1);
	if (is_online) {
		ml.online(input_data,output_data);
	}
	else {
		ml.patch(input_data,output_data);
	}
	/**
	* 穴埋めポイント
	* forword()を使って出力結果を出す。
	* 教師データと未学習データの出力結果を出力する。
	*/
	
	
	vector<vector<double>> output(output_data.size(), vector<double>(output_data[0].size()));
	printf("学習データの出力結果\n");
	for (int i = 0; i < input_data.size();i++) {
		output[i] = ml.forword(input_data[i]);
		printf("%lf\n", output[i][0]);
	}
	
	vector<vector<double>> test_output(test_output_data.size(), vector<double>(test_output_data[0].size()));
	printf("未学習データの出力結果\n");
	for (int i = 0; i < test_input_data.size(); i++) {
		test_output[i] = ml.forword(test_input_data[i]);
		printf("%lf\n", test_output[i][0]);
	}
	
	
	

	return 0;
}
vector<vector<double>> get_vector_from_file(string filename) {
	ifstream ifs(filename);
	if (ifs.fail()) {
		std::cerr << "失敗" << std::endl;
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