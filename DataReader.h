#include <map>
#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <vector>

using namespace std;
using std::string;
using std::cerr;
using std::ifstream;
using std::ios;
using std::cout;
using std::endl;

class DataReader {

	public:
		DataReader(char* option_file);
        void RemoveData();
        void ReadData(char* train_file, char* test_file, char* label_file);
        void ReadDevData(char* dev_file, char* label_file);
		vector< vector<double> >* getTrainData();
		vector< vector<double> >* getDevData();
		vector< vector<double> >* getTestData();
		vector<int> getLabTrain();
		vector<int> getLabDev();
		void SetParameters();
	
	private:
		map<string, double> options;
		vector< vector<double> > train_data;
		vector< vector<double> > dev_data;
		vector< vector<double> > test_data;
		vector<int> lab_train;
		vector<int> lab_dev;
};
