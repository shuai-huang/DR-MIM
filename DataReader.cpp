#include "DataReader.h"
#include "global.h"

DataReader::DataReader(char* option_file) {
	// read option files and reset the parameter values
	ifstream read_option;
	read_option.open(option_file);
	string opt_tmp, opt_name;
	double opt_val;
	int opt_idx=0;
	while(read_option>>opt_tmp){
		if(opt_idx==0){opt_name=opt_tmp; opt_idx++;}
		else{opt_val=atof(opt_tmp.c_str()); opt_idx=0;}
		
		if(opt_idx==0) {options[opt_name]=opt_val;}
	}
	read_option.close();
}

void DataReader::SetParameters() {

	// reset the parameters
	if (options.find("use_dev")!=options.end()) {use_dev=(int)options["use_dev"];}
	if (options.find("num_tsf_dim")!=options.end()) {num_tsf_dim=(int)options["num_tsf_dim"];}

	cout<<"Setting parameters finished!"<<endl;
    cout<<use_dev<<" "<<num_tsf_dim<<endl; 
}

void DataReader::ReadData(char* train_file, char* test_file, char* label_file) {
	
	// read data files and save it in a 2-dimensional vector mat
	ifstream read_data;
	
    read_data.open(train_file);
	istringstream istr;
	string str;
	vector<double> tmpvec;
	int line_num=0;
	while(getline(read_data,str)){
		istr.str(str);
		double tmp;
		while(istr>>tmp){
			tmpvec.push_back(tmp);
		}
        train_data.push_back(tmpvec);
		
		tmpvec.clear();
		istr.clear();
		line_num++;
	}
	read_data.close();

	num_train=line_num;	// set num_train
    cout<<"Number of training data: "<<num_train<<endl;
    num_dim = train_data[0].size();

	// read label file and save it in a 1-dimensional vector label
	ifstream read_label;
	read_label.open(label_file);
	while(read_label>>str){
		lab_train.push_back(atoi(str.c_str()));
	}
	read_label.close();
	cout<<"Read Training data finished!"<<endl;

    ifstream read_testdata;
    read_testdata.open(test_file);
    line_num=0;
    while(getline(read_testdata,str)){
        	istr.str(str);
		double tmp;
		while(istr>>tmp){
			tmpvec.push_back(tmp);
		}
        test_data.push_back(tmpvec);
		
		tmpvec.clear();
		istr.clear();
		line_num++;
	}
    read_testdata.close();
    num_test=line_num;
    cout<<"Number of test data: "<<num_test<<endl;
	cout<<"Read Test data finished!"<<endl;
}

void DataReader::ReadDevData(char* dev_file, char* label_file) {
	
	// read data files and save it in a 2-dimensional vector mat
	ifstream read_data;
	
    read_data.open(dev_file);
	istringstream istr;
	string str;
	vector<double> tmpvec;

    int line_num=0;
	while(getline(read_data,str)){
		istr.str(str);
		double tmp;
		while(istr>>tmp){
			tmpvec.push_back(tmp);
		}
        dev_data.push_back(tmpvec);
		
		tmpvec.clear();
		istr.clear();
        line_num++;
	}
	read_data.close();
    num_dev = line_num;
    cout<<"Number of dev data: "<<num_dev<<endl;

	// read label file and save it in a 1-dimensional vector label
	ifstream read_label;
	read_label.open(label_file);
	while(read_label>>str){
		lab_dev.push_back(atoi(str.c_str()));
	}
	read_label.close();
	
	cout<<"Read Dev data finished!"<<endl;
}

vector< vector<double> >* DataReader::getTrainData() { return &train_data; }
vector< vector<double> >* DataReader::getDevData() {return &dev_data; }
vector< vector<double> >* DataReader::getTestData() { return &test_data; }
vector<int> DataReader::getLabTrain() {return lab_train; }
vector<int> DataReader::getLabDev() {return lab_dev; }

void DataReader::RemoveData() {
    train_data.clear();
    dev_data.clear();
    test_data.clear();
}
