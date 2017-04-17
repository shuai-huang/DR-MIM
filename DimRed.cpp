#include "DimRed.h"

DimRed::DimRed() {
    no_train = 0;
    no_dev=0;
    no_test=0;
    no_dim=0;
    set_tsf=false;
}

void DimRed::setTrainData(int no_row, int no_col, DataReader* data_tmp, vector<int> lab_train) {

	MatrixXd tmp(no_row, no_col);
	vector< vector<double> > *mat_tmp = data_tmp->getTrainData();
	for (int i=0; i<no_row; i++) {
		for (int j=0; j<no_col; j++) {
			tmp(i,j)=(*mat_tmp)[i][j];
		}
	}
	
	train_data = tmp;
	(*this).lab_train = lab_train;
	no_train = no_row;
	no_dim = no_col;
	cout<<"Reading Training data finished!"<<endl;
}

void DimRed::setDevData(int no_row, int no_col, DataReader* data_tmp, vector<int> lab_dev) {

	MatrixXd tmp(no_row, no_col);
	vector< vector<double> > *mat_tmp = data_tmp->getDevData();
	for (int i=0; i<no_row; i++) {
		for (int j=0; j<no_col; j++) {
			tmp(i,j)=(*mat_tmp)[i][j];
		}
	}
	
	dev_data = tmp;
	(*this).lab_dev = lab_dev;
	no_dev=no_row;
	cout<<"Reading Dev data finished!"<<endl;
}

void DimRed::setTestData( int no_row, int no_col, DataReader* data_tmp) {

	MatrixXd tmp(no_row, no_col);
	vector< vector<double> > *mat_tmp = data_tmp->getTestData();
	for (int i=0; i<no_row; i++) {
		for (int j=0; j<no_col; j++) {
			tmp(i,j)=(*mat_tmp)[i][j];
		}
	}
	
	test_data = tmp;
	no_test = test_data.rows();
	cout<<"Reading Test data finished!"<<endl;
}

void DimRed::setTsfOld(MatrixXd dfe_tsf_old) {
	(*this).dfe_tsf_old=dfe_tsf_old;
	set_tsf=true;
}

void DimRed::setMaxDim(int max_dim) {(*this).max_dim = max_dim;}

void DimRed::PerformTrainProcess() {}
void DimRed::PerformTrain() {train_data_red = train_data;}
void DimRed::PerformDev() {dev_data_red = dev_data;}
void DimRed::PerformTest() {test_data_red = test_data;}
void DimRed::setParameters() {}

MatrixXd DimRed::getTrainRed() { return train_data_red; }
MatrixXd DimRed::getDevRed() {return dev_data_red;}
MatrixXd DimRed::getTestRed() {return test_data_red; }
MatrixXd DimRed::getTsf() {return dfe_tsf; }
MatrixXd DimRed::getTsfOld() {return dfe_tsf_old;}

DimRed::~DimRed() {}
