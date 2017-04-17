#include <cstdio>
#include <string>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>

#include "global.h"
#include "DimRed.h"
#include "DFE.h"

using namespace std;
using namespace Eigen;

using std::ifstream;
using std::string;

void process_cmdline(char *argv[], char **train_file, char **test_file, char **option_file, char **label_train_file,
	char **train_red_file, char **test_red_file, char **tsf_file) {
	
	*train_file	= argv[1];
	*test_file	= argv[2];
	*option_file	= argv[3];
	*label_train_file = argv[4];
	*train_red_file	= argv[5];
	*test_red_file	= argv[6];
    *tsf_file = argv[7];
}

void process_cmdline_dev(char *argv[], char **train_file, char **dev_file, char **test_file, char **option_file, char **label_train_file, char **label_dev_file,
	char **train_red_file, char **dev_red_file, char **test_red_file, char **tsf_file) {
	
	*train_file	= argv[1];
	*dev_file = argv[2];
	*test_file	= argv[3];
    *option_file	= argv[4];
	*label_train_file = argv[5];
	*label_dev_file = argv[6];
	*train_red_file	= argv[7];
	*dev_red_file = argv[8];
	*test_red_file = argv[9];
    *tsf_file = argv[10];
}


int main(int argc, char *argv[]){

	char *train_file, *dev_file, *test_file, *option_file, *label_train_file, *label_dev_file, *train_red_file, *dev_red_file, *test_red_file, *tsf_file;
	if (argc==8) {
		process_cmdline(argv, &train_file, &test_file, &option_file, &label_train_file, &train_red_file, &test_red_file, &tsf_file);
	} else if (argc==11)  {	// use dev dataset
		process_cmdline_dev(argv, &train_file, &dev_file, &test_file, &option_file, &label_train_file, &label_dev_file, &train_red_file, &dev_red_file, &test_red_file, &tsf_file);
	} else {
		cout<<"Command line error!"<<endl;
	}

	DataReader data_input(option_file);
	data_input.SetParameters();
    data_input.ReadData(train_file, test_file, label_train_file);
    if (use_dev==1) {data_input.ReadDevData(dev_file, label_dev_file);}

	DimRed *dim_red;
    dim_red = new DFE();
    dim_red->setParameters();
    dim_red->setTrainData(num_train, num_dim, &data_input, data_input.getLabTrain());
    if (use_dev==1) {dim_red->setDevData(num_dev, num_dim, &data_input, data_input.getLabDev());}
    dim_red->setTestData(num_test, num_dim, &data_input);
    dim_red->PerformTrainProcess();
	
    // remove data saved in DataReader
    data_input.RemoveData();
    
    cout<<"Finding "<<num_tsf_dim<<" tsf dimensions"<<endl;
    MatrixXd tsf_mat;
    for (int i=0; i<num_tsf_dim; i++) {
    	cout<<"tsf_dim: "<<i<<endl;
    	if (i==0) {
    		dim_red->PerformTrain();
            tsf_mat = dim_red->getTsf();
    	} else {
			dim_red->setTsfOld(tsf_mat);
			dim_red->PerformTrain();
            MatrixXd tsf_mat_new = dim_red->getTsf();
            MatrixXd tsf_mat_old = dim_red->getTsfOld();
            MatrixXd tsf_mat_tmp(tsf_mat_old.rows(), tsf_mat_old.cols()+tsf_mat_new.cols());
            tsf_mat_tmp.block(0,0, tsf_mat_old.rows(), tsf_mat_old.cols()) = tsf_mat_old;
            tsf_mat_tmp.block(0, tsf_mat_old.cols(), tsf_mat_new.rows(), tsf_mat_new.cols()) = tsf_mat_new;
            tsf_mat = tsf_mat_tmp;
    	}
    	
    	// output the tsf matrix
		cout<<tsf_mat.rows()<<" "<<tsf_mat.cols()<<endl;
		ofstream write_tsf_mat(tsf_file, ios_base::trunc);
		for (int i=0; i<tsf_mat.rows(); i++) {
		    for (int j=0; j<tsf_mat.cols(); j++) {
		        write_tsf_mat<<tsf_mat(i,j)<<" ";
		    }
		    write_tsf_mat<<"\n";
		}
		write_tsf_mat.close();
    }
    
    // perform dev and test reduction
    if (use_dev==1) {dim_red->PerformDev();}
    dim_red->PerformTest();

	MatrixXd train_data_red=dim_red->getTrainRed();
	MatrixXd dev_data_red;
	if (use_dev==1) {dev_data_red=dim_red->getDevRed();}
	MatrixXd test_data_red=dim_red->getTestRed();

    ofstream write_result(train_red_file, ios_base::trunc);
    for (int i=0; i<train_data_red.rows(); i++) {
        for (int j=0; j<train_data_red.cols(); j++) {
            write_result<<train_data_red(i,j)<<" ";
        }
        write_result<<"\n";
    }
    write_result.close();

	if (use_dev==1) {
		ofstream write_dev_result(dev_red_file, ios_base::trunc);
		for (int i=0; i<dev_data_red.rows(); i++) {
		    for (int j=0; j<dev_data_red.cols(); j++) {
		        write_dev_result<<dev_data_red(i,j)<<" ";
		    }
		    write_dev_result<<"\n";
		}
		write_dev_result.close();
    }

    cout<<test_data_red.rows()<<" "<<test_data_red.cols()<<endl;
    ofstream write_test_result(test_red_file, ios_base::trunc);
    for (int i=0; i<test_data_red.rows(); i++) {
        for (int j=0; j<test_data_red.cols(); j++) {
            write_test_result<<test_data_red(i,j)<<" ";
        }
        write_test_result<<"\n";
    }
    write_test_result.close();
    
    delete dim_red;

	exit(0);
}
