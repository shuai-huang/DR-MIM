#ifndef DIMRED_H
#define DIMRED_H

#include <Eigen/Dense>
#include <vector>

#include "DataReader.h"

using Eigen::MatrixXd;

using namespace std;

class DimRed
{
    protected:
        MatrixXd train_data;
        MatrixXd dev_data;
        MatrixXd test_data;
        vector<int> lab_train;
        vector<int> lab_dev;
        vector<int> lab_test;

    public:
    	int no_train;
        int no_dev;
    	int no_test;
    	int no_dim;
    	bool set_tsf;
    	int max_dim;
    	
    	MatrixXd train_data_red;
    	MatrixXd dev_data_red;
    	MatrixXd test_data_red;
    	
        MatrixXd dfe_tsf;
        MatrixXd dfe_tsf_old;

    public:
        DimRed();
        void setTrainData(int no_row, int no_col, DataReader* data_tmp, vector<int> lab_train);
        void setDevData(int no_row, int no_col, DataReader* data_tmp, vector<int> lab_dev);
        void setTestData(int no_row, int no_col, DataReader* data_tmp);
        void setTsfOld(MatrixXd dfe_tsf_old);
        void setMaxDim(int max_dim);
        MatrixXd getTrainRed();
        MatrixXd getDevRed();
        MatrixXd getTestRed();
        MatrixXd getTsf();
        MatrixXd getTsfOld();

		virtual void PerformTrainProcess();
        virtual void setParameters();
        virtual void PerformTrain();
        virtual void PerformDev();
        virtual void PerformTest();
        virtual ~DimRed();
        
};

#endif // DIMRED_H
