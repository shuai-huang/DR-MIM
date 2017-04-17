#ifndef DFE_H
#define DFE_H

#include <map>
#include <vector>
#include <Eigen/Core>
#include <GenEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>
#include <iostream>
#include "DimRed.h"
#include "global.h"


using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace Spectra;

using namespace std;

class DFE:public DimRed
{
	private:
		MatrixXd mut;
		VectorXd train_data_mean;
		VectorXd train_data_sd;
		MatrixXd fold_mat;
		map<int, double> class_statistics;
		map<int, int> class_idx;
		map<int, double> rho_initial;
		map<int, double> rho_update;
		map<int, double> rho_optimum;
		map<int, map<int, double> > rho_update_fold;

		map<int, MatrixXd> dwd_1;
		map<int, MatrixXd> dwd_2;
		MatrixXd twt_1;
		MatrixXd twt_2;
		double num_precision;	
        double eigs_tol;
		double eta;	
		
		int max_ite;
        int max_eigs_ite;
		int dfe_dim;	// the actual dfe dimension
        int centralize;
		int normalize;
		int max_dfe_dim;

        int read_initial;
        int nr_fold; // cross validation fold number
	
	public:
		DFE();
		double ComputeMI(MatrixXd data, vector<int> lab_data);
		MatrixXd ProductOpenBLASBi(MatrixXd A, MatrixXd B);
		MatrixXd ProductOpenBLASTri(MatrixXd A, MatrixXd B, MatrixXd C);
		virtual void PerformTrainProcess();
		virtual void PerformTrain();
        virtual void PerformDev();
		virtual void PerformTest();
		virtual void setParameters();
		virtual ~DFE();
};

#endif // DFE_H
