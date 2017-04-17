#include "DFE.h"
#include <cblas.h>
#include <math.h>
#include <fstream>
#define BUFSIZE 256

// Could remove the columns that are all zeros and speed up the computation

DFE::DFE(){}

void DFE::setParameters() {

	map<string, double> dfe_options;
	ifstream read_option;
	read_option.open("dfe.options");
	string opt_tmp, opt_name;
	double opt_val;
	int opt_idx=0;
	while(read_option>>opt_tmp){
		if(opt_idx==0){opt_name=opt_tmp; opt_idx++;}
		else{opt_val=atof(opt_tmp.c_str()); opt_idx=0;}
		
		if(opt_idx==0) {dfe_options[opt_name]=opt_val;}
	}
	read_option.close();
	
    if (dfe_options.find("max_dim")!=dfe_options.end()) { max_dim=(int)dfe_options["max_dim"];}
    if (dfe_options.find("read_initial")!=dfe_options.end()) { read_initial=(int)dfe_options["read_initial"]; }
    if (dfe_options.find("centralize")!=dfe_options.end()) { centralize=(int)dfe_options["centralize"];}
    if (dfe_options.find("normalize")!=dfe_options.end()) { normalize=(int)dfe_options["normalize"];}
    if (dfe_options.find("eta")!=dfe_options.end()) { eta=dfe_options["eta"];}
    if (dfe_options.find("max_ite")!=dfe_options.end()) { max_ite = (int)dfe_options["max_ite"];}
    if (dfe_options.find("max_eigs_ite")!=dfe_options.end()) { max_eigs_ite = (int)dfe_options["max_eigs_ite"];}
    if (dfe_options.find("eigs_tol")!=dfe_options.end()) { eigs_tol =dfe_options["eigs_tol"];}
    if (dfe_options.find("nr_fold")!=dfe_options.end()) { nr_fold = (int)dfe_options["nr_fold"];}
    if (dfe_options.find("num_precision")!=dfe_options.end()) { num_precision = dfe_options["num_precision"];}
    
    // if the initial values of rho is provided, read them in
    ifstream read_rho;
    read_rho.open("rho_initial.val");
    istringstream istr;
    string str;
    while(getline(read_rho, str)) {
    	istr.str(str);
    	double tmp;
    	int idx_tmp=0;
    	int name_tmp;
    	double value_tmp;
    	while (istr>>tmp){
    		if (idx_tmp==0) {
    			name_tmp=tmp;
    		} else {
    			value_tmp=tmp;
    		}
    	}
    	rho_initial[name_tmp]=value_tmp;
    }
    read_rho.close();

}

double DFE::ComputeMI(MatrixXd data, vector<int> lab_data) {

	int no_data=data.rows();
	map<int, double> class_statistics;
	for (int i=0; i<lab_data.size(); i++) {
		if (class_statistics.find(lab_data[i])!=class_statistics.end()) {
			class_statistics[lab_data[i]]++;
		} else { class_statistics[lab_data[i]]=1.0;}
	}
	map<int, double>::iterator it_c;
	double gamma_diff = 1.0/(no_data-1);
	for (it_c=class_statistics.begin(); it_c!=class_statistics.end(); it_c++) {
		int class_tmp=it_c->first;
		double num_class=it_c->second;
		class_statistics[class_tmp]=gamma_diff - 1.0/(num_class-1);
	}
	
	double mi=0;
	for (int i=0; i<lab_data.size(); i++) {
		for (int j=i+1; j<lab_data.size(); j++) {
			//double ij_norm = (((data.row(i)-data.row(j)).array().pow(2)+num_precision).log()).sum();
            double ij_norm = log(((data.row(i)-data.row(j)).array().pow(2)+num_precision).sum());
			if (lab_data[i]==lab_data[j] ) {
				mi+=class_statistics[lab_data[i]]*ij_norm;
			} else {mi+=gamma_diff*ij_norm;}
		}
	}
	return mi;
}

MatrixXd DFE::ProductOpenBLASBi(MatrixXd A, MatrixXd B) {

	int m=A.rows(), k=A.cols(), n=B.cols();
	double *A_mem , *B_mem , *C_mem;

	// have to use malloc to avoid conflict 
	A_mem = (double *) malloc( sizeof(double) * m * k );
	B_mem = (double *) malloc( sizeof(double) * k * n );
	C_mem = (double *) malloc( sizeof(double) * m * n );
	
	int idx_tmp=0;
	for (int i=0; i<k; i++) {
		for (int j=0; j<m; j++) {
			A_mem[idx_tmp]=A(j,i);
			idx_tmp++;
		}
	}
	
	idx_tmp=0;
	for (int i=0; i<n; i++) {
		for (int j=0; j<k; j++) {
			B_mem[idx_tmp]=B(j,i);
			idx_tmp++;
		}
	}

	cblas_dgemm ( CblasColMajor, CblasNoTrans, CblasNoTrans , m , n , k , 1.0 , A_mem , m , B_mem , k , 0.0 , C_mem , m );
	MatrixXd C(m,n);

	idx_tmp=0;
	for (int i=0; i<n; i++) {
		for (int j=0; j<m; j++) {
			C(j,i)=C_mem[idx_tmp];
			idx_tmp++;
		}
	}
	free(A_mem);
	free(B_mem);
	free(C_mem);
	return C;
}

MatrixXd DFE::ProductOpenBLASTri(MatrixXd A, MatrixXd B, MatrixXd C) {

	int m=A.rows(), k=A.cols(), n=B.cols(), l=C.cols();
	double *A_mem , *B_mem , *AB_mem, *C_mem, *D_mem;

	A_mem = (double *) malloc( sizeof(double) * m * k );
	B_mem = (double *) malloc( sizeof(double) * k * n );
	AB_mem = (double *) malloc( sizeof(double) * m * n );
	C_mem = (double *) malloc( sizeof(double) * n * l );
	D_mem = (double *) malloc( sizeof(double) * m * l );
	
	int idx_tmp=0;
	for (int i=0; i<k; i++) {
		for (int j=0; j<m; j++) {
			A_mem[idx_tmp]=A(j,i);
			idx_tmp++;
		}
	}
	
	idx_tmp=0;
	for (int i=0; i<n; i++) {
		for (int j=0; j<k; j++) {
			B_mem[idx_tmp]=B(j,i);
			idx_tmp++;
		}
	}
	
	idx_tmp=0;
	for (int i=0; i<l; i++) {
		for (int j=0; j<n; j++) {
			C_mem[idx_tmp]=C(j,i);
			idx_tmp++;
		}
	}

	cblas_dgemm ( CblasColMajor, CblasNoTrans, CblasNoTrans , m , n , k , 1.0 , A_mem , m , B_mem , k , 0.0 , AB_mem , m );
	cblas_dgemm ( CblasColMajor, CblasNoTrans, CblasNoTrans , m , l , n , 1.0 , AB_mem , m , C_mem , n , 0.0 , D_mem , m );
	MatrixXd D(m,l);
	
	idx_tmp=0;
	for (int i=0; i<l; i++) {
		for (int j=0; j<m; j++) {
			D(j,i)=D_mem[idx_tmp];
			idx_tmp++;
		}
	}
	free(A_mem);
	free(B_mem);
	free(AB_mem);
	free(C_mem);
	free(D_mem);
	return D;
}

void DFE::PerformTrainProcess() {

	// Preprocess data if necessary
	// subtracted the mean
    if (centralize==1) {
    	train_data_mean = train_data.colwise().sum() / no_train;
	    for (int i=0; i<no_dim; i++) {
		    train_data.col(i)= train_data.col(i).array()-train_data_mean(i);
	    }
    }
	// divided by the sd
	if (normalize==1) {
		if (centralize==0) {	// compute train_data_mean if not done
			train_data_mean = train_data.colwise().sum() / no_train;
		}
		MatrixXd train_data_cen = train_data;
		for (int i=0; i<train_data_cen.cols(); i++) {
			train_data_cen.col(i)= train_data_cen.col(i).array() - train_data_mean(i);
		}
		train_data_sd = (train_data_cen.array()*train_data_cen.array()).matrix().colwise().sum() / no_train;
		train_data_sd = train_data_sd.array().sqrt();
		for (int i=0; i<no_dim; i++) {
            if (train_data_sd(i)!=0) {
			train_data.col(i)=train_data.col(i)/train_data_sd(i);
            }
		}
	}

	// Compute the class statistics
	// class_idx is where each class starts, so the data should be stored sequentially according to class labels, strict orders
	map<int, double>::iterator it_c;
	int idx_tmp;
	for (int i=0; i<lab_train.size(); i++) {
		if (class_statistics.find(lab_train[i])!=class_statistics.end()) {
			class_statistics[lab_train[i]]++;
		} else { class_statistics[lab_train[i]]=1.0; class_idx[lab_train[i]]=i;}
	}

	// set fold_mat
	//Random shuffling the dataset within each class should be done prior to running the program
	MatrixXd fold_mat_tmp(class_statistics.size(), nr_fold+1); // the index where each fold starts
	idx_tmp=0;
	for (it_c=class_statistics.begin(); it_c!=class_statistics.end(); it_c++) {
		int class_tmp=it_c->first, class_num_tmp= (int)it_c->second;
		int num_base = class_num_tmp/nr_fold, num_res = class_num_tmp%nr_fold;
		fold_mat_tmp(idx_tmp,0)=class_idx[class_tmp];
		for (int i=1; i<=nr_fold; i++) {
			if (i<=num_res) {
				fold_mat_tmp(idx_tmp,i)=fold_mat_tmp(idx_tmp,i-1)+num_base+1;
			} else {
				fold_mat_tmp(idx_tmp,i)=fold_mat_tmp(idx_tmp,i-1)+num_base;
			}
		}
		idx_tmp++;
	}
	fold_mat = fold_mat_tmp;

    // set mut to identity matrix
    mut=MatrixXd::Identity(no_dim, no_dim);

    // set dwd_1 and dwd_2
    cout<<"set dwd_1 and dwd_2"<<endl;
	for (int fold_num=0; fold_num<nr_fold; fold_num++) {
		int num_lo=0;
		for (int j=0; j<class_statistics.size(); j++) {
			num_lo=num_lo+fold_mat(j,fold_num+1)-fold_mat(j,fold_num);
		}
		MatrixXd data(no_train-num_lo, no_dim), data_lo(num_lo, no_dim);
		vector<int> lab_data, lab_data_lo;
		int data_idx=0, data_lo_idx=0;
		for (int j=0; j<class_statistics.size(); j++) {
			for (int k=0; k<nr_fold; k++) {
				for (int l=fold_mat(j,k); l<fold_mat(j,k+1); l++) {
					if (fold_num==k) {
						lab_data_lo.push_back(lab_train[l]); 
						data_lo.row(data_lo_idx)=train_data.row(l);
						data_lo_idx++;
					} else {
						lab_data.push_back(lab_train[l]);
						data.row(data_idx)=train_data.row(l);
						data_idx++;
					}
				}
			}
		}

		int no_data = data.rows();
		cout<<"NO_data: "<<no_data<<" "<<data.cols()<<" "<<lab_data.size()<<endl;
		map<int, double> class_data_statistics;
		map<int, int> class_data_idx;
		for (int i=0; i<lab_data.size(); i++) {
			if (class_data_statistics.find(lab_data[i])!=class_data_statistics.end()) {
				class_data_statistics[lab_data[i]]++;
			} else { class_data_statistics[lab_data[i]]=1.0; class_data_idx[lab_data[i]]=i;}
		}

		map<int, double> class_data_coefficient = class_data_statistics;
		for (it_c=class_data_coefficient.begin(); it_c!=class_data_coefficient.end(); it_c++) {
			int class_tmp=it_c->first;
			double val_tmp=it_c->second;
			class_data_coefficient[class_tmp]= 1.0/2/no_data/(val_tmp-1);
		}
		
		MatrixXd W=MatrixXd::Constant(no_data, no_data, 1.0/2/no_data/(no_data-1));
		VectorXd W_sum = W.colwise().sum();
		W=-W;
		for (int i=0; i<no_data; i++) {W(i,i)=W_sum(i)+W(i,i);}
		dwd_1[fold_num] = ProductOpenBLASTri(data.transpose(), W, data);

		MatrixXd dwd_2_tmp=MatrixXd::Zero(no_dim, no_dim);
		for (it_c=class_data_statistics.begin(); it_c!=class_data_statistics.end(); it_c++) {
			int class_tmp=it_c->first, class_num_tmp= (int)it_c->second;
			int class_start = class_data_idx[class_tmp];
			MatrixXd data_tmp = data.block(class_start,0,class_num_tmp, no_dim);
			MatrixXd W_tmp = MatrixXd::Constant(class_num_tmp, class_num_tmp, class_data_coefficient[class_tmp]);
			VectorXd W_tmp_sum = W_tmp.colwise().sum();
			W_tmp=-W_tmp;
			for (int i=0; i<class_num_tmp; i++) {W_tmp(i,i)=W_tmp_sum(i)+W_tmp(i,i);}
			dwd_2_tmp = dwd_2_tmp+ProductOpenBLASTri(data_tmp.transpose(), W_tmp, data_tmp);
		}
		dwd_2[fold_num] = dwd_2_tmp;
	}

	// set twt_1 and twt_2
	cout<<"set twt_1 and twt_2"<<endl;
	MatrixXd W=MatrixXd::Constant(no_train, no_train, 1.0/2/no_train/(no_train-1));
	VectorXd W_sum = W.colwise().sum();
	W=-W;
	for (int i=0; i<no_train; i++) {W(i,i)=W_sum(i)+W(i,i);}
	twt_1 = ProductOpenBLASTri(train_data.transpose(), W, train_data);
	
	map<int, double> class_coefficient = class_statistics;
	for (it_c=class_coefficient.begin(); it_c!=class_coefficient.end(); it_c++) {
		int class_tmp=it_c->first;
		double val_tmp=it_c->second;
		class_coefficient[class_tmp]= 1.0/2/no_train/(val_tmp-1);
	}

	MatrixXd twt_2_tmp=MatrixXd::Zero(no_dim, no_dim);
	for (it_c=class_statistics.begin(); it_c!=class_statistics.end(); it_c++) {
		int class_tmp=it_c->first, class_num_tmp= (int)it_c->second;
		int class_start = class_idx[class_tmp];
		MatrixXd train_data_tmp = train_data.block(class_start,0,class_num_tmp, no_dim);
		MatrixXd W_tmp = MatrixXd::Constant(class_num_tmp, class_num_tmp, class_coefficient[class_tmp]);
		VectorXd W_tmp_sum = W_tmp.colwise().sum();
		W_tmp=-W_tmp;
		for (int i=0; i<class_num_tmp; i++) {W_tmp(i,i)=W_tmp_sum(i)+W_tmp(i,i);}
		twt_2_tmp = twt_2_tmp+ProductOpenBLASTri(train_data_tmp.transpose(), W_tmp, train_data_tmp);
	}
	twt_2 = twt_2_tmp;
	
}


void DFE::PerformTrain(){

	map<int, double>::iterator it_c;
	// if read_initial is 0, initialize rho here
	if (read_initial==0) {
        // set the rho to 1, i.e. at the start mutual information is zero, only this is fair
        rho_initial[-1]=1;
        for (it_c=class_statistics.begin(); it_c!=class_statistics.end(); it_c++) {
            int class_tmp=it_c->first;
            rho_initial[class_tmp] = 1;
        }
	}


	// speed up by openblas
	// create mut matrix using dfe_tsf_old
	map<int, MatrixXd> dwd_1_mut=dwd_1;
	map<int, MatrixXd> dwd_2_mut=dwd_2;
	if (set_tsf) {
		int i_tmp = dfe_tsf_old.cols()-1;
		mut=mut-ProductOpenBLASBi(dfe_tsf_old.col(i_tmp), dfe_tsf_old.col(i_tmp).transpose());
		for (int fold_num=0; fold_num<nr_fold; fold_num++) { 
			dwd_1_mut[fold_num] = ProductOpenBLASBi(dwd_1_mut[fold_num], mut);
			dwd_2_mut[fold_num] = ProductOpenBLASBi(dwd_2_mut[fold_num], mut);
		}
		
	}

	vector<double> mi_vect;
	
    map<int, MatrixXd> data_red_dist_old;
    MatrixXd train_data_red_dist_old;
	
	int num_ite=0;
	while (num_ite<max_ite) {
		
		int einfo_fold = 1;
		vector<double> mi_fold_vect;
		for (int fold_num=0; fold_num<nr_fold; fold_num++) {
		    cout<<"Fold: "<<fold_num<<endl;
			int num_lo=0;
			for (int j=0; j<class_statistics.size(); j++) {
				num_lo=num_lo+fold_mat(j,fold_num+1)-fold_mat(j,fold_num);
			}
			MatrixXd data(no_train-num_lo, no_dim), data_lo(num_lo, no_dim);
			vector<int> lab_data, lab_data_lo;
			int data_idx=0, data_lo_idx=0;
			for (int j=0; j<class_statistics.size(); j++) {
				for (int k=0; k<nr_fold; k++) {
					for (int l=fold_mat(j,k); l<fold_mat(j,k+1); l++) {
						if (fold_num==k) {
							lab_data_lo.push_back(lab_train[l]); 
							data_lo.row(data_lo_idx)=train_data.row(l);
							data_lo_idx++;
						} else {
							lab_data.push_back(lab_train[l]);
							data.row(data_idx)=train_data.row(l);
							data_idx++;
						}
					}
				}
			}

			int no_data = data.rows();
			cout<<"NO_data: "<<no_data<<" "<<data.cols()<<" "<<lab_data.size()<<endl;
			map<int, double> class_data_statistics;
			map<int, int> class_data_idx;
			for (int i=0; i<lab_data.size(); i++) {
				if (class_data_statistics.find(lab_data[i])!=class_data_statistics.end()) {
					class_data_statistics[lab_data[i]]++;
				} else { class_data_statistics[lab_data[i]]=1.0; class_data_idx[lab_data[i]]=i;}
			}

			// use previously computed results, since who_update is the same for all classes, we just choose rho_update[1] here
			// initialize rho_update
			if (num_ite==0) {
				rho_update = rho_initial;
			} else {
				rho_update = rho_update_fold[fold_num];
			}

            cout<<rho_update[-1]<<" "<<rho_update[1]<<endl;
			MatrixXd W=rho_update[-1]*dwd_1_mut[fold_num]-rho_update[1]*dwd_2_mut[fold_num];
			// speed up by openblas
			W.transposeInPlace();

            int ncv = (2*max_dim) > no_dim ? no_dim : 2*max_dim;

            // Construct matrix operation object using the wrapper class
            DenseGenMatProd<double> op(W);
            // Construct eigen solver object, requesting the largest three eigenvalues and corresponding eigenvalues
            GenEigsSolver< double, LARGEST_REAL, DenseGenMatProd<double> > eigs(&op, max_dim, ncv);

            // Initialize and compute
            eigs.init();
            int conv_marker = eigs.compute(max_eigs_ite, eigs_tol, LARGEST_REAL);

            // Retrieve results
            //Eigen::VectorXcd evalues;
            //evalues = eigs.eigenvalues();

            Eigen::MatrixXcd evectors;
            evectors = eigs.eigenvectors(max_dim);

            MatrixXd tsf_matrix = evectors.real();

			double mi_tmp = ComputeMI(data_lo*tsf_matrix, lab_data_lo);
			mi_fold_vect.push_back(mi_tmp);
			
			// compute new projections and rho_update_fold_tmp
			MatrixXd data_red = data*tsf_matrix;
			
			VectorXd data_red_sum = (data_red.array()*data_red.array()).matrix().rowwise().sum();
			MatrixXd data_red_sum_mat(no_data, no_data);
			for (int i=0; i<no_data; i++) {data_red_sum_mat.row(i)=data_red_sum;}
			MatrixXd data_red_dist = data_red_sum_mat + data_red_sum_mat.transpose()-2*data_red*data_red.transpose();
            //data_red_dist=data_red_dist-2*ProductOpenBLASBi(data_red, data_red.transpose());
            
            // move a small step
            if (num_ite>0) {
                data_red_dist = data_red_dist_old[fold_num] + eta*(data_red_dist-data_red_dist_old[fold_num]);
            }
            // save data_red_dist_old before logrithm operation
            MatrixXd data_red_dist_old_tmp=data_red_dist;

			for (int i=0; i<no_data; i++) {
				for (int j=0; j<no_data; j++) {
					if (data_red_dist(i,j)<=num_precision) {
						data_red_dist(i,j)=1;
						data_red_dist_old_tmp(i,j)=0;
					}
				}
			}
			data_red_dist_old[fold_num]=data_red_dist_old_tmp;

			data_red_dist=(data_red_dist.array().log()).matrix();
			map<int, double> data_red_dist_sum;
			data_red_dist_sum[-1]=(data_red_dist.sum())/(no_data*(no_data-1.0));
            double data_red_dist_sum_class=0;
			for (it_c=class_data_statistics.begin(); it_c!=class_data_statistics.end(); it_c++) {
				int class_tmp=it_c->first, class_num_tmp= (int)it_c->second;
				int class_start = class_data_idx[class_tmp];
				data_red_dist_sum[class_tmp]=(data_red_dist.block(class_start,class_start,class_num_tmp,class_num_tmp).sum())/(no_data*(class_num_tmp-1.0));
                data_red_dist_sum_class = data_red_dist_sum_class + data_red_dist_sum[class_tmp];
			}
			
			map<int, double> rho_update_fold_tmp;
			rho_update_fold_tmp[-1]=1.0/exp((data_red_dist_sum[-1]));
			for (it_c=class_data_statistics.begin(); it_c!=class_data_statistics.end(); it_c++) {
				int class_tmp=it_c->first, class_num_tmp= (int)it_c->second;
				rho_update_fold_tmp[class_tmp] = 1.0/exp(data_red_dist_sum_class);
			}


			rho_update_fold[fold_num] = rho_update_fold_tmp;

		}
		
		if (einfo_fold==0) {break;}
		
		// save mi_vect
		double mi_vect_tmp=0;
		for (int i=0; i<nr_fold; i++) {
			cout<<mi_fold_vect[i]<<" ";
			mi_vect_tmp=mi_vect_tmp+mi_fold_vect[i];
		}
		cout<<endl;
        // stop searching if mi decrease, this is done to save time for the 20newsgroup
        bool mi_small  = false;
        if (mi_vect.size()>0) {
            if (mi_vect_tmp < mi_vect[mi_vect.size()-1]) {
                mi_small = true;
            }
        }
        cout<<"mi: "<<mi_vect_tmp<<endl;
        if (mi_small) {break;}
		mi_vect.push_back(mi_vect_tmp);
		
		
		
		/////////////////////////////////
		// Do it for the training data //
		/////////////////////////////////
		cout<<"NO_train: "<<no_train<<" "<<train_data.cols()<<endl;
		if (num_ite==0) {
			rho_optimum = rho_initial;
		}
		
		// use previously computed results
		MatrixXd W=rho_optimum[-1]*twt_1-rho_optimum[1]*twt_2;	
		// speed up by openblas
		W=ProductOpenBLASBi(W, mut);
		W.transposeInPlace();

        int ncv = (2*max_dim) > no_dim ? no_dim : 2*max_dim;

        // Construct matrix operation object using the wrapper class
        DenseGenMatProd<double> op(W);
        // Construct eigen solver object, requesting the largest three eigenvalues and corresponding eigenvalues
        GenEigsSolver< double, LARGEST_REAL, DenseGenMatProd<double> > eigs(&op, max_dim, ncv);
        
        // Initialize and compute
        eigs.init();
        int conv_marker = eigs.compute(max_eigs_ite, eigs_tol, LARGEST_REAL);
        
        // Retrieve results
        //Eigen::VectorXcd evalues;
        //evalues = eigs.eigenvalues();
        
        Eigen::MatrixXcd evectors;
        evectors = eigs.eigenvectors(max_dim);
        
        MatrixXd tsf_matrix = evectors.real();

		dfe_tsf = tsf_matrix;
		cout<<train_data.rows()<<" "<<train_data.cols()<<" "<<dfe_tsf.rows()<<" "<<dfe_tsf.cols()<<endl;
		
		// compute the transformed dataset
		train_data_red = train_data*dfe_tsf;

		VectorXd train_data_red_sum = (train_data_red.array()*train_data_red.array()).matrix().rowwise().sum();
		MatrixXd train_data_red_sum_mat(no_train, no_train);
		for (int i=0; i<no_train; i++) {train_data_red_sum_mat.row(i)=train_data_red_sum;}
		MatrixXd train_data_red_dist = train_data_red_sum_mat + train_data_red_sum_mat.transpose()-2*train_data_red*train_data_red.transpose();
        //data_red_dist=data_red_dist-2*ProductOpenBLASBi(data_red, data_red.transpose());

        // move a small step
        if (num_ite>0) {
            train_data_red_dist = train_data_red_dist_old + eta*(train_data_red_dist-train_data_red_dist_old);
        }

        // save train_data_red_dist_old before logrithm operation
        MatrixXd train_data_red_dist_old_tmp=train_data_red_dist;

		for (int i=0; i<no_train; i++) {
			for (int j=0; j<no_train; j++) {
				if (train_data_red_dist(i,j)<=num_precision) {
					train_data_red_dist(i,j)=1;
					train_data_red_dist_old_tmp(i,j)=0;
				}
			}
		}
		train_data_red_dist_old=train_data_red_dist_old_tmp;

		train_data_red_dist=(train_data_red_dist.array().log()).matrix();
		map<int, double> train_data_red_dist_sum;
		train_data_red_dist_sum[-1]=(train_data_red_dist.sum())/(no_train*(no_train-1.0));
        double train_data_red_dist_sum_class=0;
		for (it_c=class_statistics.begin(); it_c!=class_statistics.end(); it_c++) {
			int class_tmp=it_c->first, class_num_tmp= (int)it_c->second;
			int class_start = class_idx[class_tmp];
			train_data_red_dist_sum[class_tmp]=(train_data_red_dist.block(class_start,class_start,class_num_tmp,class_num_tmp).sum())/(no_train*(class_num_tmp-1.0));
            train_data_red_dist_sum_class = train_data_red_dist_sum_class + train_data_red_dist_sum[class_tmp];
		}
		
		rho_optimum[-1]=1.0/exp((train_data_red_dist_sum[-1]));
		cout<<rho_optimum[-1]<<" ";
		for (it_c=class_statistics.begin(); it_c!=class_statistics.end(); it_c++) {
			int class_tmp=it_c->first, class_num_tmp= (int)it_c->second;
			rho_optimum[class_tmp] = 1.0/exp(train_data_red_dist_sum_class);
			cout<<rho_optimum[class_tmp]<<" ";
		}
		cout<<endl;

		// release memory
				
		//if (einfo!=0) {
		//	cout<<"Error occured when computing the eigenvalue."<<endl;
		//	break;
		//}
		
		num_ite++;
	}

}

void DFE::PerformDev() {
    // subtracted the mean
    MatrixXd dev_data_tmp = dev_data;
    if (centralize==1) {
        for (int i=0; i<no_dim; i++) {
            dev_data_tmp.col(i)= dev_data_tmp.col(i).array()-train_data_mean(i);
        }
    }
    // divided by the sd
    if (normalize==1) {
        for (int i=0; i<no_dim; i++) {
            if (train_data_sd(i)!=0) {
            dev_data_tmp.col(i)=dev_data_tmp.col(i)/train_data_sd(i);
            }
        }
    }
    dev_data_red = dev_data_tmp*dfe_tsf;    // compute the transformed dataset

}

void DFE::PerformTest() {

	// subtracted the mean
    if (centralize==1) {
	    for (int i=0; i<no_dim; i++) {
		    test_data.col(i)= test_data.col(i).array()-train_data_mean(i);
	    }
    }
	// divided by the sd
	if (normalize==1) {
		for (int i=0; i<no_dim; i++) {
            if (train_data_sd(i)!=0) {
			test_data.col(i)=test_data.col(i)/train_data_sd(i);
            }
		}
	}

    cout<<"Test data: "<<test_data.rows()<<" "<<test_data.cols()<<endl;
	test_data_red = test_data*dfe_tsf;
}

DFE::~DFE() {}
