#include "bvharomp.h"
#include "olsspillover.h"

//' Rolling-sample Total Spillover Index of VAR
//' 
//' @param y Time series data of which columns indicate the variables
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param lag VAR order
//' @param include_mean Add constant term
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd dynamic_var_tot_spillover(Eigen::MatrixXd y, int window, int step, int lag, bool include_mean, int method, int nthreads) {
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	std::vector<std::unique_ptr<bvhar::OlsVar>> ols_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		ols_objs[i] = std::unique_ptr<bvhar::OlsVar>(new bvhar::OlsVar(roll_mat, lag, include_mean, method));
	}
	std::vector<std::unique_ptr<bvhar::OlsSpillover>> spillover(num_horizon);
	Eigen::VectorXd res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
		bvhar::StructuralFit ols_fit = ols_objs[i]->returnStructuralFit(step - 1);
		spillover[i].reset(new bvhar::OlsSpillover(ols_fit));
		spillover[i]->computeSpillover();
		res[i] = spillover[i]->returnTot();
		ols_objs[i].reset(); // free the memory by making nullptr
		spillover[i].reset(); // free the memory by making nullptr
	}
	return res;
}

//' Rolling-sample Total Spillover Index of VHAR
//' 
//' @param y Time series data of which columns indicate the variables
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param har VHAR order
//' @param include_mean Add constant term
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd dynamic_vhar_tot_spillover(Eigen::MatrixXd y, int window, int step, int week, int month, bool include_mean, int method, int nthreads) {
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	std::vector<std::unique_ptr<bvhar::OlsVhar>> ols_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		ols_objs[i] = std::unique_ptr<bvhar::OlsVhar>(new bvhar::OlsVhar(roll_mat, week, month, include_mean, method));
	}
	std::vector<std::unique_ptr<bvhar::OlsSpillover>> spillover(num_horizon);
	Eigen::VectorXd res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
		bvhar::StructuralFit ols_fit = ols_objs[i]->returnStructuralFit(step - 1);
		spillover[i].reset(new bvhar::OlsSpillover(ols_fit));
		spillover[i]->computeSpillover();
		res[i] = spillover[i]->returnTot();
		ols_objs[i].reset(); // free the memory by making nullptr
		spillover[i].reset(); // free the memory by making nullptr
	}
	return res;
}
