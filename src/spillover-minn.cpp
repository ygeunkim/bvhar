#include "bvharomp.h"
#include "minnspillover.h"

//' Rolling-sample Total Spillover Index of BVAR
//' 
//' @param y Time series data of which columns indicate the variables
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param lag BVAR order
//' @param bayes_spec BVAR specification
//' @param include_mean Add constant term
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd dynamic_bvar_tot_spillover(Eigen::MatrixXd y, int window, int step, int num_iter, int num_burn,
																 					 int lag, Rcpp::List bayes_spec, bool include_mean, Eigen::VectorXi seed_chain, int nthreads) {
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	std::vector<std::unique_ptr<bvhar::MinnBvar>> mn_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		bvhar::BvarSpec mn_spec(bayes_spec);
		mn_objs[i] = std::unique_ptr<bvhar::MinnBvar>(new bvhar::MinnBvar(roll_mat, lag, mn_spec, include_mean));
	}
	std::vector<std::unique_ptr<bvhar::MinnSpillover>> spillover(num_horizon);
	Eigen::VectorXd res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
		bvhar::MinnFit mn_fit = mn_objs[i]->returnMinnFit();
		spillover[i].reset(new bvhar::MinnSpillover(mn_fit, step, num_iter, num_burn, lag, static_cast<unsigned int>(seed_chain[i])));
		spillover[i]->updateMniw();
		spillover[i]->computeSpillover();
		res[i] = spillover[i]->returnTot();
		mn_objs[i].reset(); // free the memory by making nullptr
		spillover[i].reset(); // free the memory by making nullptr
	}
	return res;
}

//' Rolling-sample Total Spillover Index of BVHAR
//' 
//' @param y Time series data of which columns indicate the variables
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param har BVHAR order
//' @param bayes_spec BVHAR specification
//' @param include_mean Add constant term
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd dynamic_bvhar_tot_spillover(Eigen::MatrixXd y, int window, int step, int num_iter, int num_burn,
																			      int week, int month, Rcpp::List bayes_spec, bool include_mean, Eigen::VectorXi seed_chain, int nthreads) {
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	int dim = y.cols();
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, include_mean);
	std::vector<std::unique_ptr<bvhar::MinnBvhar>> mn_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		if (bayes_spec.containsElementNamed("delta")) {
			bvhar::BvarSpec bvhar_spec(bayes_spec);
			mn_objs[i] = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharS(roll_mat, week, month, bvhar_spec, include_mean));
		} else {
			bvhar::BvharSpec bvhar_spec(bayes_spec);
			mn_objs[i] = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharL(roll_mat, week, month, bvhar_spec, include_mean));
		}
	}
	std::vector<std::unique_ptr<bvhar::BvharSpillover>> spillover(num_horizon);
  Eigen::VectorXd res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
		bvhar::MinnFit mn_fit = mn_objs[i]->returnMinnFit();
		spillover[i].reset(new bvhar::BvharSpillover(mn_fit, step, num_iter, num_burn, month, har_trans, static_cast<unsigned int>(seed_chain[i])));
		spillover[i]->updateMniw();
		spillover[i]->computeSpillover();
		res[i] = spillover[i]->returnTot();
		mn_objs[i].reset(); // free the memory by making nullptr
		spillover[i].reset(); // free the memory by making nullptr
	}
	return res;
}
