#include "bvharomp.h"
#include "svspillover.h"

//' Dynamic Total Spillover Index of BVAR-SV
//' 
//' @param lag VAR lag.
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param response_mat Response matrix.
//' @param phi_record Coefficients MCMC record
//' @param h_record log volatility MCMC record
//' @param a_record Contemporaneous coefficients MCMC record
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List dynamic_bvarsv_spillover(int lag, int step, int num_design,
																	 	Eigen::MatrixXd alpha_record, Eigen::MatrixXd h_record, Eigen::MatrixXd a_record, int nthreads) {
	int dim = h_record.cols() / num_design;
	Eigen::VectorXd tot(num_design); // length = T - p
	Eigen::MatrixXd to_sp(num_design, dim);
	Eigen::MatrixXd from_sp(num_design, dim);
	std::vector<std::unique_ptr<bvhar::SvSpillover>> spillover(num_design);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_design; i++) {
		bvhar::SvRecords sv_record(alpha_record, h_record, a_record, Eigen::MatrixXd::Zero(h_record.rows(), dim));
		spillover[i].reset(new bvhar::SvSpillover(sv_record, step, lag, i));
		spillover[i]->computeSpillover();
		to_sp.row(i) = spillover[i]->returnTo();
		from_sp.row(i) = spillover[i]->returnFrom();
		tot[i] = spillover[i]->returnTot();
		spillover[i].reset(); // free the memory by making nullptr
	}
	return Rcpp::List::create(
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = tot,
		Rcpp::Named("net") = to_sp - from_sp
	);
}

//' Dynamic Total Spillover Index of BVHAR-SV
//' 
//' @param month VHAR month order.
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param response_mat Response matrix.
//' @param HARtrans VHAR linear transformation matrix
//' @param phi_record Coefficients MCMC record
//' @param h_record log volatility MCMC record
//' @param a_record Contemporaneous coefficients MCMC record
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List dynamic_bvharsv_spillover(int week, int month, int step, int num_design,
																	 	 Eigen::MatrixXd phi_record, Eigen::MatrixXd h_record, Eigen::MatrixXd a_record, int nthreads) {
	int dim = h_record.cols() / num_design;
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, false);
	Eigen::VectorXd tot(num_design); // length = T - p
	Eigen::MatrixXd to_sp(num_design, dim);
	Eigen::MatrixXd from_sp(num_design, dim);
	std::vector<std::unique_ptr<bvhar::SvVharSpillover>> spillover(num_design);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_design; i++) {
		bvhar::SvRecords sv_record(phi_record, h_record, a_record, Eigen::MatrixXd::Zero(h_record.rows(), dim));
		spillover[i].reset(new bvhar::SvVharSpillover(sv_record, step, month, i, har_trans));
		spillover[i]->computeSpillover();
		to_sp.row(i) = spillover[i]->returnTo();
		from_sp.row(i) = spillover[i]->returnFrom();
		tot[i] = spillover[i]->returnTot();
		spillover[i].reset(); // free the memory by making nullptr
	}
	return Rcpp::List::create(
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = tot,
		Rcpp::Named("net") = to_sp - from_sp
	);
}
