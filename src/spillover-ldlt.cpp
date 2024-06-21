#include "bvharomp.h"
#include "regspillover.h"

// [[Rcpp::export]]
Rcpp::List compute_varldlt_spillover(int lag, int step,
																		 Eigen::MatrixXd alpha_record, Eigen::MatrixXd d_record, Eigen::MatrixXd a_record) {
	int dim = d_record.cols();
	bvhar::LdltRecords reg_record(alpha_record, a_record, d_record);
	std::unique_ptr<bvhar::RegSpillover> spillover;
	spillover.reset(new bvhar::RegSpillover(reg_record, step, lag));
	spillover->computeSpillover();
	Eigen::VectorXd to_sp = spillover->returnTo();
	Eigen::VectorXd from_sp = spillover->returnFrom();
	return Rcpp::List::create(
		Rcpp::Named("connect") = spillover->returnSpillover(),
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = spillover->returnTot(),
		Rcpp::Named("net") = to_sp - from_sp,
		Rcpp::Named("net_pairwise") = spillover->returnNet()
	);
}

// [[Rcpp::export]]
Rcpp::List compute_vharldlt_spillover(int week, int month, int step,
																			Eigen::MatrixXd phi_record, Eigen::MatrixXd d_record, Eigen::MatrixXd a_record) {
	int dim = d_record.cols();
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, false);
	bvhar::LdltRecords reg_record(phi_record, a_record, d_record);
	std::unique_ptr<bvhar::RegSpillover> spillover;
	spillover.reset(new bvhar::RegVharSpillover(reg_record, step, month, har_trans));
	spillover->computeSpillover();
	Eigen::VectorXd to_sp = spillover->returnTo();
	Eigen::VectorXd from_sp = spillover->returnFrom();
	return Rcpp::List::create(
		Rcpp::Named("connect") = spillover->returnSpillover(),
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = spillover->returnTot(),
		Rcpp::Named("net") = to_sp - from_sp,
		Rcpp::Named("net_pairwise") = spillover->returnNet()
	);
}
