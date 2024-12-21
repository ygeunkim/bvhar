#include "bvharomp.h"
#include <regspillover.h>
// #include <bvharspillover.h>
#include <algorithm>

// [[Rcpp::export]]
Rcpp::List compute_varldlt_spillover(int lag, int step, Rcpp::List fit_record, bool sparse) {
	// auto spillover = bvhar::initialize_spillover<bvhar::LdltRecords>(0, lag, step, fit_record, sparse, 0);
	// return spillover->returnSpilloverDensity();
	auto spillover = std::make_unique<bvhar::McmcSpilloverRun<bvhar::LdltRecords>>(lag, step, fit_record, sparse);
	return spillover->returnSpillover();
}

// [[Rcpp::export]]
Rcpp::List compute_vharldlt_spillover(int week, int month, int step, Rcpp::List fit_record, bool sparse) {
	// auto spillover = bvhar::initialize_spillover<bvhar::LdltRecords>(0, month, step, fit_record, sparse, 0, NULLOPT, week);
	// return spillover->returnSpilloverDensity();
	auto spillover = std::make_unique<bvhar::McmcSpilloverRun<bvhar::LdltRecords>>(week, month, step, fit_record, sparse);
	return spillover->returnSpillover();
}

// [[Rcpp::export]]
Rcpp::List dynamic_bvarldlt_spillover(Eigen::MatrixXd y, int window, int step, int num_chains, int num_iter, int num_burn, int thin, bool sparse,
																			int lag, Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init,
																			int prior_type, bool ggl, Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
																			bool include_mean, Eigen::MatrixXi seed_chain, int nthreads) {
	auto spillover = std::make_unique<bvhar::DynamicLdltSpillover>(
		y, window, step, lag, num_chains, num_iter, num_burn, thin, sparse,
		param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
		grp_id, own_id, cross_id, grp_mat,
		include_mean, seed_chain, nthreads
	);
	return spillover->returnSpillover();
}

// [[Rcpp::export]]
Rcpp::List dynamic_bvharldlt_spillover(Eigen::MatrixXd y, int window, int step, int num_chains, int num_iter, int num_burn, int thin, bool sparse,
																			 int week, int month, Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init,
																			 int prior_type, bool ggl, Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
																			 bool include_mean, Eigen::MatrixXi seed_chain, int nthreads) {
	auto spillover = std::make_unique<bvhar::DynamicLdltSpillover>(
		y, window, step, week, month, num_chains, num_iter, num_burn, thin, sparse,
		param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
		grp_id, own_id, cross_id, grp_mat,
		include_mean, seed_chain, nthreads
	);
	return spillover->returnSpillover();
}
