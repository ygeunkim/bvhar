#include "mcmcsv.h"
#include "bvharinterrupt.h"

//' VAR-SV by Gibbs Sampler
//' 
//' This function generates parameters \eqn{\beta, a, \sigma_{h,i}^2, h_{0,i}} and log-volatilities \eqn{h_{i,1}, \ldots, h_{i, n}}.
//' 
//' @param num_chain Number of MCMC chains
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param thin Thinning
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param param_sv SV specification list
//' @param param_prior Prior specification list
//' @param param_intercept Intercept specification list
//' @param param_init Initialization specification list
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param seed_chain Seed for each chain
//' @param display_progress Progress bar
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_var_sv(int num_chains, int num_iter, int num_burn, int thin,
                           Eigen::MatrixXd x, Eigen::MatrixXd y,
													 Rcpp::List param_sv,
													 Rcpp::List param_prior,
													 Rcpp::List param_intercept,
													 Rcpp::List param_init,
                           int prior_type,
                           Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id,
                           Eigen::MatrixXi grp_mat,
                           bool include_mean,
													 Eigen::VectorXi seed_chain,
                           bool display_progress, int nthreads) {
	auto mcmc_run = std::make_unique<bvhar::McmcRun<bvhar::McmcSv>>(
		num_chains, num_iter, num_burn, thin, x, y,
		param_sv, param_prior, param_intercept, param_init, prior_type,
		grp_id, own_id, cross_id, grp_mat,
		include_mean, seed_chain,
		display_progress, nthreads
	);
  // Start Gibbs sampling-----------------------------------
	return mcmc_run->returnRecords();
}
