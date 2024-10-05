#include <bayesforecast.h>

//' Forecasting predictive density of BVAR and BVHAR
//' 
//' @param num_chains Number of chains
//' @param ord VAR order of length 1 or VHAR order of length 2 (week, month).
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param sv Use Innovation?
//' @param sparse Use restricted model?
//' @param level CI level to give sparsity. Valid when `prior_type` is 0.
//' @param fit_record MCMC records list
//' @param prior_type Prior type. If 0, use CI. Valid when sparse is true.
//' @param seed_chain Seed for each chain
//' @param include_mean Include constant term?
//' @param nthreads OpenMP number of threads
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_mcmc(int num_chains, Eigen::VectorXi& ord, int step, Eigen::MatrixXd& response_mat,
												 bool sparse, double level, Rcpp::List& fit_record, int prior_type,
												 Eigen::VectorXi& seed_chain, bool include_mean, int nthreads) {
	std::unique_ptr<bvhar::McmcForecastInterface> forecaster;
	bvhar::init_mcmcforecaster(
		forecaster, num_chains, ord, step, response_mat, sparse, level,
		fit_record, prior_type, seed_chain, include_mean, nthreads
	);
	return Rcpp::wrap(forecaster->returnForecast());
}
