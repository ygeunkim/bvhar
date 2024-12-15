#include <svforecaster.h>

//' Forecasting predictive density of VAR-SV
//' 
//' @param num_chains Number of chains
//' @param var_lag VAR order.
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param sv Use Innovation?
//' @param sparse Use restricted model?
//' @param level CI level to give sparsity. Valid when `prior_type` is 0.
//' @param fit_record MCMC records list
//' @param prior_type Prior type. If 0, use CI. Valid when sparse is true.
//' @param seed_chain Seed for each chain
//' @param stable Filter stable draws
//' @param include_mean Include constant term?
//' @param nthreads OpenMP number of threads
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvarsv(int num_chains, int var_lag, int step, Eigen::MatrixXd response_mat,
													 bool sv, bool sparse, double level, Rcpp::List fit_record, int prior_type,
													 Eigen::VectorXi seed_chain, bool include_mean, bool stable, int nthreads) {
	auto forecaster = std::make_unique<bvhar::McmcForecastRun<bvhar::SvForecaster>>(
		num_chains, var_lag, step, response_mat,
		sparse, level, fit_record,
		seed_chain, include_mean, stable, nthreads,
		sv
	);
	return Rcpp::wrap(forecaster->returnForecast());
}

//' Forecasting Predictive Density of VHAR-SV
//' 
//' @param num_chains Number of MCMC chains
//' @param month VHAR month order.
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param HARtrans VHAR linear transformation matrix
//' @param sv Use Innovation?
//' @param sparse Use restricted model?
//' @param level CI level to give sparsity. Valid when `prior_type` is 0.
//' @param fit_record MCMC records list
//' @param prior_type Prior type. If 0, use CI. Valid when sparse is true.
//' @param seed_chain Seed for each chain
//' @param include_mean Include constant term?
//' @param stable Filter stable draws
//' @param nthreads OpenMP number of threads 
//'
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvharsv(int num_chains, int month, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd HARtrans,
														bool sv, bool sparse, double level, Rcpp::List fit_record, int prior_type,
														Eigen::VectorXi seed_chain, bool include_mean, bool stable, int nthreads) {
	auto forecaster = std::make_unique<bvhar::McmcForecastRun<bvhar::SvForecaster>>(
		num_chains, month, step, response_mat, HARtrans,
		sparse, level, fit_record,
		seed_chain, include_mean, stable, nthreads,
		sv
	);
	return Rcpp::wrap(forecaster->returnForecast());
}

//' Out-of-Sample Forecasting of VAR-SV based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVAR-SV.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag VAR order
//' @param num_chains Number of MCMC chains
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param thinning Thinning
//' @param param_sv SV specification list
//' @param param_prior Prior specification list
//' @param param_intercept Intercept specification list
//' @param param_init Initialization specification list
//' @param get_lpl Compute LPL
//' @param seed_chain Seed for each window and chain in the form of matrix
//' @param seed_forecast Seed for each window forecast
//' @param nthreads Number of threads for openmp
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param stable Filter stable draws
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads Number of threads
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bvarsv(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning,
											 bool sv, bool sparse, double level, Rcpp::List fit_record,
											 Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
											 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<bvhar::McmcOutforecastRun<bvhar::SvForecaster>> {
		if (ggl) {
			return std::make_unique<bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster, true>>(
				y, lag, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record,
				param_sv, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
			);
		}
		return std::make_unique<bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster, false>>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_sv, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
		);
	}();
	return forecaster->returnForecast();
}

//' Out-of-Sample Forecasting of VAR-SV based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVAR-SV.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag VAR order
//' @param num_chains Number of MCMC chains
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param thinning Thinning
//' @param param_sv SV specification list
//' @param param_prior Prior specification list
//' @param param_intercept Intercept specification list
//' @param param_init Initialization specification list
//' @param get_lpl Compute LPL
//' @param seed_chain Seed for each window and chain in the form of matrix
//' @param seed_forecast Seed for each window forecast
//' @param nthreads Number of threads for openmp
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param stable Filter stable draws
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads Number of threads
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bvharsv(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning,
												bool sv, bool sparse, double level, Rcpp::List fit_record,
											  Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											  Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<bvhar::McmcOutforecastRun<bvhar::SvForecaster>> {
		if (ggl) {
			return std::make_unique<bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster, true>>(
				y, week, month, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record,
				param_sv, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
			);
		}
		return std::make_unique<bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster, false>>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_sv, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
		);
	}();
	return forecaster->returnForecast();
}

//' Out-of-Sample Forecasting of VAR-SV based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVAR-SV.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag VAR order
//' @param num_chains Number of MCMC chains
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param thinning Thinning
//' @param param_sv SV specification list
//' @param param_prior Prior specification list
//' @param param_intercept Intercept specification list
//' @param param_init Initialization specification list
//' @param get_lpl Compute LPL
//' @param seed_chain Seed for each window and chain in the form of matrix
//' @param seed_forecast Seed for each window forecast
//' @param nthreads Number of threads for openmp
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param stable Filter stable draws
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads Number of threads
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_bvarsv(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning,
												 bool sv, bool sparse, double level, Rcpp::List fit_record,
											 	 Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											 	 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	 bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<bvhar::McmcOutforecastRun<bvhar::SvForecaster>> {
		if (ggl) {
			return std::make_unique<bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster, true>>(
				y, lag, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record,
				param_sv, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
			);
		}
		return std::make_unique<bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster, false>>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_sv, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
		);
	}();
	return forecaster->returnForecast();
}

//' Out-of-Sample Forecasting of VAR-SV based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVAR-SV.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag VAR order
//' @param num_chains Number of MCMC chains
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param thinning Thinning
//' @param param_sv SV specification list
//' @param param_prior Prior specification list
//' @param param_intercept Intercept specification list
//' @param param_init Initialization specification list
//' @param get_lpl Compute LPL
//' @param seed_chain Seed for each window and chain in the form of matrix
//' @param seed_forecast Seed for each window forecast
//' @param nthreads Number of threads for openmp
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param stable Filter stable draws
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads Number of threads
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_bvharsv(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning,
													bool sv, bool sparse, double level, Rcpp::List fit_record,
											  	Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											  	Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
													bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  	bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<bvhar::McmcOutforecastRun<bvhar::SvForecaster>> {
		if (ggl) {
			return std::make_unique<bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster, true>>(
				y, week, month, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record,
				param_sv, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
			);
		}
		return std::make_unique<bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster, false>>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_sv, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
		);
	}();
	return forecaster->returnForecast();
}
