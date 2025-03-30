#include <bvhar/triangular>

//' VAR with Shrinkage Priors
//' 
//' This function generates parameters \eqn{\beta, a, \sigma_{h,i}^2, h_{0,i}} and log-volatilities \eqn{h_{i,1}, \ldots, h_{i, n}}.
//' 
//' @param num_chain Number of MCMC chains
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param thin Thinning
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param param_reg Regression specification list
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
Rcpp::List estimate_sur(int num_chains, int num_iter, int num_burn, int thin,
                        Eigen::MatrixXd x, Eigen::MatrixXd y,
												Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept,
												Rcpp::List param_init, int prior_type, bool ggl,
                        Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
                        bool include_mean, Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
	auto mcmc_run = [&]() -> std::unique_ptr<bvhar::McmcInterface> {
		if (param_reg.containsElementNamed("initial_mean")) {
			if (ggl) {
				return std::make_unique<bvhar::McmcRun<bvhar::McmcSv, true>>(
					num_chains, num_iter, num_burn, thin, x, y,
					param_reg, param_prior, param_intercept, param_init, prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain,
					display_progress, nthreads
				);
			}
			return std::make_unique<bvhar::McmcRun<bvhar::McmcSv, false>>(
				num_chains, num_iter, num_burn, thin, x, y,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain,
				display_progress, nthreads
			);
		}
		if (ggl) {
			return std::make_unique<bvhar::McmcRun<bvhar::McmcReg, true>>(
				num_chains, num_iter, num_burn, thin, x, y,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain,
				display_progress, nthreads
			);
		}
		return std::make_unique<bvhar::McmcRun<bvhar::McmcReg, false>>(
			num_chains, num_iter, num_burn, thin, x, y,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat,
			include_mean, seed_chain,
			display_progress, nthreads
		);
	}();
  // Start Gibbs sampling-----------------------------------
	return mcmc_run->returnRecords();
}

//' Forecasting predictive density of BVAR
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
Rcpp::List forecast_bvarldlt(int num_chains, int var_lag, int step, Eigen::MatrixXd response_mat,
													 	 bool sparse, double level, Rcpp::List fit_record, int prior_type,
													 	 Eigen::VectorXi seed_chain, bool include_mean, bool stable, int nthreads) {
	auto forecaster = std::make_unique<bvhar::McmcForecastRun<bvhar::RegForecaster>>(
		num_chains, var_lag, step, response_mat,
		sparse, level, fit_record,
		seed_chain, include_mean, stable, nthreads
	);
	return Rcpp::wrap(forecaster->returnForecast());
}

//' Forecasting Predictive Density of BVHAR
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
Rcpp::List forecast_bvharldlt(int num_chains, int month, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd HARtrans,
															bool sparse, double level, Rcpp::List fit_record, int prior_type,
															Eigen::VectorXi seed_chain, bool include_mean, bool stable, int nthreads) {
	auto forecaster = std::make_unique<bvhar::McmcForecastRun<bvhar::RegForecaster>>(
		num_chains, month, step, response_mat, HARtrans,
		sparse, level, fit_record,
		seed_chain, include_mean, stable, nthreads
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
//' @param param_reg SV specification list
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
Rcpp::List roll_bvarldlt(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning,
											 	 bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											 	 Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											 	 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
											 	 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	 bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = bvhar::initialize_outforecaster<bvhar::McmcRollforecastRun, bvhar::RegForecaster>(
		y, lag, num_chains, num_iter, num_burn, thinning,
		sparse, level, fit_record, run_mcmc,
		param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
		grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
		get_lpl, seed_chain, seed_forecast, display_progress, nthreads, true
	);
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
Rcpp::List roll_bvharldlt(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning,
													bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											  	Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											  	Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
													bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  	bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = bvhar::initialize_outforecaster<bvhar::McmcRollforecastRun, bvhar::RegForecaster>(
		y, week, month, num_chains, num_iter, num_burn, thinning,
		sparse, level, fit_record, run_mcmc,
		param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
		grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
		get_lpl, seed_chain, seed_forecast, display_progress, nthreads, true
	);
	return forecaster->returnForecast();
}

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
											 bool sv, bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											 Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
											 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = bvhar::initialize_outforecaster<bvhar::McmcRollforecastRun, bvhar::SvForecaster>(
		y, lag, num_chains, num_iter, num_burn, thinning,
		sparse, level, fit_record, run_mcmc,
		param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
		grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
		get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
	);
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
												bool sv, bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											  Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											  Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = bvhar::initialize_outforecaster<bvhar::McmcRollforecastRun, bvhar::SvForecaster>(
		y, week, month, num_chains, num_iter, num_burn, thinning,
		sparse, level, fit_record, run_mcmc,
		param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
		grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
		get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
	);
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
Rcpp::List expand_bvarldlt(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning,
												 	 bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											 	 	 Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											 	 	 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												 	 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	 	 bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = bvhar::initialize_outforecaster<bvhar::McmcExpandforecastRun, bvhar::RegForecaster>(
		y, lag, num_chains, num_iter, num_burn, thinning,
		sparse, level, fit_record, run_mcmc,
		param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
		grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
		get_lpl, seed_chain, seed_forecast, display_progress, nthreads, true
	);
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
Rcpp::List expand_bvharldlt(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning,
														bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											  		Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											  		Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
														bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  		bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = bvhar::initialize_outforecaster<bvhar::McmcExpandforecastRun, bvhar::RegForecaster>(
		y, week, month, num_chains, num_iter, num_burn, thinning,
		sparse, level, fit_record, run_mcmc,
		param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
		grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
		get_lpl, seed_chain, seed_forecast, display_progress, nthreads, true
	);
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
												 bool sv, bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											 	 Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											 	 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	 bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = bvhar::initialize_outforecaster<bvhar::McmcExpandforecastRun, bvhar::SvForecaster>(
		y, lag, num_chains, num_iter, num_burn, thinning,
		sparse, level, fit_record, run_mcmc,
		param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
		grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
		get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
	);
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
													bool sv, bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											  	Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											  	Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
													bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  	bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = bvhar::initialize_outforecaster<bvhar::McmcExpandforecastRun, bvhar::SvForecaster>(
		y, week, month, num_chains, num_iter, num_burn, thinning,
		sparse, level, fit_record, run_mcmc,
		param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
		grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
		get_lpl, seed_chain, seed_forecast, display_progress, nthreads, sv
	);
	return forecaster->returnForecast();
}

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
Rcpp::List dynamic_bvarsv_spillover(int lag, int step, int num_design, Rcpp::List fit_record, bool sparse, bool include_mean, int nthreads) {
	auto spillover = std::make_unique<bvhar::DynamicSvSpillover>(lag, step, num_design, fit_record, include_mean, sparse, nthreads);
	return spillover->returnSpillover();
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
Rcpp::List dynamic_bvharsv_spillover(int week, int month, int step, int num_design, Rcpp::List fit_record, bool sparse, bool include_mean, int nthreads) {
	auto spillover = std::make_unique<bvhar::DynamicSvSpillover>(week, month, step, num_design, fit_record, include_mean, sparse, nthreads);
	return spillover->returnSpillover();
}
