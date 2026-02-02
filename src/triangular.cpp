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
												Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
												Rcpp::List exogen_prior, Rcpp::List exogen_init, int exogen_prior_type, int exogen_cols,
												Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor,
                        Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
                        bool include_mean, Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
	auto mcmc_run = [&]() -> std::unique_ptr<baecon::bvhar::McmcRun> {
		if (factor_prior_type != 0) {
			if (exogen_prior_type != 0) {
				if (param_reg.containsElementNamed("initial_mean")) {
					if (ggl) {
						return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcSv, true>>(
							num_chains, num_iter, num_burn, thin, x, y,
							param_reg, param_prior, param_intercept, param_init, prior_type,
							contem_prior, contem_init, contem_prior_type,
							grp_id, own_id, cross_id, grp_mat,
							include_mean, seed_chain,
							display_progress, nthreads,
							exogen_prior, exogen_init, exogen_prior_type, exogen_cols,
							factor_prior, factor_init, factor_prior_type, size_factor
						);
					}
					return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcSv, false>>(
						num_chains, num_iter, num_burn, thin, x, y,
						param_reg, param_prior, param_intercept, param_init, prior_type,
						contem_prior, contem_init, contem_prior_type,
						grp_id, own_id, cross_id, grp_mat,
						include_mean, seed_chain,
						display_progress, nthreads,
						exogen_prior, exogen_init, exogen_prior_type, exogen_cols,
						factor_prior, factor_init, factor_prior_type, size_factor
					);
				}
				if (ggl) {
					return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcReg, true>>(
						num_chains, num_iter, num_burn, thin, x, y,
						param_reg, param_prior, param_intercept, param_init, prior_type,
						contem_prior, contem_init, contem_prior_type,
						grp_id, own_id, cross_id, grp_mat,
						include_mean, seed_chain,
						display_progress, nthreads,
						exogen_prior, exogen_init, exogen_prior_type, exogen_cols,
						factor_prior, factor_init, factor_prior_type, size_factor
					);
				}
				return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcReg, false>>(
					num_chains, num_iter, num_burn, thin, x, y,
					param_reg, param_prior, param_intercept, param_init, prior_type,
					contem_prior, contem_init, contem_prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain,
					display_progress, nthreads,
					exogen_prior, exogen_init, exogen_prior_type, exogen_cols,
					factor_prior, factor_init, factor_prior_type, size_factor
				);
			}
			if (param_reg.containsElementNamed("initial_mean")) {
				if (ggl) {
					return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcSv, true>>(
						num_chains, num_iter, num_burn, thin, x, y,
						param_reg, param_prior, param_intercept, param_init, prior_type,
						contem_prior, contem_init, contem_prior_type,
						grp_id, own_id, cross_id, grp_mat,
						include_mean, seed_chain,
						display_progress, nthreads,
						BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
						factor_prior, factor_init, factor_prior_type, size_factor
					);
				}
				return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcSv, false>>(
					num_chains, num_iter, num_burn, thin, x, y,
					param_reg, param_prior, param_intercept, param_init, prior_type,
					contem_prior, contem_init, contem_prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain,
					display_progress, nthreads,
					BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
					factor_prior, factor_init, factor_prior_type, size_factor
				);
			}
			if (ggl) {
				return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcReg, true>>(
					num_chains, num_iter, num_burn, thin, x, y,
					param_reg, param_prior, param_intercept, param_init, prior_type,
					contem_prior, contem_init, contem_prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain,
					display_progress, nthreads,
					BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
					factor_prior, factor_init, factor_prior_type, size_factor
				);
			}
			return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcReg, false>>(
				num_chains, num_iter, num_burn, thin, x, y,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain,
				display_progress, nthreads,
				BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
				factor_prior, factor_init, factor_prior_type, size_factor
			);
		}
		if (exogen_prior_type != 0) {
			if (param_reg.containsElementNamed("initial_mean")) {
				if (ggl) {
					return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcSv, true>>(
						num_chains, num_iter, num_burn, thin, x, y,
						param_reg, param_prior, param_intercept, param_init, prior_type,
						contem_prior, contem_init, contem_prior_type,
						grp_id, own_id, cross_id, grp_mat,
						include_mean, seed_chain,
						display_progress, nthreads,
						exogen_prior, exogen_init, exogen_prior_type, exogen_cols
					);
				}
				return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcSv, false>>(
					num_chains, num_iter, num_burn, thin, x, y,
					param_reg, param_prior, param_intercept, param_init, prior_type,
					contem_prior, contem_init, contem_prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain,
					display_progress, nthreads,
					exogen_prior, exogen_init, exogen_prior_type, exogen_cols
				);
			}
			if (ggl) {
				return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcReg, true>>(
					num_chains, num_iter, num_burn, thin, x, y,
					param_reg, param_prior, param_intercept, param_init, prior_type,
					contem_prior, contem_init, contem_prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain,
					display_progress, nthreads,
					exogen_prior, exogen_init, exogen_prior_type, exogen_cols
				);
			}
			return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcReg, false>>(
				num_chains, num_iter, num_burn, thin, x, y,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain,
				display_progress, nthreads,
				exogen_prior, exogen_init, exogen_prior_type, exogen_cols
			);
		}
		if (param_reg.containsElementNamed("initial_mean")) {
			if (ggl) {
				return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcSv, true>>(
					num_chains, num_iter, num_burn, thin, x, y,
					param_reg, param_prior, param_intercept, param_init, prior_type,
					contem_prior, contem_init, contem_prior_type,
					grp_id, own_id, cross_id, grp_mat,
					include_mean, seed_chain,
					display_progress, nthreads
				);
			}
			return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcSv, false>>(
				num_chains, num_iter, num_burn, thin, x, y,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain,
				display_progress, nthreads
			);
		}
		if (ggl) {
			return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcReg, true>>(
				num_chains, num_iter, num_burn, thin, x, y,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat,
				include_mean, seed_chain,
				display_progress, nthreads
			);
		}
		return std::make_unique<baecon::bvhar::CtaRun<baecon::bvhar::McmcReg, false>>(
			num_chains, num_iter, num_burn, thin, x, y,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			contem_prior, contem_init, contem_prior_type,
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
														 int size_factor, int factor_lag,
													 	 bool sparse, double level, Rcpp::List fit_record,
													 	 Eigen::VectorXi seed_chain, bool include_mean, bool stable, int nthreads,
														 bool insample) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>> {
		if (size_factor == 0) {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>>(
				num_chains, var_lag, step, response_mat,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads
			);
		} else {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>>(
				num_chains, var_lag, step, response_mat,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads,
				true, BVHAR_NULLOPT, BVHAR_NULLOPT,
				size_factor, factor_lag, insample
			);
		}
	}();
	// auto forecaster = std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>>(
	// 	num_chains, var_lag, step, response_mat,
	// 	sparse, level, fit_record,
	// 	seed_chain, include_mean, stable, nthreads
	// );
	if (insample) {
		return Rcpp::wrap(forecaster->returnPredict());
	}
	return Rcpp::wrap(forecaster->returnForecast());
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvarxldlt(int num_chains, int var_lag, int step, Eigen::MatrixXd response_mat,
															int size_factor, int factor_lag,
													 	  bool sparse, double level, Rcpp::List fit_record,
													 	  Eigen::VectorXi seed_chain, bool include_mean,
															Eigen::MatrixXd exogen, int exogen_lag,
															bool stable, int nthreads,
															bool insample) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>> {
		if (size_factor == 0) {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>>(
				num_chains, var_lag, step, response_mat,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads,
				true, exogen, exogen_lag
			);
		} else {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>>(
				num_chains, var_lag, step, response_mat,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads,
				true, exogen, exogen_lag,
				size_factor, factor_lag, insample
			);
		}
	}();
	// auto forecaster = std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>>(
	// 	num_chains, var_lag, step, response_mat,
	// 	sparse, level, fit_record,
	// 	seed_chain, include_mean, stable, nthreads,
	// 	true, exogen, exogen_lag
	// );
	if (insample) {
		return Rcpp::wrap(forecaster->returnPredict());
	}
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
															int size_factor, int factor_lag,
															bool sparse, double level, Rcpp::List fit_record,
															Eigen::VectorXi seed_chain, bool include_mean, bool stable, int nthreads,
															bool insample) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>> {
		if (size_factor == 0) {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>>(
				num_chains, month, step, response_mat, HARtrans,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads
			);
		} else {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>>(
				num_chains, month, step, response_mat, HARtrans,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads,
				true, BVHAR_NULLOPT, BVHAR_NULLOPT,
				size_factor, factor_lag, insample
			);
		}
	}();
	// auto forecaster = std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>>(
	// 	num_chains, month, step, response_mat, HARtrans,
	// 	sparse, level, fit_record,
	// 	seed_chain, include_mean, stable, nthreads
	// );
	if (insample) {
		return Rcpp::wrap(forecaster->returnPredict());
	}
	return Rcpp::wrap(forecaster->returnForecast());
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvharxldlt(int num_chains, int month, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd HARtrans,
															 int size_factor, int factor_lag,
													 	   bool sparse, double level, Rcpp::List fit_record,
													 	   Eigen::VectorXi seed_chain, bool include_mean,
															 Eigen::MatrixXd exogen, int exogen_lag,
															 bool stable, int nthreads,
															 bool insample) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>> {
		if (size_factor == 0) {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>>(
				num_chains, month, step, response_mat, HARtrans,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads,
				true, exogen, exogen_lag
			);
		} else {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>>(
				num_chains, month, step, response_mat, HARtrans,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads,
				true, exogen, exogen_lag,
				size_factor, factor_lag, insample
			);
		}
	}();
	// auto forecaster = std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>>(
	// 	num_chains, month, step, response_mat, HARtrans,
	// 	sparse, level, fit_record,
	// 	seed_chain, include_mean, stable, nthreads,
	// 	true, exogen, exogen_lag
	// );
	if (insample) {
		return Rcpp::wrap(forecaster->returnPredict());
	}
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
												 Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
												 Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											 	 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
											 	 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	 bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster>(
				y, lag, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true,
				BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true
		);
	}();
	// auto forecaster = baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster>(
	// 	y, lag, num_chains, num_iter, num_burn, thinning,
	// 	sparse, level, fit_record, run_mcmc,
	// 	param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
	// 	contem_prior, contem_init, contem_prior_type,
	// 	grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
	// 	get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true
	// );
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bvarxldlt(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning,
											 	  bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											 	  Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
												  Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
													Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											 	  Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
											 	  bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	  bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads,
													Eigen::MatrixXd exogen, int exogen_lag,
												  Rcpp::List exogen_prior, Rcpp::List exogen_init, int exogen_prior_type) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster>(
				y, lag, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true,
				exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
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
Rcpp::List roll_bvharldlt(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning,
													bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											  	Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
													Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
													Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											  	Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
													bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  	bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster>(
				y, week, month, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true,
				BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true
		);
	}();
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bvharxldlt(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning,
											 	   bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											 	   Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
												   Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
													 Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											 	   Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
											 	   bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	   bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads,
													 Eigen::MatrixXd exogen, int exogen_lag,
												   Rcpp::List exogen_prior, Rcpp::List exogen_init, int exogen_prior_type) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster>(
				y, week, month, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true,
				exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
		);
	}();
	// auto forecaster = baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster>(
	// 	y, week, month, num_chains, num_iter, num_burn, thinning,
	// 	sparse, level, fit_record, run_mcmc,
	// 	param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
	// 	contem_prior, contem_init, contem_prior_type,
	// 	grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
	// 	get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true,
	// 	exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
	// );
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
													 int size_factor, int factor_lag,
													 bool sv, bool sparse, double level, Rcpp::List fit_record,
													 Eigen::VectorXi seed_chain, bool include_mean, bool stable, int nthreads,
													 bool insample) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>> {
		if (size_factor == 0) {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>>(
				num_chains, var_lag, step, response_mat,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads,
				sv
			);
		} else {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>>(
				num_chains, var_lag, step, response_mat,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads,
				sv, BVHAR_NULLOPT, BVHAR_NULLOPT,
				size_factor, factor_lag, insample
			);
		}
	}();
	// auto forecaster = std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>>(
	// 	num_chains, var_lag, step, response_mat,
	// 	sparse, level, fit_record,
	// 	seed_chain, include_mean, stable, nthreads,
	// 	sv
	// );
	if (insample) {
		return Rcpp::wrap(forecaster->returnPredict());
	}
	return Rcpp::wrap(forecaster->returnForecast());
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvarxsv(int num_chains, int var_lag, int step, Eigen::MatrixXd response_mat,
														int size_factor, int factor_lag,
													 	bool sv, bool sparse, double level, Rcpp::List fit_record,
													 	Eigen::VectorXi seed_chain, bool include_mean,
														Eigen::MatrixXd exogen, int exogen_lag,
														bool stable, int nthreads,
														bool insample) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>> {
		if (size_factor == 0) {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>>(
				num_chains, var_lag, step, response_mat,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads,
				sv, exogen, exogen_lag
			);
		} else {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>>(
				num_chains, var_lag, step, response_mat,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads,
				sv, exogen, exogen_lag,
				size_factor, factor_lag, insample
			);
		}
	}();
	// auto forecaster = std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>>(
	// 	num_chains, var_lag, step, response_mat,
	// 	sparse, level, fit_record,
	// 	seed_chain, include_mean, stable, nthreads,
	// 	sv, exogen, exogen_lag
	// );
	if (insample) {
		return Rcpp::wrap(forecaster->returnPredict());
	}
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
														int size_factor, int factor_lag,
														bool sv, bool sparse, double level, Rcpp::List fit_record,
														Eigen::VectorXi seed_chain, bool include_mean, bool stable, int nthreads,
													  bool insample) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>> {
		if (size_factor == 0) {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>>(
				num_chains, month, step, response_mat, HARtrans,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads,
				sv
			);
		} else {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>>(
				num_chains, month, step, response_mat, HARtrans,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads,
				sv, BVHAR_NULLOPT, BVHAR_NULLOPT,
				size_factor, factor_lag, insample
			);
		}
	}();
	// auto forecaster = std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>>(
	// 	num_chains, month, step, response_mat, HARtrans,
	// 	sparse, level, fit_record,
	// 	seed_chain, include_mean, stable, nthreads,
	// 	sv
	// );
	if (insample) {
		return Rcpp::wrap(forecaster->returnPredict());
	}
	return Rcpp::wrap(forecaster->returnForecast());
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvharxsv(int num_chains, int month, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd HARtrans,
														 int size_factor, int factor_lag,
													 	 bool sv, bool sparse, double level, Rcpp::List fit_record,
													 	 Eigen::VectorXi seed_chain, bool include_mean,
														 Eigen::MatrixXd exogen, int exogen_lag,
														 bool stable, int nthreads,
														 bool insample) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>> {
		if (size_factor == 0) {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>>(
				num_chains, month, step, response_mat, HARtrans,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads,
				sv, exogen, exogen_lag
			);
		} else {
			return std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>>(
				num_chains, month, step, response_mat, HARtrans,
				sparse, level, fit_record,
				seed_chain, include_mean, stable, nthreads,
				sv, exogen, exogen_lag,
				size_factor, factor_lag, insample
			);
		}
	}();
	// auto forecaster = std::make_unique<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>>(
	// 	num_chains, month, step, response_mat, HARtrans,
	// 	sparse, level, fit_record,
	// 	seed_chain, include_mean, stable, nthreads,
	// 	sv, exogen, exogen_lag
	// );
	if (insample) {
		return Rcpp::wrap(forecaster->returnPredict());
	}
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
											 Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
											 Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
											 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster>(
				y, lag, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
				BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv
		);
	}();
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bvarxsv(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning,
											  bool sv, bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											  Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											  Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
												Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											  Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
											  bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads,
												Eigen::MatrixXd exogen, int exogen_lag,
												Rcpp::List exogen_prior, Rcpp::List exogen_init, int exogen_prior_type) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster>(
				y, lag, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
				exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
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
												bool sv, bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											  Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
												Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
												Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											  Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster>(
				y, week, month, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
				BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv
		);
	}();
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bvharxsv(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning,
												 bool sv, bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											   Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
												 Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
												 Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											   Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											   bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads,
												 Eigen::MatrixXd exogen, int exogen_lag,
												 Rcpp::List exogen_prior, Rcpp::List exogen_init, int exogen_prior_type) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster>(
				y, week, month, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
				exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
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
Rcpp::List expand_bvarldlt(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning,
												 	 bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											 	 	 Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
													 Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
													 Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											 	 	 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												 	 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	 	 bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster>(
				y, lag, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true,
				BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true
		);
	}();
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_bvarxldlt(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning,
												 	  bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											 	 	  Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
													  Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
														Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											 	 	  Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												 	  bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	 	  bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads,
													  Eigen::MatrixXd exogen, int exogen_lag,
												    Rcpp::List exogen_prior, Rcpp::List exogen_init, int exogen_prior_type) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster>(
				y, lag, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true,
				exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
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
Rcpp::List expand_bvharldlt(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning,
														bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											  		Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
														Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
														Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											  		Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
														bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  		bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster>(
				y, week, month, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true,
				BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true
		);
	}();
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_bvharxldlt(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning,
														 bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											  		 Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
														 Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
														 Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											  		 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
														 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  		 bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads,
														 Eigen::MatrixXd exogen, int exogen_lag,
												  	 Rcpp::List exogen_prior, Rcpp::List exogen_init, int exogen_prior_type) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster>(
				y, week, month, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true,
				exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, true,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
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
												 bool sv, bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											 	 Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
												 Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
												 Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											 	 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	 bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster>(
				y, lag, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
				BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv
		);
	}();
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_bvarxsv(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning,
												  bool sv, bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											 	  Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
												  Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
													Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											 	  Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												  bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	  bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads,
													Eigen::MatrixXd exogen, int exogen_lag,
												  Rcpp::List exogen_prior, Rcpp::List exogen_init, int exogen_prior_type) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster>(
				y, lag, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
				exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
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
													bool sv, bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											  	Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
													Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
													Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											  	Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
													bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  	bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster>(
				y, week, month, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
				BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT, BVHAR_NULLOPT,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv
		);
	}();
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_bvharxsv(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning,
													 bool sv, bool sparse, double level, Rcpp::List fit_record, bool run_mcmc,
											  	 Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
													 Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
													 Rcpp::List factor_prior, Rcpp::List factor_init, int factor_prior_type, int size_factor, int factor_lag,
											  	 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
													 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  	 bool get_lpl, bool use_fit, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads,
													 Eigen::MatrixXd exogen, int exogen_lag,
												   Rcpp::List exogen_prior, Rcpp::List exogen_init, int exogen_prior_type) {
	auto forecaster = [&]() -> std::unique_ptr<baecon::bvhar::McmcOutforecastInterface> {
		if (size_factor > 0) {
			return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster>(
				y, week, month, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record, run_mcmc,
				param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
				contem_prior, contem_init, contem_prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
				exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag,
				factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
			);
		}
		return baecon::bvhar::initialize_ctaoutforecaster<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record, run_mcmc,
			param_sv, param_prior, param_intercept, param_init, prior_type, ggl,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, use_fit, seed_chain, seed_forecast, display_progress, nthreads, sv,
			exogen_prior, exogen_init, exogen_prior_type, exogen, exogen_lag
		);
	}();
	return forecaster->returnForecast();
}

// [[Rcpp::export]]
Rcpp::List compute_varldlt_spillover(int lag, int step, Rcpp::List fit_record, bool sparse) {
	// auto spillover = baecon::bvhar::initialize_spillover<baecon::bvhar::LdltRecords>(0, lag, step, fit_record, sparse, 0);
	// return spillover->returnSpilloverDensity();
	auto spillover = std::make_unique<baecon::bvhar::McmcSpilloverRun<baecon::bvhar::LdltRecords>>(lag, step, fit_record, sparse);
	return spillover->returnSpillover();
}

// [[Rcpp::export]]
Rcpp::List compute_vharldlt_spillover(int week, int month, int step, Rcpp::List fit_record, bool sparse) {
	// auto spillover = baecon::bvhar::initialize_spillover<baecon::bvhar::LdltRecords>(0, month, step, fit_record, sparse, 0, BVHAR_NULLOPT, week);
	// return spillover->returnSpilloverDensity();
	auto spillover = std::make_unique<baecon::bvhar::McmcSpilloverRun<baecon::bvhar::LdltRecords>>(week, month, step, fit_record, sparse);
	return spillover->returnSpillover();
}

// [[Rcpp::export]]
Rcpp::List dynamic_bvarldlt_spillover(Eigen::MatrixXd y, int window, int step, int num_chains, int num_iter, int num_burn, int thin, bool sparse,
																			int lag, Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init,
																			int prior_type, bool ggl,
																			Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
																			Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
																			bool include_mean, Eigen::MatrixXi seed_chain, int nthreads) {
	auto spillover = std::make_unique<baecon::bvhar::DynamicLdltSpillover>(
		y, window, step, lag, num_chains, num_iter, num_burn, thin, sparse,
		param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
		contem_prior, contem_init, contem_prior_type,
		grp_id, own_id, cross_id, grp_mat,
		include_mean, seed_chain, nthreads
	);
	return spillover->returnSpillover();
}

// [[Rcpp::export]]
Rcpp::List dynamic_bvharldlt_spillover(Eigen::MatrixXd y, int window, int step, int num_chains, int num_iter, int num_burn, int thin, bool sparse,
																			 int week, int month, Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init,
																			 int prior_type, bool ggl,
																			 Rcpp::List contem_prior, Rcpp::List contem_init, int contem_prior_type,
																			 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
																			 bool include_mean, Eigen::MatrixXi seed_chain, int nthreads) {
	auto spillover = std::make_unique<baecon::bvhar::DynamicLdltSpillover>(
		y, window, step, week, month, num_chains, num_iter, num_burn, thin, sparse,
		param_reg, param_prior, param_intercept, param_init, prior_type, ggl,
		contem_prior, contem_init, contem_prior_type,
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
	auto spillover = std::make_unique<baecon::bvhar::DynamicSvSpillover>(lag, step, num_design, fit_record, include_mean, sparse, nthreads);
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
	auto spillover = std::make_unique<baecon::bvhar::DynamicSvSpillover>(week, month, step, num_design, fit_record, include_mean, sparse, nthreads);
	return spillover->returnSpillover();
}
