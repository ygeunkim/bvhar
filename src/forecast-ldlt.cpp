#include <regforecaster.h>

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
											 	 bool sparse, double level, Rcpp::List fit_record,
											 	 Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											 	 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
											 	 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	 bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads) {
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> roll_mat(num_horizon);
	std::vector<Eigen::MatrixXd> roll_y0(num_horizon);
	for (int i = 0; i < num_horizon; i++) {
		roll_mat[i] = tot_mat.middleRows(i, num_window);
		roll_y0[i] = bvhar::build_y0(roll_mat[i], lag, lag + 1);
	}
	tot_mat.resize(0, 0); // free the memory
	std::vector<std::vector<std::unique_ptr<bvhar::McmcReg>>> reg_objs(num_horizon);
	for (auto &reg_chain : reg_objs) {
		reg_chain.resize(num_chains);
		for (auto &ptr : reg_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::RegForecaster>>> forecaster(num_horizon);
	for (auto &reg_forecast : forecaster) {
		reg_forecast.resize(num_chains);
		for (auto &ptr : reg_forecast) {
			ptr = nullptr;
		}
	}
	int sparse_type = (level > 0) ? 0 : prior_type; // use CI if level > 0
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		forecaster[0] = bvhar::initialize_forecaster<bvhar::RegForecaster>(
			num_chains, lag, step, roll_y0[0], sparse, level,
			fit_record, seed_forecast, include_mean,
			stable, nthreads, true
		);
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	for (int window = 0; window < num_horizon; ++window) {
		Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
		reg_objs[window] = bvhar::initialize_mcmc<bvhar::McmcReg>(
			num_chains, num_iter, design, roll_y0[window],
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat,
			include_mean, seed_chain
		);
		roll_mat[window].resize(0, 0); // free the memory
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning);
				break;
			}
			reg_objs[window][chain]->doPosteriorDraws();
		}
		// bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning, sparse);
		if (sparse && sparse_type == 0) {
			bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning, false);
			forecaster[window][chain].reset(new bvhar::RegVarSelectForecaster(
				reg_record, bvhar::unvectorize(reg_record.computeActivity(level), dim),
				step, roll_y0[window], lag, include_mean, stable, static_cast<unsigned int>(seed_forecast[chain])
			));
		} else {
			bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning, sparse);
			forecaster[window][chain].reset(new bvhar::RegVarForecaster(
				reg_record, step, roll_y0[window], lag, include_mean, stable, static_cast<unsigned int>(seed_forecast[chain])
			));
		}
		reg_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			Eigen::VectorXd valid_vec = y_test.row(step);
			res[window][0] = forecaster[window][0]->forecastDensity(valid_vec).bottomRows(1);
			lpl_record(window, 0) = forecaster[window][0]->returnLpl();
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, num_chains) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				Eigen::VectorXd valid_vec = y_test.row(step);
				res[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
				lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
  if (!get_lpl) {
		return Rcpp::wrap(res);
	}
	Rcpp::List res_list = Rcpp::wrap(res);
	res_list["lpl"] = lpl_record.mean();
	return res_list;
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
													bool sparse, double level, Rcpp::List fit_record,
											  	Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											  	Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
													bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  	bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads) {
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> roll_mat(num_horizon);
	std::vector<Eigen::MatrixXd> roll_y0(num_horizon);
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, include_mean);
	for (int i = 0; i < num_horizon; i++) {
		roll_mat[i] = tot_mat.middleRows(i, num_window);
		roll_y0[i] = bvhar::build_y0(roll_mat[i], month, month + 1);
	}
	tot_mat.resize(0, 0); // free the memory
	std::vector<std::vector<std::unique_ptr<bvhar::McmcReg>>> reg_objs(num_horizon);
	for (auto &reg_chain : reg_objs) {
		reg_chain.resize(num_chains);
		for (auto &ptr : reg_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::RegForecaster>>> forecaster(num_horizon);
	for (auto &reg_forecast : forecaster) {
		reg_forecast.resize(num_chains);
		for (auto &ptr : reg_forecast) {
			ptr = nullptr;
		}
	}
	int sparse_type = (level > 0) ? 0 : prior_type; // use CI if level > 0
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		forecaster[0] = bvhar::initialize_forecaster<bvhar::RegForecaster>(
			num_chains, month, step, roll_y0[0], sparse, level,
			fit_record, seed_forecast, include_mean,
			stable, nthreads, true, har_trans
		);
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	for (int window = 0; window < num_horizon; ++window) {
		Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
		reg_objs[window] = bvhar::initialize_mcmc<bvhar::McmcReg>(
			num_chains, num_iter, design, roll_y0[window],
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat,
			include_mean, seed_chain
		);
		roll_mat[window].resize(0, 0); // free the memory
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning);
				break;
			}
			reg_objs[window][chain]->doPosteriorDraws();
		}
		// bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning);
		if (sparse && sparse_type == 0) {
			bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning, false);
			forecaster[window][chain].reset(new bvhar::RegVharSelectForecaster(
				reg_record, bvhar::unvectorize(reg_record.computeActivity(level), dim),
				step, roll_y0[window], har_trans, month, include_mean, stable, static_cast<unsigned int>(seed_forecast[chain])
			));
		} else {
			bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning, sparse);
			forecaster[window][chain].reset(new bvhar::RegVharForecaster(
				reg_record, step, roll_y0[window], har_trans, month, include_mean, stable, static_cast<unsigned int>(seed_forecast[chain])
			));
		}
		reg_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			Eigen::VectorXd valid_vec = y_test.row(step);
			res[window][0] = forecaster[window][0]->forecastDensity(valid_vec).bottomRows(1);
			lpl_record(window, 0) = forecaster[window][0]->returnLpl();
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, num_chains) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				Eigen::VectorXd valid_vec = y_test.row(step);
				res[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
				lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
	if (!get_lpl) {
		return Rcpp::wrap(res);
	}
	Rcpp::List res_list = Rcpp::wrap(res);
	res_list["lpl"] = lpl_record.mean();
	return res_list;
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
												 	 bool sparse, double level, Rcpp::List fit_record,
											 	 	 Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											 	 	 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												 	 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	 	 bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads) {
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> expand_mat(num_horizon);
	std::vector<Eigen::MatrixXd> expand_y0(num_horizon);
	for (int i = 0; i < num_horizon; i++) {
		expand_mat[i] = tot_mat.topRows(num_window + i);
		expand_y0[i] = bvhar::build_y0(expand_mat[i], lag, lag + 1);
	}
	tot_mat.resize(0, 0); // free the memory
	std::vector<std::vector<std::unique_ptr<bvhar::McmcReg>>> reg_objs(num_horizon);
	for (auto &reg_chain : reg_objs) {
		reg_chain.resize(num_chains);
		for (auto &ptr : reg_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::RegForecaster>>> forecaster(num_horizon);
	for (auto &reg_forecast : forecaster) {
		reg_forecast.resize(num_chains);
		for (auto &ptr : reg_forecast) {
			ptr = nullptr;
		}
	}
	int sparse_type = (level > 0) ? 0 : prior_type; // use CI if level > 0
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		forecaster[0] = bvhar::initialize_forecaster<bvhar::RegForecaster>(
			num_chains, lag, step, expand_y0[0], sparse, level,
			fit_record, seed_forecast, include_mean,
			stable, nthreads, true
		);
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	for (int window = 0; window < num_horizon; ++window) {
		Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
		reg_objs[window] = bvhar::initialize_mcmc<bvhar::McmcReg>(
			num_chains, num_iter, design, expand_y0[window],
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat,
			include_mean, seed_chain
		);
		expand_mat[window].resize(0, 0); // free the memory
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning);
				break;
			}
			reg_objs[window][chain]->doPosteriorDraws();
		}
		// bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning);
		if (sparse && sparse_type == 0) {
			bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning, false);
			forecaster[window][chain].reset(new bvhar::RegVarSelectForecaster(
				reg_record, bvhar::unvectorize(reg_record.computeActivity(level), dim),
				step, expand_y0[window], lag, include_mean, stable, static_cast<unsigned int>(seed_forecast[chain])
			));
		} else {
			bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning, sparse);
			forecaster[window][chain].reset(new bvhar::RegVarForecaster(
				reg_record, step, expand_y0[window], lag, include_mean, stable, static_cast<unsigned int>(seed_forecast[chain])
			));
		}
		reg_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			Eigen::VectorXd valid_vec = y_test.row(step);
			res[window][0] = forecaster[window][0]->forecastDensity(valid_vec).bottomRows(1);
			lpl_record(window, 0) = forecaster[window][0]->returnLpl();
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(dynamic, num_chains) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				Eigen::VectorXd valid_vec = y_test.row(step);
				res[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
				lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
  if (!get_lpl) {
		return Rcpp::wrap(res);
	}
	Rcpp::List res_list = Rcpp::wrap(res);
	res_list["lpl"] = lpl_record.mean();
	return res_list;
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
														bool sparse, double level, Rcpp::List fit_record,
											  		Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											  		Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
														bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  		bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads) {
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> expand_mat(num_horizon);
	std::vector<Eigen::MatrixXd> expand_y0(num_horizon);
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, include_mean);
	for (int i = 0; i < num_horizon; i++) {
		expand_mat[i] = tot_mat.topRows(num_window + i);
		expand_y0[i] = bvhar::build_y0(expand_mat[i], month, month + 1);
	}
	tot_mat.resize(0, 0); // free the memory
	std::vector<std::vector<std::unique_ptr<bvhar::McmcReg>>> reg_objs(num_horizon);
	for (auto &reg_chain : reg_objs) {
		reg_chain.resize(num_chains);
		for (auto &ptr : reg_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::RegForecaster>>> forecaster(num_horizon);
	for (auto &reg_forecast : forecaster) {
		reg_forecast.resize(num_chains);
		for (auto &ptr : reg_forecast) {
			ptr = nullptr;
		}
	}
	int sparse_type = (level > 0) ? 0 : prior_type; // use CI if level > 0
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		forecaster[0] = bvhar::initialize_forecaster<bvhar::RegForecaster>(
			num_chains, month, step, expand_y0[0], sparse, level,
			fit_record, seed_forecast, include_mean,
			stable, nthreads, true, har_trans
		);
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	for (int window = 0; window < num_horizon; ++window) {
		Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
		reg_objs[window] = bvhar::initialize_mcmc<bvhar::McmcReg>(
			num_chains, num_iter, design, expand_y0[window],
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat,
			include_mean, seed_chain
		);
		expand_mat[window].resize(0, 0); // free the memory
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::LdltRecords sv_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning);
				break;
			}
			reg_objs[window][chain]->doPosteriorDraws();
		}
		// bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning);
		if (sparse && sparse_type == 0) {
			bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning, false);
			forecaster[window][chain].reset(new bvhar::RegVharSelectForecaster(
				reg_record, bvhar::unvectorize(reg_record.computeActivity(level), dim),
				step, expand_y0[window], har_trans, month, include_mean, stable, static_cast<unsigned int>(seed_forecast[chain])
			));
		} else {
			bvhar::LdltRecords reg_record = reg_objs[window][chain]->returnLdltRecords(num_burn, thinning, sparse);
			forecaster[window][chain].reset(new bvhar::RegVharForecaster(
				reg_record, step, expand_y0[window], har_trans, month, include_mean, stable, static_cast<unsigned int>(seed_forecast[chain])
			));
		}
		reg_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			Eigen::VectorXd valid_vec = y_test.row(step);
			res[window][0] = forecaster[window][0]->forecastDensity(valid_vec).bottomRows(1);
			lpl_record(window, 0) = forecaster[window][0]->returnLpl();
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(dynamic, num_chains) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				Eigen::VectorXd valid_vec = y_test.row(step);
				res[window][chain] = forecaster[window][chain]->forecastDensity(valid_vec).bottomRows(1);
				lpl_record(window, chain) = forecaster[window][chain]->returnLpl();
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
  if (!get_lpl) {
		return Rcpp::wrap(res);
	}
	Rcpp::List res_list = Rcpp::wrap(res);
	res_list["lpl"] = lpl_record.mean();
	return res_list;
}
