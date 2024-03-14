#include "svforecaster.h"
#include "bvharinterrupt.h"

//' Forecasting predictive density of VAR-SV
//' 
//' @param var_lag VAR order.
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param coef_mat Posterior mean.
//' @param alpha_record MCMC record of coefficients
//' @param h_last_record MCMC record of log-volatilities in last time
//' @param a_record MCMC record of contemporaneous coefficients
//' @param sigh_record MCMC record of variance of log-volatilities
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvarsv(int num_chains, int var_lag, int step, Eigen::MatrixXd response_mat,
                           Eigen::MatrixXd alpha_record, Eigen::MatrixXd h_record, Eigen::MatrixXd a_record, Eigen::MatrixXd sigh_record,
													 bool use_sv, Eigen::VectorXi seed_chain, bool include_mean, int nthreads) {
	int num_sim = num_chains > 1 ? alpha_record.rows() / num_chains : alpha_record.rows();
	std::vector<std::unique_ptr<bvhar::SvVarForecaster>> forecaster(num_chains);
	for (int i = 0; i < num_chains; i++ ) {
		bvhar::SvRecords sv_record(
			alpha_record.middleRows(i * num_sim, num_sim),
			h_record.middleRows(i * num_sim, num_sim),
			a_record.middleRows(i * num_sim, num_sim),
			sigh_record.middleRows(i * num_sim, num_sim)
		);
		forecaster[i] = std::unique_ptr<bvhar::SvVarForecaster>(new bvhar::SvVarForecaster(
			sv_record, step, response_mat, var_lag, include_mean, static_cast<unsigned int>(seed_chain[i])
		));
	}
	std::vector<Eigen::MatrixXd> res(num_chains);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int chain = 0; chain < num_chains; chain++) {
		res[chain] = forecaster[chain]->forecastDensity(use_sv);
		forecaster[chain].reset(); // free the memory by making nullptr
	}
	return Rcpp::wrap(res);
}

//' Forecasting Predictive Density of VHAR-SV
//' 
//' @param month VHAR month order.
//' @param step Integer, Step to forecast.
//' @param response_mat Response matrix.
//' @param coef_mat Posterior mean.
//' @param HARtrans VHAR linear transformation matrix
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvharsv(int num_chains, int month, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd HARtrans,
														Eigen::MatrixXd phi_record, Eigen::MatrixXd h_record, Eigen::MatrixXd a_record, Eigen::MatrixXd sigh_record,
														bool use_sv, Eigen::VectorXi seed_chain, bool include_mean, int nthreads) {
	int num_sim = num_chains > 1 ? phi_record.rows() / num_chains : phi_record.rows();
	std::vector<std::unique_ptr<bvhar::SvVharForecaster>> forecaster(num_chains);
	for (int i = 0; i < num_chains; i++ ) {
		bvhar::SvRecords sv_record(
			phi_record.middleRows(i * num_sim, num_sim),
			h_record.middleRows(i * num_sim, num_sim),
			a_record.middleRows(i * num_sim, num_sim),
			sigh_record.middleRows(i * num_sim, num_sim)
		);
		forecaster[i] = std::unique_ptr<bvhar::SvVharForecaster>(new bvhar::SvVharForecaster(
			sv_record, step, response_mat, HARtrans, month, include_mean, static_cast<unsigned int>(seed_chain[i])
		));
	}
	std::vector<Eigen::MatrixXd> res(num_chains);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int chain = 0; chain < num_chains; chain++) {
		res[chain] = forecaster[chain]->forecastDensity(use_sv);
		forecaster[chain].reset(); // free the memory by making nullptr
	}
	return Rcpp::wrap(res);
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
//' @param seed_chain Seed for each window and chain in the form of matrix
//' @param seed_forecast Seed for each window forecast
//' @param nthreads Number of threads for openmp
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads Number of threads
//' @param chunk_size Chunk size for OpenMP static scheduling
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bvarsv(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
											 Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											 Eigen::VectorXi grp_id, Eigen::MatrixXi grp_mat, bool include_mean, int step, Eigen::MatrixXd y_test,
											 bool use_sv, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads, int chunk_size) {
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
	std::vector<std::vector<std::unique_ptr<bvhar::McmcSv>>> sv_objs(num_horizon);
	for (auto &sv_chain : sv_objs) {
		sv_chain.resize(num_chains);
		for (auto &ptr : sv_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::SvVarForecaster>>> forecaster(num_horizon);
	for (auto &sv_forecast : forecaster) {
		sv_forecast.resize(num_chains);
		for (auto &ptr : sv_forecast) {
			ptr = nullptr;
		}
	}
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		for (int i = 0; i < num_chains; i++) {
			std::unique_ptr<bvhar::SvRecords> sv_record;
			Rcpp::List alpha_list = fit_record["alpha_record"];
			Rcpp::List h_list = fit_record["h_record"];
			Rcpp::List a_list = fit_record["a_record"];
			Rcpp::List sigh_list = fit_record["sigh_record"];
			if (include_mean) {
				Rcpp::List c_list = fit_record["c_record"];
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(c_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			} else {
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			}
			forecaster[0][i].reset(new bvhar::SvVarForecaster(
				*sv_record, step, roll_y0[0], lag, include_mean, static_cast<unsigned int>(seed_forecast[i])
			));
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::MinnParams minn_params(
					num_iter, design, roll_y0[window],
					param_sv, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SvInits sv_inits(init_spec);
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::SsvsParams ssvs_params(
					num_iter, design, roll_y0[window],
					param_sv, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec);
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, design, roll_y0[window],
					param_sv, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HorseshoeInits hs_inits(init_spec);
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
				forecaster[window][chain].reset(new bvhar::SvVarForecaster(
					sv_record, step, roll_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
				));
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
		bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
		forecaster[window][chain].reset(new bvhar::SvVarForecaster(
			sv_record, step, roll_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		));
		sv_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains > 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			res[window][0] = forecaster[window][0]->forecastDensity(use_sv).bottomRows(1);
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				res[window][chain] = forecaster[window][chain]->forecastDensity(use_sv).bottomRows(1);
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
  return Rcpp::wrap(res);
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
//' @param seed_chain Seed for each window and chain in the form of matrix
//' @param seed_forecast Seed for each window forecast
//' @param nthreads Number of threads for openmp
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads Number of threads
//' @param chunk_size Chunk size for OpenMP static scheduling
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bvharsv(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
											  Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											  Eigen::VectorXi grp_id, Eigen::MatrixXi grp_mat, bool include_mean, int step, Eigen::MatrixXd y_test,
											  bool use_sv, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads, int chunk_size) {
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
	std::vector<std::vector<std::unique_ptr<bvhar::McmcSv>>> sv_objs(num_horizon);
	for (auto &sv_chain : sv_objs) {
		sv_chain.resize(num_chains);
		for (auto &ptr : sv_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::SvVharForecaster>>> forecaster(num_horizon);
	for (auto &sv_forecast : forecaster) {
		sv_forecast.resize(num_chains);
		for (auto &ptr : sv_forecast) {
			ptr = nullptr;
		}
	}
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		for (int i = 0; i < num_chains; i++) {
			std::unique_ptr<bvhar::SvRecords> sv_record;
			Rcpp::List phi_list = fit_record["phi_record"];
			Rcpp::List h_list = fit_record["h_record"];
			Rcpp::List a_list = fit_record["a_record"];
			Rcpp::List sigh_list = fit_record["sigh_record"];
			if (include_mean) {
				Rcpp::List c_list = fit_record["c_record"];
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
					Rcpp::as<Eigen::MatrixXd>(c_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			} else {
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			}
			forecaster[0][i].reset(new bvhar::SvVharForecaster(
				*sv_record, step, roll_y0[0], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[i])
			));
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::MinnParams minn_params(
					num_iter, design, roll_y0[window],
					param_sv, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SvInits sv_inits(init_spec);
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::SsvsParams ssvs_params(
					num_iter, design, roll_y0[window],
					param_sv, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec);
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, design, roll_y0[window],
					param_sv, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HorseshoeInits hs_inits(init_spec);
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
				forecaster[window][chain].reset(new bvhar::SvVharForecaster(
					sv_record, step, roll_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
				));
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
		bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
		forecaster[window][chain].reset(new bvhar::SvVharForecaster(
			sv_record, step, roll_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		));
		sv_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			res[window][0] = forecaster[window][0]->forecastDensity(use_sv).bottomRows(1);
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				res[window][chain] = forecaster[window][chain]->forecastDensity(use_sv).bottomRows(1);
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
  return Rcpp::wrap(res);
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
//' @param seed_chain Seed for each window and chain in the form of matrix
//' @param seed_forecast Seed for each window forecast
//' @param nthreads Number of threads for openmp
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads Number of threads
//' @param chunk_size Chunk size for OpenMP static scheduling
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_bvarsv(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
											 	 Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											 	 Eigen::VectorXi grp_id, Eigen::MatrixXi grp_mat, bool include_mean, int step, Eigen::MatrixXd y_test,
											 	 bool use_sv, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads, int chunk_size) {
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
	std::vector<std::vector<std::unique_ptr<bvhar::McmcSv>>> sv_objs(num_horizon);
	for (auto &sv_chain : sv_objs) {
		sv_chain.resize(num_chains);
		for (auto &ptr : sv_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::SvVarForecaster>>> forecaster(num_horizon);
	for (auto &sv_forecast : forecaster) {
		sv_forecast.resize(num_chains);
		for (auto &ptr : sv_forecast) {
			ptr = nullptr;
		}
	}
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		for (int i = 0; i < num_chains; i++) {
			std::unique_ptr<bvhar::SvRecords> sv_record;
			Rcpp::List alpha_list = fit_record["alpha_record"];
			Rcpp::List h_list = fit_record["h_record"];
			Rcpp::List a_list = fit_record["a_record"];
			Rcpp::List sigh_list = fit_record["sigh_record"];
			if (include_mean) {
				Rcpp::List c_list = fit_record["c_record"];
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(c_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			} else {
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			}
			forecaster[0][i].reset(new bvhar::SvVarForecaster(
				*sv_record, step, expand_y0[0], lag, include_mean, static_cast<unsigned int>(seed_forecast[i])
			));
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::MinnParams minn_params(
					num_iter, design, expand_y0[window],
					param_sv, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SvInits sv_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::SsvsParams ssvs_params(
					num_iter, design, expand_y0[window],
					param_sv, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, design, expand_y0[window],
					param_sv, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HorseshoeInits hs_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
				forecaster[window][chain].reset(new bvhar::SvVarForecaster(
					sv_record, step, expand_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
				));
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
		bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
		forecaster[window][chain].reset(new bvhar::SvVarForecaster(
			sv_record, step, expand_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		));
		sv_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains > 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			res[window][0] = forecaster[window][0]->forecastDensity(use_sv).bottomRows(1);
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				res[window][chain] = forecaster[window][chain]->forecastDensity(use_sv).bottomRows(1);
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
  return Rcpp::wrap(res);
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
//' @param seed_chain Seed for each window and chain in the form of matrix
//' @param seed_forecast Seed for each window forecast
//' @param nthreads Number of threads for openmp
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param include_mean Constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads Number of threads
//' @param chunk_size Chunk size for OpenMP static scheduling
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List expand_bvharsv(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
											  	Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											  	Eigen::VectorXi grp_id, Eigen::MatrixXi grp_mat, bool include_mean, int step, Eigen::MatrixXd y_test,
											  	bool use_sv, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads, int chunk_size) {
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
	std::vector<std::vector<std::unique_ptr<bvhar::McmcSv>>> sv_objs(num_horizon);
	for (auto &sv_chain : sv_objs) {
		sv_chain.resize(num_chains);
		for (auto &ptr : sv_chain) {
			ptr = nullptr;
		}
	}
	std::vector<std::vector<std::unique_ptr<bvhar::SvVharForecaster>>> forecaster(num_horizon);
	for (auto &sv_forecast : forecaster) {
		sv_forecast.resize(num_chains);
		for (auto &ptr : sv_forecast) {
			ptr = nullptr;
		}
	}
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		for (int i = 0; i < num_chains; i++) {
			std::unique_ptr<bvhar::SvRecords> sv_record;
			Rcpp::List phi_list = fit_record["phi_record"];
			Rcpp::List h_list = fit_record["h_record"];
			Rcpp::List a_list = fit_record["a_record"];
			Rcpp::List sigh_list = fit_record["sigh_record"];
			if (include_mean) {
				Rcpp::List c_list = fit_record["c_record"];
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
					Rcpp::as<Eigen::MatrixXd>(c_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			} else {
				sv_record.reset(new bvhar::SvRecords(
					Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
					Rcpp::as<Eigen::MatrixXd>(h_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(sigh_list[i])
				));
			}
			forecaster[0][i].reset(new bvhar::SvVharForecaster(
				*sv_record, step, expand_y0[0], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[i])
			));
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::MinnParams minn_params(
					num_iter, design, expand_y0[window],
					param_sv, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SvInits sv_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::SsvsParams ssvs_params(
					num_iter, design, expand_y0[window],
					param_sv, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::HorseshoeParams horseshoe_params(
					num_iter, design, expand_y0[window],
					param_sv, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HorseshoeInits hs_inits(init_spec, expand_y0[window].rows());
					sv_objs[window][chain] = std::unique_ptr<bvhar::McmcSv>(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
				bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
				forecaster[window][chain].reset(new bvhar::SvVharForecaster(
					sv_record, step, expand_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
				));
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
		bvhar::SvRecords sv_record = sv_objs[window][chain]->returnSvRecords(num_burn, thinning);
		forecaster[window][chain].reset(new bvhar::SvVharForecaster(
			sv_record, step, expand_y0[window], har_trans, month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		));
		sv_objs[window][chain].reset(); // free the memory by making nullptr
	};
	if (num_chains == 1) {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			if (!use_fit || window != 0) {
				run_gibbs(window, 0);
			}
			res[window][0] = forecaster[window][0]->forecastDensity(use_sv).bottomRows(1);
			forecaster[window][0].reset(); // free the memory by making nullptr
		}
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	#endif
		for (int window = 0; window < num_horizon; window++) {
			for (int chain = 0; chain < num_chains; chain++) {
				if (!use_fit || window != 0) {
					run_gibbs(window, chain);
				}
				res[window][chain] = forecaster[window][chain]->forecastDensity(use_sv).bottomRows(1);
				forecaster[window][chain].reset(); // free the memory by making nullptr
			}
		}
	}
  return Rcpp::wrap(res);
}
