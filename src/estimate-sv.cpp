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
                           Eigen::VectorXi grp_id,
                           Eigen::MatrixXi grp_mat,
                           bool include_mean,
													 Eigen::VectorXi seed_chain,
                           bool display_progress, int nthreads) {
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	std::vector<std::unique_ptr<bvhar::McmcSv>> sv_objs(num_chains);
	std::vector<Rcpp::List> res(num_chains);
	switch (prior_type) {
		case 1: {
			bvhar::MinnParams minn_params(
				num_iter, x, y,
				param_sv, param_prior,
				param_intercept, include_mean
			);
			for (int i = 0; i < num_chains; i++ ) {
				Rcpp::List init_spec = param_init[i];
				bvhar::SvInits sv_inits(init_spec);
				sv_objs[i] = std::unique_ptr<bvhar::McmcSv>(new bvhar::MinnSv(minn_params, sv_inits, static_cast<unsigned int>(seed_chain[i])));
			}
			break;
		}
		case 2: {
			bvhar::SsvsParams ssvs_params(
				num_iter, x, y,
				param_sv,
				grp_id, grp_mat,
				param_prior,
				param_intercept,
				include_mean
			);
			for (int i = 0; i < num_chains; i++ ) {
				Rcpp::List init_spec = param_init[i];
				bvhar::SsvsInits ssvs_inits(init_spec);
				sv_objs[i] = std::unique_ptr<bvhar::McmcSv>(new bvhar::SsvsSv(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain[i])));
			}
			break;
		}
		case 3: {
			bvhar::HorseshoeParams horseshoe_params(
				num_iter, x, y,
				param_sv,
				grp_id, grp_mat,
				param_intercept, include_mean
			);
			for (int i = 0; i < num_chains; i++ ) {
				Rcpp::List init_spec = param_init[i];
				bvhar::HorseshoeInits hs_inits(init_spec);
				sv_objs[i] = std::unique_ptr<bvhar::McmcSv>(new bvhar::HorseshoeSv(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain[i])));
			}
			break;
		}
	}
  // Start Gibbs sampling-----------------------------------
	auto run_gibbs = [&](int chain) {
		bvhar::bvharprogress bar(num_iter, display_progress);
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					res[chain] = sv_objs[chain]->returnRecords(0, 1);
				}
				break;
			}
			bar.increment();
			if (display_progress) {
				bar.update();
			}
			sv_objs[chain]->doPosteriorDraws(); // alpha -> a -> h -> sigma_h -> h0
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			res[chain] = sv_objs[chain]->returnRecords(num_burn, thin);
		}
	};
	if (num_chains == 1) {
		run_gibbs(0);
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int chain = 0; chain < num_chains; chain++) {
			run_gibbs(chain);
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
//' @param nthreads_roll Number of threads when rolling windows
//' @param nthreads_mod Number of threads when fitting models
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bvarsv(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning,
											 Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											 Eigen::VectorXi grp_id, Eigen::MatrixXi grp_mat, bool include_mean, int step, Eigen::MatrixXd y_test,
											 Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads_roll, int nthreads_mod) {
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
	std::vector<std::vector<Rcpp::List>> records(num_horizon, std::vector<Rcpp::List>(num_chains));
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
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				// Eigen::MatrixXd response = bvhar::build_y0(roll_mat[window], lag, lag + 1);
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
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				// Eigen::MatrixXd response = bvhar::build_y0(roll_mat[window], lag, lag + 1);
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
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				// Eigen::MatrixXd response = bvhar::build_y0(roll_mat[window], lag, lag + 1);
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
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					records[window][chain] = sv_objs[window][chain]->returnRecords(0, 1);
				}
				bvhar::SvRecords sv_record(
					Rcpp::as<Eigen::MatrixXd>(records[window][chain]["alpha_record"]),
					Rcpp::as<Eigen::MatrixXd>(records[window][chain]["h_record"]),
					Rcpp::as<Eigen::MatrixXd>(records[window][chain]["a_record"]),
					Rcpp::as<Eigen::MatrixXd>(records[window][chain]["sigh_record"])
				);
				if (records[window][chain].containsElementNamed("c_record")) {
					sv_record = bvhar::SvRecords(
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["alpha_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["c_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["h_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["a_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["sigh_record"])
					);
				}
				forecaster[window][chain].reset(new bvhar::SvVarForecaster(
					sv_record, step, roll_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
				));
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			records[window][chain] = sv_objs[window][chain]->returnRecords(num_burn, thinning);
		}
		bvhar::SvRecords sv_record(
			Rcpp::as<Eigen::MatrixXd>(records[window][chain]["alpha_record"]),
			Rcpp::as<Eigen::MatrixXd>(records[window][chain]["h_record"]),
			Rcpp::as<Eigen::MatrixXd>(records[window][chain]["a_record"]),
			Rcpp::as<Eigen::MatrixXd>(records[window][chain]["sigh_record"])
		);
		if (records[window][chain].containsElementNamed("c_record")) {
			sv_record = bvhar::SvRecords(
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["alpha_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["c_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["h_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["a_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["sigh_record"])
			);
		}
		forecaster[window][chain].reset(new bvhar::SvVarForecaster(
			sv_record, step, roll_y0[window], lag, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		));
	};
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads_roll)
#endif
	for (int window = 0; window < num_horizon; window++) {
		if (num_chains == 1) {
			run_gibbs(window, 0);
			res[window][0] = forecaster[window][0]->forecastDensity().row(step - 1);
		} else {
		#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthreads_mod)
		#endif
			for (int chain = 0; chain < num_chains; chain++) {
				run_gibbs(window, chain);
				res[window][chain] = forecaster[window][chain]->forecastDensity().row(step - 1);
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
//' @param nthreads_roll Number of threads when rolling windows
//' @param nthreads_mod Number of threads when fitting models
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List roll_bvharsv(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning,
											  Rcpp::List param_sv, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type,
											  Eigen::VectorXi grp_id, Eigen::MatrixXi grp_mat, bool include_mean, int step, Eigen::MatrixXd y_test,
											  Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, int nthreads_roll, int nthreads_mod) {
  int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> roll_mat(num_horizon);
	std::vector<Eigen::MatrixXd> roll_y0(num_horizon);
	std::vector<Eigen::MatrixXd> roll_har(num_horizon);
	for (int i = 0; i < num_horizon; i++) {
		roll_mat[i] = tot_mat.middleRows(i, num_window);
		roll_y0[i] = bvhar::build_y0(roll_mat[i], month, month + 1);
		roll_har[i] = bvhar::build_vhar(dim, week, month, include_mean);
	}
	std::vector<std::vector<Rcpp::List>> records(num_horizon, std::vector<Rcpp::List>(num_chains));
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
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * roll_har[window].transpose();
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
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * roll_har[window].transpose();
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
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				// Eigen::MatrixXd response = bvhar::build_y0(roll_mat[window], lag, lag + 1);
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * roll_har[window].transpose();
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
			}
			break;
		}
	}
	auto run_gibbs = [&](int window, int chain) {
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					records[window][chain] = sv_objs[window][chain]->returnRecords(0, 1);
				}
				bvhar::SvRecords sv_record(
					Rcpp::as<Eigen::MatrixXd>(records[window][chain]["alpha_record"]),
					Rcpp::as<Eigen::MatrixXd>(records[window][chain]["h_record"]),
					Rcpp::as<Eigen::MatrixXd>(records[window][chain]["a_record"]),
					Rcpp::as<Eigen::MatrixXd>(records[window][chain]["sigh_record"])
				);
				if (records[window][chain].containsElementNamed("c_record")) {
					sv_record = bvhar::SvRecords(
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["alpha_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["c_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["h_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["a_record"]),
						Rcpp::as<Eigen::MatrixXd>(records[window][chain]["sigh_record"])
					);
				}
				forecaster[window][chain].reset(new bvhar::SvVharForecaster(
					sv_record, step, roll_y0[window], roll_har[window], month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
				));
				break;
			}
			sv_objs[window][chain]->doPosteriorDraws();
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			records[window][chain] = sv_objs[window][chain]->returnRecords(num_burn, thinning);
		}
		bvhar::SvRecords sv_record(
			Rcpp::as<Eigen::MatrixXd>(records[window][chain]["alpha_record"]),
			Rcpp::as<Eigen::MatrixXd>(records[window][chain]["h_record"]),
			Rcpp::as<Eigen::MatrixXd>(records[window][chain]["a_record"]),
			Rcpp::as<Eigen::MatrixXd>(records[window][chain]["sigh_record"])
		);
		if (records[window][chain].containsElementNamed("c_record")) {
			sv_record = bvhar::SvRecords(
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["alpha_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["c_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["h_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["a_record"]),
				Rcpp::as<Eigen::MatrixXd>(records[window][chain]["sigh_record"])
			);
		}
		forecaster[window][chain].reset(new bvhar::SvVharForecaster(
			sv_record, step, roll_y0[window], roll_har[window], month, include_mean, static_cast<unsigned int>(seed_forecast[chain])
		));
	};
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads_roll)
#endif
	for (int window = 0; window < num_horizon; window++) {
		if (num_chains == 1) {
			run_gibbs(window, 0);
			res[window][0] = forecaster[window][0]->forecastDensity().row(step - 1);
		} else {
		#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthreads_mod)
		#endif
			for (int chain = 0; chain < num_chains; chain++) {
				run_gibbs(window, chain);
				res[window][chain] = forecaster[window][chain]->forecastDensity().row(step - 1);
			}
		}
	}
  return Rcpp::wrap(res);
}
