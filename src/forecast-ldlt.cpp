#include <regforecaster.h>
#include <bvharinterrupt.h>

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
	std::vector<std::unique_ptr<bvhar::RegVarForecaster>> forecaster(num_chains);
	if (sparse && prior_type == 0) {
		for (int i = 0; i < num_chains; ++i) {
			std::unique_ptr<bvhar::LdltRecords> reg_record;
			Rcpp::List alpha_list = fit_record["alpha_record"];
			Rcpp::List a_list = fit_record["a_record"];
			Rcpp::List d_list = fit_record["d_record"];
			if (include_mean) {
				Rcpp::List c_list = fit_record["c_record"];
				reg_record.reset(new bvhar::LdltRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(c_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(d_list[i])
				));
			} else {
				reg_record.reset(new bvhar::LdltRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(d_list[i])
				));
			}
			forecaster[i].reset(new bvhar::RegVarSelectForecaster(
				*reg_record, bvhar::unvectorize(reg_record->computeActivity(level), response_mat.cols()),
				step, response_mat, var_lag, include_mean, stable, static_cast<unsigned int>(seed_chain[i])
			));
		}
	} else {
		std::string alpha_name = sparse ? "alpha_sparse_record" : "alpha_record";
		std::string a_name = sparse ? "a_sparse_record" : "a_record";
		std::string c_name = sparse ? "c_sparse_record" : "c_record";
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int i = 0; i < num_chains; i++ ) {
			std::unique_ptr<bvhar::LdltRecords> reg_record;
		#ifdef _OPENMP
			#pragma omp critical
		#endif
			{
				Rcpp::List alpha_list = fit_record[alpha_name];
				Rcpp::List a_list = fit_record[a_name];
				Rcpp::List d_list = fit_record["d_record"];
				if (include_mean) {
					Rcpp::List c_list = fit_record[c_name];
					reg_record.reset(new bvhar::LdltRecords(
						Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
						Rcpp::as<Eigen::MatrixXd>(c_list[i]),
						Rcpp::as<Eigen::MatrixXd>(a_list[i]),
						Rcpp::as<Eigen::MatrixXd>(d_list[i])
					));
				} else {
					reg_record.reset(new bvhar::LdltRecords(
						Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
						Rcpp::as<Eigen::MatrixXd>(a_list[i]),
						Rcpp::as<Eigen::MatrixXd>(d_list[i])
					));
				}
			}
			forecaster[i].reset(new bvhar::RegVarForecaster(
				*reg_record, step, response_mat, var_lag, include_mean, stable, static_cast<unsigned int>(seed_chain[i])
			));
		}
	}
	std::vector<Eigen::MatrixXd> res(num_chains);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int chain = 0; chain < num_chains; chain++) {
		res[chain] = forecaster[chain]->forecastDensity();
		forecaster[chain].reset(); // free the memory by making nullptr
	}
	return Rcpp::wrap(res);
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
	std::vector<std::unique_ptr<bvhar::RegVharForecaster>> forecaster(num_chains);
	if (sparse && prior_type == 0) {
		for (int i = 0; i < num_chains; ++i) {
			std::unique_ptr<bvhar::LdltRecords> reg_record;
			Rcpp::List alpha_list = fit_record["phi_record"];
			Rcpp::List a_list = fit_record["a_record"];
			Rcpp::List d_list = fit_record["d_record"];
			if (include_mean) {
				Rcpp::List c_list = fit_record["c_record"];
				reg_record.reset(new bvhar::LdltRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(c_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(d_list[i])
				));
			} else {
				reg_record.reset(new bvhar::LdltRecords(
					Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
					Rcpp::as<Eigen::MatrixXd>(a_list[i]),
					Rcpp::as<Eigen::MatrixXd>(d_list[i])
				));
			}
			forecaster[i].reset(new bvhar::RegVharSelectForecaster(
				*reg_record, bvhar::unvectorize(reg_record->computeActivity(level), response_mat.cols()),
				step, response_mat, HARtrans, month, include_mean, stable, static_cast<unsigned int>(seed_chain[i])
			));
		}
	} else {
		std::string alpha_name = sparse ? "phi_sparse_record" : "phi_record";
		std::string a_name = sparse ? "a_sparse_record" : "a_record";
		std::string c_name = sparse ? "c_sparse_record" : "c_record";
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int i = 0; i < num_chains; i++ ) {
			std::unique_ptr<bvhar::LdltRecords> reg_record;
		#ifdef _OPENMP
			#pragma omp critical
		#endif
			{
				Rcpp::List alpha_list = fit_record[alpha_name];
				Rcpp::List a_list = fit_record[a_name];
				Rcpp::List d_list = fit_record["d_record"];
				if (include_mean) {
					Rcpp::List c_list = fit_record[c_name];
					reg_record.reset(new bvhar::LdltRecords(
						Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
						Rcpp::as<Eigen::MatrixXd>(c_list[i]),
						Rcpp::as<Eigen::MatrixXd>(a_list[i]),
						Rcpp::as<Eigen::MatrixXd>(d_list[i])
					));
				} else {
					reg_record.reset(new bvhar::LdltRecords(
						Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
						Rcpp::as<Eigen::MatrixXd>(a_list[i]),
						Rcpp::as<Eigen::MatrixXd>(d_list[i])
					));
				}
			}
			forecaster[i].reset(new bvhar::RegVharForecaster(
				*reg_record, step, response_mat, HARtrans, month, include_mean, stable, static_cast<unsigned int>(seed_chain[i])
			));
		}
	}
	std::vector<Eigen::MatrixXd> res(num_chains);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int chain = 0; chain < num_chains; chain++) {
		res[chain] = forecaster[chain]->forecastDensity();
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
	std::vector<std::vector<std::unique_ptr<bvhar::RegVarForecaster>>> forecaster(num_horizon);
	for (auto &reg_forecast : forecaster) {
		reg_forecast.resize(num_chains);
		for (auto &ptr : reg_forecast) {
			ptr = nullptr;
		}
	}
	int sparse_type = (level > 0) ? 0 : prior_type; // use CI if level > 0
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		if (sparse && sparse_type == 0) {
			std::unique_ptr<bvhar::LdltRecords> reg_record;
			Rcpp::List d_list = fit_record["d_record"];
			Rcpp::List alpha_list = fit_record["alpha_record"];
			Rcpp::List a_list = fit_record["a_record"];
			for (int i = 0; i < num_chains; ++i) {
				if (include_mean) {
					Rcpp::List c_list = fit_record["c_record"];
					reg_record.reset(new bvhar::LdltRecords(
						Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
						Rcpp::as<Eigen::MatrixXd>(c_list[i]),
						Rcpp::as<Eigen::MatrixXd>(a_list[i]),
						Rcpp::as<Eigen::MatrixXd>(d_list[i])
					));
				} else {
					reg_record.reset(new bvhar::LdltRecords(
						Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
						Rcpp::as<Eigen::MatrixXd>(a_list[i]),
						Rcpp::as<Eigen::MatrixXd>(d_list[i])
					));
				}
				forecaster[0][i].reset(new bvhar::RegVarSelectForecaster(
					*reg_record, bvhar::unvectorize(reg_record->computeActivity(level), dim),
					step, roll_y0[0], lag, include_mean, stable, static_cast<unsigned int>(seed_forecast[i])
				));
			}
		} else {
			std::string alpha_name = sparse ? "alpha_sparse_record" : "alpha_record";
			std::string a_name = sparse ? "a_sparse_record" : "a_record";
			std::string c_name = sparse ? "c_sparse_record" : "c_record";
		#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthreads)
		#endif
			for (int i = 0; i < num_chains; ++i) {
				std::unique_ptr<bvhar::LdltRecords> reg_record;
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					Rcpp::List d_list = fit_record["d_record"];
					Rcpp::List alpha_list = fit_record[alpha_name];
					Rcpp::List a_list = fit_record[a_name];
					if (include_mean) {
						Rcpp::List c_list = fit_record[c_name];
						reg_record.reset(new bvhar::LdltRecords(
							Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
							Rcpp::as<Eigen::MatrixXd>(c_list[i]),
							Rcpp::as<Eigen::MatrixXd>(a_list[i]),
							Rcpp::as<Eigen::MatrixXd>(d_list[i])
						));
					} else {
						reg_record.reset(new bvhar::LdltRecords(
							Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
							Rcpp::as<Eigen::MatrixXd>(a_list[i]),
							Rcpp::as<Eigen::MatrixXd>(d_list[i])
						));
					}
				}
				forecaster[0][i].reset(new bvhar::RegVarForecaster(
					*reg_record, step, roll_y0[0], lag, include_mean, stable, static_cast<unsigned int>(seed_forecast[i])
				));
			}
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::MinnParams<bvhar::RegParams> minn_params(
					num_iter, design, roll_y0[window],
					param_reg, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::LdltInits sv_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::MinnReg(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::SsvsParams<bvhar::RegParams> ssvs_params(
					num_iter, design, roll_y0[window],
					param_reg, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::SsvsReg(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::HorseshoeParams<bvhar::RegParams> horseshoe_params(
					num_iter, design, roll_y0[window],
					param_reg, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HsInits hs_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::HorseshoeReg(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 4: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::HierminnParams<bvhar::RegParams> minn_params(
					num_iter, design, roll_y0[window],
					param_reg,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HierminnInits minn_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::HierminnReg(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 5: {
			for (int window = 0; window < num_horizon; ++window) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::NgParams<bvhar::RegParams> ng_params(
					num_iter, design, roll_y0[window],
					param_reg,
					grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::NgInits ng_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::NgReg(ng_params, ng_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 6: {
			for (int window = 0; window < num_horizon; ++window) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
				bvhar::DlParams<bvhar::RegParams> dl_params(
					num_iter, design, roll_y0[window],
					param_reg,
					grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::GlInits dl_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::DlReg(dl_params, dl_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		default: {
			Rf_error("not specified");
		}
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
	std::vector<std::vector<std::unique_ptr<bvhar::RegVharForecaster>>> forecaster(num_horizon);
	for (auto &reg_forecast : forecaster) {
		reg_forecast.resize(num_chains);
		for (auto &ptr : reg_forecast) {
			ptr = nullptr;
		}
	}
	int sparse_type = (level > 0) ? 0 : prior_type; // use CI if level > 0
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		if (sparse && sparse_type == 0) {
			std::unique_ptr<bvhar::LdltRecords> reg_record;
			Rcpp::List d_list = fit_record["d_record"];
			Rcpp::List phi_list = fit_record["phi_record"];
			Rcpp::List a_list = fit_record["a_record"];
			for (int i = 0; i < num_chains; i++) {
				if (include_mean) {
					Rcpp::List c_list = fit_record["c_record"];
					reg_record.reset(new bvhar::LdltRecords(
						Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
						Rcpp::as<Eigen::MatrixXd>(c_list[i]),
						Rcpp::as<Eigen::MatrixXd>(a_list[i]),
						Rcpp::as<Eigen::MatrixXd>(d_list[i])
					));
				} else {
					reg_record.reset(new bvhar::LdltRecords(
						Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
						Rcpp::as<Eigen::MatrixXd>(a_list[i]),
						Rcpp::as<Eigen::MatrixXd>(d_list[i])
					));
				}
				forecaster[0][i].reset(new bvhar::RegVharSelectForecaster(
					*reg_record, bvhar::unvectorize(reg_record->computeActivity(level), dim),
					step, roll_y0[0], har_trans, month, include_mean, stable, static_cast<unsigned int>(seed_forecast[i])
				));
			}
		} else {
			std::string phi_name = sparse ? "phi_sparse_record" : "phi_record";
			std::string a_name = sparse ? "a_sparse_record" : "a_record";
			std::string c_name = sparse ? "c_sparse_record" : "c_record";
		#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthreads)
		#endif
			for (int i = 0; i < num_chains; ++i) {
				std::unique_ptr<bvhar::LdltRecords> reg_record;
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					Rcpp::List d_list = fit_record["d_record"];
					Rcpp::List phi_list = fit_record[phi_name];
					Rcpp::List a_list = fit_record[a_name];
					if (include_mean) {
						Rcpp::List c_list = fit_record[c_name];
						reg_record.reset(new bvhar::LdltRecords(
							Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
							Rcpp::as<Eigen::MatrixXd>(c_list[i]),
							Rcpp::as<Eigen::MatrixXd>(a_list[i]),
							Rcpp::as<Eigen::MatrixXd>(d_list[i])
						));
					} else {
						reg_record.reset(new bvhar::LdltRecords(
							Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
							Rcpp::as<Eigen::MatrixXd>(a_list[i]),
							Rcpp::as<Eigen::MatrixXd>(d_list[i])
						));
					}
				}
				forecaster[0][i].reset(new bvhar::RegVharForecaster(
					*reg_record, step, roll_y0[0], har_trans, month, include_mean, stable, static_cast<unsigned int>(seed_forecast[i])
				));
			}
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::MinnParams<bvhar::RegParams> minn_params(
					num_iter, design, roll_y0[window],
					param_reg, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::LdltInits reg_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::MinnReg(minn_params, reg_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::SsvsParams<bvhar::RegParams> ssvs_params(
					num_iter, design, roll_y0[window],
					param_reg, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::SsvsReg(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::HorseshoeParams<bvhar::RegParams> horseshoe_params(
					num_iter, design, roll_y0[window],
					param_reg, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HsInits hs_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::HorseshoeReg(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 4: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::HierminnParams<bvhar::RegParams> minn_params(
					num_iter, design, roll_y0[window],
					param_reg,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HierminnInits minn_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::HierminnReg(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 5: {
			for (int window = 0; window < num_horizon; ++window) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::NgParams<bvhar::RegParams> ng_params(
					num_iter, design, roll_y0[window],
					param_reg,
					grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::NgInits ng_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::NgReg(ng_params, ng_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 6: {
			for (int window = 0; window < num_horizon; ++window) {
				Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::DlParams<bvhar::RegParams> dl_params(
					num_iter, design, roll_y0[window],
					param_reg,
					grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::GlInits dl_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::DlReg(dl_params, dl_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				roll_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		default: {
			Rf_error("not specified");
		}
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
	std::vector<std::vector<std::unique_ptr<bvhar::RegVarForecaster>>> forecaster(num_horizon);
	for (auto &reg_forecast : forecaster) {
		reg_forecast.resize(num_chains);
		for (auto &ptr : reg_forecast) {
			ptr = nullptr;
		}
	}
	int sparse_type = (level > 0) ? 0 : prior_type; // use CI if level > 0
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		if (sparse && sparse_type == 0) {
			std::unique_ptr<bvhar::LdltRecords> reg_record;
			Rcpp::List d_list = fit_record["d_record"];
			Rcpp::List alpha_list = fit_record["alpha_record"];
			Rcpp::List a_list = fit_record["a_record"];
			for (int i = 0; i < num_chains; ++i) {
				if (include_mean) {
					Rcpp::List c_list = fit_record["c_record"];
					reg_record.reset(new bvhar::LdltRecords(
						Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
						Rcpp::as<Eigen::MatrixXd>(c_list[i]),
						Rcpp::as<Eigen::MatrixXd>(a_list[i]),
						Rcpp::as<Eigen::MatrixXd>(d_list[i])
					));
				} else {
					reg_record.reset(new bvhar::LdltRecords(
						Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
						Rcpp::as<Eigen::MatrixXd>(a_list[i]),
						Rcpp::as<Eigen::MatrixXd>(d_list[i])
					));
				}
				forecaster[0][i].reset(new bvhar::RegVarSelectForecaster(
					*reg_record, bvhar::unvectorize(reg_record->computeActivity(level), dim),
					step, expand_y0[0], lag, include_mean, stable, static_cast<unsigned int>(seed_forecast[i])
				));
			}
		} else {
			std::string alpha_name = sparse ? "alpha_sparse_record" : "alpha_record";
			std::string a_name = sparse ? "a_sparse_record" : "a_record";
			std::string c_name = sparse ? "c_sparse_record" : "c_record";
		#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthreads)
		#endif
			for (int i = 0; i < num_chains; i++) {
				std::unique_ptr<bvhar::LdltRecords> reg_record;
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					Rcpp::List d_list = fit_record["d_record"];
					Rcpp::List alpha_list = fit_record[alpha_name];
					Rcpp::List a_list = fit_record[a_name];
					if (include_mean) {
						Rcpp::List c_list = fit_record[c_name];
						reg_record.reset(new bvhar::LdltRecords(
							Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
							Rcpp::as<Eigen::MatrixXd>(c_list[i]),
							Rcpp::as<Eigen::MatrixXd>(a_list[i]),
							Rcpp::as<Eigen::MatrixXd>(d_list[i])
						));
					} else {
						reg_record.reset(new bvhar::LdltRecords(
							Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
							Rcpp::as<Eigen::MatrixXd>(a_list[i]),
							Rcpp::as<Eigen::MatrixXd>(d_list[i])
						));
					}
				}
				forecaster[0][i].reset(new bvhar::RegVarForecaster(
					*reg_record, step, expand_y0[0], lag, include_mean, stable, static_cast<unsigned int>(seed_forecast[i])
				));
			}
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::MinnParams<bvhar::RegParams> minn_params(
					num_iter, design, expand_y0[window],
					param_reg, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::LdltInits sv_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::MinnReg(minn_params, sv_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::SsvsParams<bvhar::RegParams> ssvs_params(
					num_iter, design, expand_y0[window],
					param_reg, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::SsvsReg(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::HorseshoeParams<bvhar::RegParams> horseshoe_params(
					num_iter, design, expand_y0[window],
					param_reg, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HsInits hs_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::HorseshoeReg(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 4: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::HierminnParams<bvhar::RegParams> minn_params(
					num_iter, design, expand_y0[window],
					param_reg,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HierminnInits minn_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::HierminnReg(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 5: {
			for (int window = 0; window < num_horizon; ++window) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::NgParams<bvhar::RegParams> ng_params(
					num_iter, design, expand_y0[window],
					param_reg,
					grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::NgInits ng_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::NgReg(ng_params, ng_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 6: {
			for (int window = 0; window < num_horizon; ++window) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
				bvhar::DlParams<bvhar::RegParams> dl_params(
					num_iter, design, expand_y0[window],
					param_reg,
					grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::GlInits dl_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::DlReg(dl_params, dl_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		default: {
			Rf_error("not specified");
		}
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
	std::vector<std::vector<std::unique_ptr<bvhar::RegVharForecaster>>> forecaster(num_horizon);
	for (auto &reg_forecast : forecaster) {
		reg_forecast.resize(num_chains);
		for (auto &ptr : reg_forecast) {
			ptr = nullptr;
		}
	}
	int sparse_type = (level > 0) ? 0 : prior_type; // use CI if level > 0
	bool use_fit = fit_record.size() > 0;
	if (use_fit) {
		if (sparse && sparse_type == 0) {
			std::unique_ptr<bvhar::LdltRecords> reg_record;
			Rcpp::List d_list = fit_record["d_record"];
			Rcpp::List phi_list = fit_record["phi_record"];
			Rcpp::List a_list = fit_record["a_record"];
			for (int i = 0; i < num_chains; ++i) {
				if (include_mean) {
					Rcpp::List c_list = fit_record["c_record"];
					reg_record.reset(new bvhar::LdltRecords(
						Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
						Rcpp::as<Eigen::MatrixXd>(c_list[i]),
						Rcpp::as<Eigen::MatrixXd>(a_list[i]),
						Rcpp::as<Eigen::MatrixXd>(d_list[i])
					));
				} else {
					reg_record.reset(new bvhar::LdltRecords(
						Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
						Rcpp::as<Eigen::MatrixXd>(a_list[i]),
						Rcpp::as<Eigen::MatrixXd>(d_list[i])
					));
				}
				forecaster[0][i].reset(new bvhar::RegVharSelectForecaster(
					*reg_record, bvhar::unvectorize(reg_record->computeActivity(level), dim),
					step, expand_y0[0], har_trans, month, include_mean, stable, static_cast<unsigned int>(seed_forecast[i])
				));
			}
		} else {
			std::string phi_name = sparse ? "phi_sparse_record" : "phi_record";
			std::string a_name = sparse ? "a_sparse_record" : "a_record";
			std::string c_name = sparse ? "c_sparse_record" : "c_record";
		#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthreads)
		#endif
			for (int i = 0; i < num_chains; i++) {
				std::unique_ptr<bvhar::LdltRecords> reg_record;
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					Rcpp::List d_list = fit_record["d_record"];
					Rcpp::List phi_list = fit_record[phi_name];
					Rcpp::List a_list = fit_record[a_name];
					if (include_mean) {
						Rcpp::List c_list = fit_record[c_name];
						reg_record.reset(new bvhar::LdltRecords(
							Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
							Rcpp::as<Eigen::MatrixXd>(c_list[i]),
							Rcpp::as<Eigen::MatrixXd>(a_list[i]),
							Rcpp::as<Eigen::MatrixXd>(d_list[i])
						));
					} else {
						reg_record.reset(new bvhar::LdltRecords(
							Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
							Rcpp::as<Eigen::MatrixXd>(a_list[i]),
							Rcpp::as<Eigen::MatrixXd>(d_list[i])
						));
					}
				}
				forecaster[0][i].reset(new bvhar::RegVharForecaster(
					*reg_record, step, expand_y0[0], har_trans, month, include_mean, stable, static_cast<unsigned int>(seed_forecast[i])
				));
			}
		}
	}
	std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	Eigen::MatrixXd lpl_record(num_horizon, num_chains);
	switch (prior_type) {
		case 1: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::MinnParams<bvhar::RegParams> minn_params(
					num_iter, design, expand_y0[window],
					param_reg, param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::LdltInits reg_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::MinnReg(minn_params, reg_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 2: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::SsvsParams<bvhar::RegParams> ssvs_params(
					num_iter, design, expand_y0[window],
					param_reg, grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::SsvsInits ssvs_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::SsvsReg(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 3: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::HorseshoeParams<bvhar::RegParams> horseshoe_params(
					num_iter, design, expand_y0[window],
					param_reg, grp_id, grp_mat,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HsInits hs_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::HorseshoeReg(horseshoe_params, hs_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 4: {
			for (int window = 0; window < num_horizon; window++) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::HierminnParams<bvhar::RegParams> minn_params(
					num_iter, design, expand_y0[window],
					param_reg,
					own_id, cross_id, grp_mat,
					param_prior,
					param_intercept, include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::HierminnInits minn_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::HierminnReg(minn_params, minn_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 5: {
			for (int window = 0; window < num_horizon; ++window) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::NgParams<bvhar::RegParams> ng_params(
					num_iter, design, expand_y0[window],
					param_reg,
					grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::NgInits ng_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::NgReg(ng_params, ng_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		case 6: {
			for (int window = 0; window < num_horizon; ++window) {
				Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
				bvhar::DlParams<bvhar::RegParams> dl_params(
					num_iter, design, expand_y0[window],
					param_reg,
					grp_id, grp_mat,
					param_prior, param_intercept,
					include_mean
				);
				for (int chain = 0; chain < num_chains; chain++) {
					Rcpp::List init_spec = param_init[chain];
					bvhar::GlInits dl_inits(init_spec);
					reg_objs[window][chain].reset(new bvhar::DlReg(dl_params, dl_inits, static_cast<unsigned int>(seed_chain(window, chain))));
				}
				expand_mat[window].resize(0, 0); // free the memory
			}
			break;
		}
		default: {
			Rf_error("not specified");
		}
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
