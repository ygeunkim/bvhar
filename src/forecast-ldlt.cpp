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
//' @param include_mean Include constant term?
//' @param nthreads OpenMP number of threads
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvarldlt(int num_chains, int var_lag, int step, Eigen::MatrixXd response_mat,
													 	 bool sparse, double level, Rcpp::List fit_record, int prior_type,
													 	 Eigen::VectorXi seed_chain, bool include_mean, int nthreads) {
	std::vector<std::unique_ptr<bvhar::RegVarForecaster>> forecaster(num_chains);
	if (sparse) {
		switch (prior_type) {
			case 0: {
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
						*reg_record, level, step, response_mat, var_lag, include_mean, static_cast<unsigned int>(seed_chain[i])
					));
				}
				break;
			}
			case 1: {
				Rf_error("not specified");
			}
			case 2: {
				Rcpp::List alpha_list = fit_record["alpha_record"];
				Rcpp::List a_list = fit_record["a_record"];
				Rcpp::List d_list = fit_record["d_record"];
				Rcpp::List gamma_list = fit_record["gamma_record"];
				for (int i = 0; i < num_chains; ++i) {
					std::unique_ptr<bvhar::LdltRecords> reg_record;
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
					bvhar::SsvsRecords ssvs_record(
						Rcpp::as<Eigen::MatrixXd>(gamma_list[i]),
						Eigen::MatrixXd(),
						Eigen::MatrixXd(),
						Eigen::MatrixXd()
					);
					forecaster[i].reset(new bvhar::RegVarSparseForecaster(
						*reg_record, ssvs_record, step, response_mat, var_lag, include_mean, static_cast<unsigned int>(seed_chain[i])
					));
				}
				break;
			}
			case 3: {
				Rcpp::List alpha_list = fit_record["alpha_record"];
				Rcpp::List a_list = fit_record["a_record"];
				Rcpp::List d_list = fit_record["d_record"];
				Rcpp::List kappa_list = fit_record["kappa_record"];
				for (int i = 0; i < num_chains; ++i) {
					std::unique_ptr<bvhar::LdltRecords> reg_record;
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
					bvhar::HorseshoeRecords hs_record(
						Eigen::MatrixXd(),
						Eigen::MatrixXd(),
						Eigen::VectorXd(),
						Rcpp::as<Eigen::MatrixXd>(kappa_list[i])
					);
					forecaster[i].reset(new bvhar::RegVarSparseForecaster(
						*reg_record, hs_record, step, response_mat, var_lag, include_mean, static_cast<unsigned int>(seed_chain[i])
					));
				}
				break;
			}
			default:
				Rf_error("Not defined");
				break;
		}
	} else {
		for (int i = 0; i < num_chains; i++ ) {
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
			forecaster[i].reset(new bvhar::RegVarForecaster(
				*reg_record, step, response_mat, var_lag, include_mean, static_cast<unsigned int>(seed_chain[i])
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
//' @param nthreads OpenMP number of threads 
//'
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvharldlt(int num_chains, int month, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd HARtrans,
															bool sparse, double level, Rcpp::List fit_record, int prior_type,
															Eigen::VectorXi seed_chain, bool include_mean, int nthreads) {
	std::vector<std::unique_ptr<bvhar::RegVharForecaster>> forecaster(num_chains);
	if (sparse) {
		switch (prior_type) {
			case 0: {
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
						*reg_record, level, step, response_mat, HARtrans, month, include_mean, static_cast<unsigned int>(seed_chain[i])
					));
				}
				break;
			}
			case 1: {
				Rf_error("not specified");
			}
			case 2: {
				Rcpp::List alpha_list = fit_record["phi_record"];
				Rcpp::List a_list = fit_record["a_record"];
				Rcpp::List d_list = fit_record["d_record"];
				Rcpp::List gamma_list = fit_record["gamma_record"];
				for (int i = 0; i < num_chains; ++i) {
					std::unique_ptr<bvhar::LdltRecords> reg_record;
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
					bvhar::SsvsRecords ssvs_record(
						Rcpp::as<Eigen::MatrixXd>(gamma_list[i]),
						Eigen::MatrixXd(),
						Eigen::MatrixXd(),
						Eigen::MatrixXd()
					);
					forecaster[i].reset(new bvhar::RegVharSparseForecaster(
						*reg_record, ssvs_record, step, response_mat, HARtrans, month, include_mean, static_cast<unsigned int>(seed_chain[i])
					));
				}
				break;
			}
			case 3: {
				Rcpp::List alpha_list = fit_record["phi_record"];
				Rcpp::List a_list = fit_record["a_record"];
				Rcpp::List d_list = fit_record["d_record"];
				Rcpp::List kappa_list = fit_record["kappa_record"];
				for (int i = 0; i < num_chains; ++i) {
					std::unique_ptr<bvhar::LdltRecords> reg_record;
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
					bvhar::HorseshoeRecords hs_record(
						Eigen::MatrixXd(),
						Eigen::MatrixXd(),
						Eigen::VectorXd(),
						Rcpp::as<Eigen::MatrixXd>(kappa_list[i])
					);
					forecaster[i].reset(new bvhar::RegVharSparseForecaster(
						*reg_record, hs_record, step, response_mat, HARtrans, month, include_mean, static_cast<unsigned int>(seed_chain[i])
					));
				}
				break;
			}
			default:
				Rf_error("Not defined");
				break;
		}
	} else {
		for (int i = 0; i < num_chains; i++ ) {
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
			forecaster[i].reset(new bvhar::RegVharForecaster(
				*reg_record, step, response_mat, HARtrans, month, include_mean, static_cast<unsigned int>(seed_chain[i])
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
