#include "minnforecaster.h"
#include "bvharinterrupt.h"

//' Forecasting BVAR(p)
//' 
//' @param object `bvarmn` or `bvarflat` object
//' @param step Integer, Step to forecast
//' @param num_sim Integer, number to simulate parameters from posterior distribution
//' @details
//' n-step ahead forecasting using BVAR(p) recursively.
//' 
//' For given number of simulation (`num_sim`),
//' 
//' 1. Generate \eqn{(A^{(b)}, \Sigma_e^{(b)}) \sim MIW} (posterior)
//' 2. Recursively, \eqn{j = 1, \ldots, h} (`step`)
//'     - Point forecast: Use \eqn{\hat{A}}
//'     - Predictive distribution: Again generate \eqn{\tilde{Y}_{n + j}^{(b)} \sim A^{(b)}, \Sigma_e^{(b)} \sim MN}
//'     - tilde notation indicates simulated ones
//' 
//' @references
//' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' 
//' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
//' 
//' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
//' 
//' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). [https://doi:10.1080/01621459.2018.1437043](https://doi:10.1080/01621459.2018.1437043)
//' 
//' Karlsson, S. (2013). *Chapter 15 Forecasting with Bayesian Vector Autoregression*. Handbook of Economic Forecasting, 2, 791–897. doi:[10.1016/b978-0-444-62731-5.00015-4](https://doi.org/10.1016/B978-0-444-62731-5.00015-4)
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvar(Rcpp::List object, int step, int num_sim) {
	// (int num_chains, int var_lag, int step, Eigen::MatrixXd response_mat,
	// 											 Eigen::MatrixXd alpha_record, Eigen::MatrixXd sig_record,
	// 											 bool include_mean, int nthreads) {
  // int num_sim = num_chains > 1 ? alpha_record.rows() / num_chains : alpha_record.rows();
	// std::vector<std::unique_ptr<bvhar::BvarForecaster>> forecaster(num_chains);
	// for (int i = 0; i < num_chains; ++i) {
	// 	bvhar::MinnRecords mn_record(
	// 		alpha_record.middleRows(i * num_sim, num_sim),
	// 		sig_record.middleRows(i * num_sim, num_sim)
	// 	);
	// 	forecaster[i].reset(new bvhar::BvarForecaster(mn_record, step, response_mat, var_lag, include_mean));
	// }
	if (!object.inherits("bvarmn") && !object.inherits("bvarflat")) {
    Rcpp::stop("'object' must be bvarmn or bvarflat object.");
  }
  Eigen::MatrixXd response_mat = object["y0"]; // Y0
  Eigen::MatrixXd posterior_mean_mat = object["coefficients"]; // Ahat = posterior mean of MN
  Eigen::MatrixXd posterior_prec_mat = object["mn_prec"]; // vhat = posterior precision of MN to compute SE
  Eigen::MatrixXd posterior_scale = object["covmat"]; // Sighat = posterior scale of IW
  double posterior_shape = object["iw_shape"]; // posterior shape of IW
  int var_lag = object["p"]; // VAR(p)
	bool include_mean = Rcpp::as<std::string>(object["type"]) == "const";
	bvhar::MinnFit mn_fit(posterior_mean_mat, posterior_prec_mat, posterior_scale, posterior_shape);
	std::unique_ptr<bvhar::BvarForecaster> forecaster(new bvhar::BvarForecaster(mn_fit, step, response_mat, var_lag, num_sim, include_mean));
	forecaster->forecastDensity();
	return forecaster->returnForecast();
// 	std::vector<Eigen::MatrixXd> res(num_chains);
// #ifdef _OPENMP
// 	#pragma omp parallel for num_threads(nthreads)
// #endif
// 	for (int chain = 0; chain < num_chains; ++chain) {
// 		res[chain] = forecaster[chain]->forecastDensity();
// 		forecaster[chain].reset();
// 	}
// 	return Rcpp::wrap(res);
}

//' Forecasting Bayesian VHAR
//' 
//' @param object `bvharmn` object
//' @param step Integer, Step to forecast
//' @param num_sim Integer, number to simulate parameters from posterior distribution
//' @details
//' n-step ahead forecasting using VHAR recursively.
//' 
//' For given number of simulation (`num_sim`),
//' 
//' 1. Generate \eqn{(\Phi^{(b)}, \Sigma_e^{(b)}) \sim MIW} (posterior)
//' 2. Recursively, \eqn{j = 1, \ldots, h} (`step`)
//'     - Point forecast: Use \eqn{\hat\Phi}
//'     - Predictive distribution: Again generate \eqn{\tilde{Y}_{n + j}^{(b)} \sim \Phi^{(b)}, \Sigma_e^{(b)} \sim MN}
//'     - tilde notation indicates simulated ones
//' 
//' @references Kim, Y. G., and Baek, C. (n.d.). *Bayesian vector heterogeneous autoregressive modeling*. submitted.
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvharmn(Rcpp::List object, int step, int num_sim) {
	// (int num_chains, int month, int step, Eigen::MatrixXd response_mat, Eigen::MatrixXd har_trans,
	// 											 		Eigen::MatrixXd phi_record, Eigen::MatrixXd sig_record,
	// 											 		bool include_mean, int nthreads) {
	// int num_sim = num_chains > 1 ? phi_record.rows() / num_chains : phi_record.rows();
	// std::vector<std::unique_ptr<bvhar::BvharForecaster>> forecaster(num_chains);
	// for (int i = 0; i < num_chains; ++i) {
	// 	bvhar::MinnRecords mn_record(
	// 		phi_record.middleRows(i * num_sim, num_sim),
	// 		sig_record.middleRows(i * num_sim, num_sim)
	// 	);
	// 	forecaster[i].reset(new bvhar::BvharForecaster(mn_record, step, response_mat, har_trans, month, include_mean));
	// }
	if (!object.inherits("bvharmn")) {
    Rcpp::stop("'object' must be bvharmn object.");
  }
  Eigen::MatrixXd response_mat = object["y0"]; // Y0
  Eigen::MatrixXd posterior_mean_mat = object["coefficients"]; // Phihat = posterior mean of MN: h x m, h = 3m (+ 1)
  Eigen::MatrixXd posterior_prec_mat = object["mn_prec"]; // Psihat = posterior precision of MN to compute SE: h x h
  Eigen::MatrixXd posterior_mn_scale_u = posterior_prec_mat.inverse();
  Eigen::MatrixXd posterior_scale = object["covmat"]; // Sighat = posterior scale of IW: m x m
  double posterior_shape = object["iw_shape"]; // posterior shape of IW
  Eigen::MatrixXd HARtrans = object["HARtrans"]; // HAR transformation: h x k0, k0 = 22m (+ 1)
  Eigen::MatrixXd transformed_prec_mat = HARtrans.transpose() * posterior_prec_mat.inverse() * HARtrans; // to compute SE: play a role V in BVAR
  int month = object["month"];
	bool include_mean = Rcpp::as<std::string>(object["type"]) == "const";
	bvhar::MinnFit mn_fit(posterior_mean_mat, posterior_prec_mat, posterior_scale, posterior_shape);
	std::unique_ptr<bvhar::BvharForecaster> forecaster(new bvhar::BvharForecaster(mn_fit, step, response_mat, HARtrans, month, num_sim, include_mean));
	forecaster->forecastDensity();
	return forecaster->returnForecast();
// 	std::vector<Eigen::MatrixXd> res(num_chains);
// #ifdef _OPENMP
// 	#pragma omp parallel for num_threads(nthreads)
// #endif
// 	for (int chain = 0; chain < num_chains; ++chain) {
// 		res[chain] = forecaster[chain]->forecastDensity();
// 		forecaster[chain].reset();
// 	}
// 	return Rcpp::wrap(res);
}

//' Out-of-Sample Forecasting of BVAR based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVAR with Minnesota prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag BVAR order
//' @param bayes_spec List, BVAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_bvar(Eigen::MatrixXd y, int lag, Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test, int nthreads) {
// Rcpp::List roll_bvar(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
// 										 Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test,
// 										 Eigen::MatrixXi seed_chain, int nthreads, int chunk_size) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1; // longest forecast horizon
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
	// std::vector<std::vector<std::unique_ptr<bvhar::Minnesota>>> mn_objs(num_horizon);
	// for (auto &mn_chain : mn_objs) {
	// 	mn_chain.resize(num_chains);
	// 	for (auto &ptr : mn_chain) {
	// 		ptr = nullptr;
	// 	}
	// }
	// std::vector<std::vector<std::unique_ptr<bvhar::BvarForecaster>>> forecaster(num_horizon);
	// for (auto &mn_forecast : forecaster) {
	// 	mn_forecast.resize(num_chains);
	// 	for (auto &ptr : mn_forecast) {
	// 		ptr = nullptr;
	// 	}
	// }
	// bool use_fit = fit_record.size() > 0;
	// if (use_fit) {
	// 	Rcpp::List alpha_list = fit_record["alpha_record"];
	// 	Rcpp::List sig_list = fit_record["sigma_record"];
	// 	for (int i = 0; i < num_chains; ++i) {
	// 		bvhar::MinnRecords mn_record(
	// 			Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
	// 			Rcpp::as<Eigen::MatrixXd>(sig_list[i])
	// 		);
	// 		forecaster[0][i].reset(new bvhar::BvarForecaster(mn_record, step, roll_y0[0], lag, include_mean));
	// 	}
	// }
	// std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	// bvhar::BvarSpec mn_spec(bayes_spec);
	// for (int window = 0; window < num_horizon; ++window) {
	// 	Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
	// 		lag, mn_spec._sigma, mn_spec._lambda,
	// 		mn_spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
	// 		include_mean
	// 	);
	// 	Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
	// 		Eigen::VectorXd::LinSpaced(lag, 1, lag),
	// 		mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
	// 	);
	// 	for (int chain = 0; chain < num_chains; ++chain) {
	// 		Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
	// 		// 	dummy_response = build_ydummy(
	// 		// 	lag, spec._sigma,
	// 		// 	spec._lambda, spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
	// 		// 	const_term
	// 		// );
	// 		mn_objs[window][chain].reset(new bvhar::Minnesota(num_iter, design, roll_y0[window], x_dummy, y_dummy, static_cast<unsigned int>(seed_chain(window, chain))));
	// 		mn_objs[window][chain]->computePosterior();
	// 	}
	// 	roll_mat[window].resize(0, 0);
	// }
	// auto run_conj = [&](int window, int chain) {
	// 	bvhar::bvharinterrupt();
	// 	for (int i = 0; i < num_iter; ++i) {
	// 		if (bvhar::bvharinterrupt::is_interrupted()) {
	// 			// 
	// 			break;
	// 		}
	// 		mn_objs[window][chain]->doPosteriorDraws();
	// 	}
	// 	bvhar::MinnRecords mn_record = mn_objs[window][chain]->returnMinnRecords(num_burn, thinning);
	// 	forecaster[window][chain].reset(new bvhar::BvarForecaster(
	// 		mn_record, step, roll_y0[window], lag, include_mean
	// 	));
	// 	mn_objs[window][chain].reset();
	// };
	// if (num_chains == 1) {
	// #ifdef _OPENMP
	// 	#pragma omp parallel for num_threads(nthreads)
	// #endif
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		if (!use_fit || window != 0) {
	// 			run_conj(window, 0);
	// 		}
	// 		res[window][0] = forecaster[window][0]->forecastDensity().bottomRows(1);
	// 		forecaster[window][0].reset();
	// 	}
	// } else {
	// #ifdef _OPENMP
	// 	#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	// #endif
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		for (int chain = 0; chain < num_chains; ++chain) {
	// 			if (!use_fit || window != 0) {
	// 				run_conj(window, chain);
	// 			}
	// 			res[window][chain] = forecaster[window][chain]->forecastDensity().bottomRows(1);
	// 			forecaster[window][chain].reset();
	// 		}
	// 	}
	// }
	// return Rcpp::wrap(res);

// #ifdef _OPENMP
//   Eigen::setNbThreads(nthreads);
// #endif
	// Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	// tot_mat << y,
	// 					y_test;
	// std::vector<Eigen::MatrixXd> roll_mat(num_horizon);
	// std::vector<Eigen::MatrixXd> roll_y0(num_horizon);
	// for (int i = 0; i < num_horizon; i++) {
	// 	roll_mat[i] = tot_mat.middleRows(i, num_window);
	// 	roll_y0[i] = bvhar::build_y0(roll_mat[i], lag, lag + 1);
	// }
	std::vector<std::unique_ptr<bvhar::MinnBvar>> mn_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		bvhar::BvarSpec mn_spec(bayes_spec);
		mn_objs[i] = std::unique_ptr<bvhar::MinnBvar>(new bvhar::MinnBvar(roll_mat[i], lag, mn_spec, include_mean));
	}
	std::vector<std::unique_ptr<bvhar::BvarForecaster>> forecaster(num_horizon);
	std::vector<Eigen::MatrixXd> res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int window = 0; window < num_horizon; window++) {
		bvhar::MinnFit mn_fit = mn_objs[window]->returnMinnFit();
		forecaster[window].reset(new bvhar::BvarForecaster(mn_fit, step, roll_y0[window], lag, 1, include_mean));
		res[window] = forecaster[window]->returnPoint().bottomRows(1);
		mn_objs[window].reset(); // free the memory by making nullptr
		forecaster[window].reset(); // free the memory by making nullptr
	}
	return std::accumulate(
		res.begin() + 1, res.end(), res[0],
		[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
			Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
			concat_mat << acc,
										curr;
			return concat_mat;
		}
	);
}

//' Out-of-Sample Forecasting of BVAR based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVAR with Flat prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag BVAR order
//' @param bayes_spec List, BVAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_bvarflat(Eigen::MatrixXd y, int lag, Eigen::MatrixXd U, bool include_mean, int step, Eigen::MatrixXd y_test, int nthreads) {
// Rcpp::List roll_bvarflat(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
// 												 Eigen::MatrixXd U, bool include_mean, int step, Eigen::MatrixXd y_test,
// 												 Eigen::MatrixXi seed_chain, int nthreads, int chunk_size) {
  // if (!bayes_spec.inherits("bvharspec")) {
  //   Rcpp::stop("'object' must be bvharspec object.");
  // }
  // Rcpp::Function fit("bvar_flat");
  int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1; // longest forecast horizon
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
	// std::vector<std::vector<std::unique_ptr<bvhar::MinnFlat>>> mn_objs(num_horizon);
	// for (auto &mn_chain : mn_objs) {
	// 	mn_chain.resize(num_chains);
	// 	for (auto &ptr : mn_chain) {
	// 		ptr = nullptr;
	// 	}
	// }
	// std::vector<std::vector<std::unique_ptr<bvhar::BvarForecaster>>> forecaster(num_horizon);
	// for (auto &mn_forecast : forecaster) {
	// 	mn_forecast.resize(num_chains);
	// 	for (auto &ptr : mn_forecast) {
	// 		ptr = nullptr;
	// 	}
	// }
	// bool use_fit = fit_record.size() > 0;
	// if (use_fit) {
	// 	Rcpp::List alpha_list = fit_record["alpha_record"];
	// 	Rcpp::List sig_list = fit_record["sigma_record"];
	// 	for (int i = 0; i < num_chains; ++i) {
	// 		bvhar::MinnRecords mn_record(
	// 			Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
	// 			Rcpp::as<Eigen::MatrixXd>(sig_list[i])
	// 		);
	// 		forecaster[0][i].reset(new bvhar::BvarForecaster(mn_record, step, roll_y0[0], lag, include_mean));
	// 	}
	// }
	// std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	// for (int window = 0; window < num_horizon; ++window) {
	// 	for (int chain = 0; chain < num_chains; ++chain) {
	// 		Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], lag, include_mean);
	// 		// 	dummy_response = build_ydummy(
	// 		// 	lag, spec._sigma,
	// 		// 	spec._lambda, spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
	// 		// 	const_term
	// 		// );
	// 		mn_objs[window][chain].reset(new bvhar::MinnFlat(num_iter, design, roll_y0[window], U, static_cast<unsigned int>(seed_chain(window, chain))));
	// 		mn_objs[window][chain]->computePosterior();
	// 	}
	// 	roll_mat[window].resize(0, 0);
	// }
	// auto run_conj = [&](int window, int chain) {
	// 	bvhar::bvharinterrupt();
	// 	for (int i = 0; i < num_iter; ++i) {
	// 		if (bvhar::bvharinterrupt::is_interrupted()) {
	// 			// 
	// 			break;
	// 		}
	// 		mn_objs[window][chain]->doPosteriorDraws();
	// 	}
	// 	bvhar::MinnRecords mn_record = mn_objs[window][chain]->returnMinnRecords(num_burn, thinning);
	// 	forecaster[window][chain].reset(new bvhar::BvarForecaster(
	// 		mn_record, step, roll_y0[window], lag, include_mean
	// 	));
	// 	mn_objs[window][chain].reset();
	// };
	// if (num_chains == 1) {
	// #ifdef _OPENMP
	// 	#pragma omp parallel for num_threads(nthreads)
	// #endif
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		if (!use_fit || window != 0) {
	// 			run_conj(window, 0);
	// 		}
	// 		res[window][0] = forecaster[window][0]->forecastDensity().bottomRows(1);
	// 		forecaster[window][0].reset();
	// 	}
	// } else {
	// #ifdef _OPENMP
	// 	#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	// #endif
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		for (int chain = 0; chain < num_chains; ++chain) {
	// 			if (!use_fit || window != 0) {
	// 				run_conj(window, chain);
	// 			}
	// 			res[window][chain] = forecaster[window][chain]->forecastDensity().bottomRows(1);
	// 			forecaster[window][chain].reset();
	// 		}
	// 	}
	// }
	// return Rcpp::wrap(res);

	std::vector<std::unique_ptr<bvhar::MinnFlat>> mn_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd x = bvhar::build_x0(roll_mat[i], lag, include_mean);
		mn_objs[i].reset(new bvhar::MinnFlat(x, roll_y0[i], U));
	}
	std::vector<std::unique_ptr<bvhar::BvarForecaster>> forecaster(num_horizon);
	std::vector<Eigen::MatrixXd> res(num_horizon);
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int window = 0; window < num_horizon; window++) {
		bvhar::MinnFit mn_fit = mn_objs[window]->returnMinnFit();
		forecaster[window].reset(new bvhar::BvarForecaster(mn_fit, step, roll_y0[window], lag, 1, include_mean));
		res[window] = forecaster[window]->returnPoint().bottomRows(1);
		mn_objs[window].reset(); // free the memory by making nullptr
		forecaster[window].reset(); // free the memory by making nullptr
	}
	return std::accumulate(
		res.begin() + 1, res.end(), res[0],
		[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
			Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
			concat_mat << acc,
										curr;
			return concat_mat;
		}
	);
	// 
  // Eigen::MatrixXd roll_mat = y; // same size as y
  // Rcpp::List bvar_mod = fit(roll_mat, lag, bayes_spec, include_mean);
  // Rcpp::List bvar_pred = forecast_bvar(bvar_mod, step, 1);
  // Eigen::MatrixXd y_pred = bvar_pred["posterior_mean"]; // step x m
  // Eigen::MatrixXd res(num_horizon, dim);
  // res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  // for (int i = 1; i < num_horizon; i++) {
  //   roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
  //   roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
  //   bvar_mod = fit(roll_mat, lag, bayes_spec, include_mean);
  //   bvar_pred = forecast_bvar(bvar_mod, step, 1);
  //   y_pred = bvar_pred["posterior_mean"];
  //   res.row(i) = y_pred.row(step - 1);
  // }
  // return res;
}

//' Out-of-Sample Forecasting of BVHAR based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVHAR with Minnesota prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param har `r lifecycle::badge("experimental")` Numeric vector for weekly and monthly order.
//' @param bayes_spec List, BVHAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_bvhar(Eigen::MatrixXd y, int week, int month, Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test, int nthreads) {
// Rcpp::List roll_bvhar(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
// 											Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test,
// 											Eigen::MatrixXi seed_chain, int nthreads, int chunk_size) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1; // longest forecast horizon
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
	// std::vector<std::vector<std::unique_ptr<bvhar::Minnesota>>> mn_objs(num_horizon);
	// for (auto &mn_chain : mn_objs) {
	// 	mn_chain.resize(num_chains);
	// 	for (auto &ptr : mn_chain) {
	// 		ptr = nullptr;
	// 	}
	// }
	std::vector<std::unique_ptr<bvhar::MinnBvhar>> mn_objs(num_horizon);
	// std::vector<std::vector<std::unique_ptr<bvhar::BvharForecaster>>> forecaster(num_horizon);
	// for (auto &mn_forecast : forecaster) {
	// 	mn_forecast.resize(num_chains);
	// 	for (auto &ptr : mn_forecast) {
	// 		ptr = nullptr;
	// 	}
	// }
	// bool use_fit = fit_record.size() > 0;
	// if (use_fit) {
	// 	Rcpp::List phi_list = fit_record["phi_record"];
	// 	Rcpp::List sig_list = fit_record["sigma_record"];
	// 	for (int i = 0; i < num_chains; ++i) {
	// 		bvhar::MinnRecords mn_record(
	// 			Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
	// 			Rcpp::as<Eigen::MatrixXd>(sig_list[i])
	// 		);
	// 		forecaster[0][i].reset(new bvhar::BvharForecaster(mn_record, step, roll_y0[0], har_trans, month, include_mean));
	// 	}
	// }
	// std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	if (bayes_spec.containsElementNamed("delta")) {
		bvhar::BvarSpec mn_spec(bayes_spec);
		for (int i = 0; i < num_horizon; ++i) {
			mn_objs[i].reset(new bvhar::MinnBvharS(roll_mat[i], week, month, mn_spec, include_mean));
		}
		// Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
		// 	Eigen::VectorXd::LinSpaced(3, 1, 3),
		// 	mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
		// );
		// Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
		// 	3, mn_spec._sigma, mn_spec._lambda,
		// 	mn_spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
		// 	include_mean
		// );
		// for (int window = 0; window < num_horizon; ++window) {
		// 	for (int chain = 0; chain < num_chains; ++chain) {
		// 		Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
		// 		mn_objs[window][chain].reset(new bvhar::Minnesota(num_iter, design, roll_y0[window], x_dummy, y_dummy, static_cast<unsigned int>(seed_chain(window, chain))));
		// 		mn_objs[window][chain]->computePosterior();
		// 	}
		// 	roll_mat[window].resize(0, 0);
		// }
	} else {
		bvhar::BvharSpec mn_spec(bayes_spec);
		// Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
		// 	Eigen::VectorXd::LinSpaced(3, 1, 3),
		// 	mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
		// );
		// Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
		// 	3, mn_spec._sigma, mn_spec._lambda,
		// 	mn_spec._daily, mn_spec._weekly, mn_spec._monthly,
		// 	include_mean
		// );
		for (int i = 0; i < num_horizon; ++i) {
			bvhar::BvharSpec mn_spec(bayes_spec);
			mn_objs[i].reset(new bvhar::MinnBvharL(roll_mat[i], week, month, mn_spec, include_mean));
		}
		// for (int window = 0; window < num_horizon; ++window) {
		// 	for (int chain = 0; chain < num_chains; ++chain) {
		// 		Eigen::MatrixXd design = bvhar::build_x0(roll_mat[window], month, include_mean) * har_trans.transpose();
		// 		mn_objs[window][chain].reset(new bvhar::Minnesota(num_iter, design, roll_y0[window], x_dummy, y_dummy, static_cast<unsigned int>(seed_chain(window, chain))));
		// 		mn_objs[window][chain]->computePosterior();
		// 	}
		// 	roll_mat[window].resize(0, 0);
		// }
	}
	// auto run_conj = [&](int window, int chain) {
	// 	bvhar::bvharinterrupt();
	// 	for (int i = 0; i < num_iter; ++i) {
	// 		if (bvhar::bvharinterrupt::is_interrupted()) {
	// 			// 
	// 			break;
	// 		}
	// 		mn_objs[window][chain]->doPosteriorDraws();
	// 	}
	// 	bvhar::MinnRecords mn_record = mn_objs[window][chain]->returnMinnRecords(num_burn, thinning);
	// 	forecaster[window][chain].reset(new bvhar::BvharForecaster(
	// 		mn_record, step, roll_y0[window], har_trans, month, include_mean
	// 	));
	// 	mn_objs[window][chain].reset();
	// };
	// if (num_chains == 1) {
	// #ifdef _OPENMP
	// 	#pragma omp parallel for num_threads(nthreads)
	// #endif
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		if (!use_fit || window != 0) {
	// 			run_conj(window, 0);
	// 		}
	// 		res[window][0] = forecaster[window][0]->forecastDensity().bottomRows(1);
	// 		forecaster[window][0].reset();
	// 	}
	// } else {
	// #ifdef _OPENMP
	// 	#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	// #endif
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		for (int chain = 0; chain < num_chains; ++chain) {
	// 			if (!use_fit || window != 0) {
	// 				run_conj(window, chain);
	// 			}
	// 			res[window][chain] = forecaster[window][chain]->forecastDensity().bottomRows(1);
	// 			forecaster[window][chain].reset();
	// 		}
	// 	}
	// }
	// return Rcpp::wrap(res);
	std::vector<std::unique_ptr<bvhar::BvharForecaster>> forecaster(num_horizon);
	std::vector<Eigen::MatrixXd> res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int window = 0; window < num_horizon; window++) {
		bvhar::MinnFit mn_fit = mn_objs[window]->returnMinnFit();
		forecaster[window].reset(new bvhar::BvharForecaster(mn_fit, step, roll_y0[window], har_trans, month, 1, include_mean));
		res[window] = forecaster[window]->returnPoint().bottomRows(1);
		mn_objs[window].reset(); // free the memory by making nullptr
		forecaster[window].reset(); // free the memory by making nullptr
	}
	return std::accumulate(
		res.begin() + 1, res.end(), res[0],
		[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
			Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
			concat_mat << acc,
										curr;
			return concat_mat;
		}
	);
}

//' Out-of-Sample Forecasting of BVAR based on Expanding Window
//' 
//' This function conducts an expanding window forecasting of BVAR with Minnesota prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag BVAR order
//' @param bayes_spec List, BVAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd expand_bvar(Eigen::MatrixXd y, int lag, Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test, int nthreads) {
// Rcpp::List expand_bvar(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
// 											 Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test,
// 											 Eigen::MatrixXi seed_chain, int nthreads, int chunk_size) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
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
	// std::vector<std::vector<std::unique_ptr<bvhar::Minnesota>>> mn_objs(num_horizon);
	// for (auto &mn_chain : mn_objs) {
	// 	mn_chain.resize(num_chains);
	// 	for (auto &ptr : mn_chain) {
	// 		ptr = nullptr;
	// 	}
	// }
	// std::vector<std::vector<std::unique_ptr<bvhar::BvarForecaster>>> forecaster(num_horizon);
	// for (auto &mn_forecast : forecaster) {
	// 	mn_forecast.resize(num_chains);
	// 	for (auto &ptr : mn_forecast) {
	// 		ptr = nullptr;
	// 	}
	// }
	// bool use_fit = fit_record.size() > 0;
	// if (use_fit) {
	// 	Rcpp::List alpha_list = fit_record["alpha_record"];
	// 	Rcpp::List sig_list = fit_record["sigma_record"];
	// 	for (int i = 0; i < num_chains; ++i) {
	// 		bvhar::MinnRecords mn_record(
	// 			Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
	// 			Rcpp::as<Eigen::MatrixXd>(sig_list[i])
	// 		);
	// 		forecaster[0][i].reset(new bvhar::BvarForecaster(mn_record, step, expand_y0[0], lag, include_mean));
	// 	}
	// }
	// std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	// bvhar::BvarSpec mn_spec(bayes_spec);
	// for (int window = 0; window < num_horizon; ++window) {
	// 	Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
	// 		lag, mn_spec._sigma, mn_spec._lambda,
	// 		mn_spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
	// 		include_mean
	// 	);
	// 	Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
	// 		Eigen::VectorXd::LinSpaced(lag, 1, lag),
	// 		mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
	// 	);
	// 	for (int chain = 0; chain < num_chains; ++chain) {
	// 		Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
	// 		// 	dummy_response = build_ydummy(
	// 		// 	lag, spec._sigma,
	// 		// 	spec._lambda, spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
	// 		// 	const_term
	// 		// );
	// 		mn_objs[window][chain].reset(new bvhar::Minnesota(num_iter, design, expand_y0[window], x_dummy, y_dummy, static_cast<unsigned int>(seed_chain(window, chain))));
	// 		mn_objs[window][chain]->computePosterior();
	// 	}
	// 	expand_mat[window].resize(0, 0);
	// }
	std::vector<std::unique_ptr<bvhar::MinnBvar>> mn_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		// 
		bvhar::BvarSpec mn_spec(bayes_spec);
		mn_objs[i].reset(new bvhar::MinnBvar(expand_mat[i], lag, mn_spec, include_mean));
	}
	std::vector<std::unique_ptr<bvhar::BvarForecaster>> forecaster(num_horizon);
	std::vector<Eigen::MatrixXd> res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int window = 0; window < num_horizon; window++) {
		bvhar::MinnFit mn_fit = mn_objs[window]->returnMinnFit();
		forecaster[window].reset(new bvhar::BvarForecaster(mn_fit, step, expand_y0[window], lag, 1, include_mean));
		res[window] = forecaster[window]->returnPoint().bottomRows(1);
		mn_objs[window].reset(); // free the memory by making nullptr
		forecaster[window].reset(); // free the memory by making nullptr
	}
	return std::accumulate(
		res.begin() + 1, res.end(), res[0],
		[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
			Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
			concat_mat << acc,
										curr;
			return concat_mat;
		}
	);
	// auto run_conj = [&](int window, int chain) {
	// 	bvhar::bvharinterrupt();
	// 	for (int i = 0; i < num_iter; ++i) {
	// 		if (bvhar::bvharinterrupt::is_interrupted()) {
	// 			// 
	// 			break;
	// 		}
	// 		mn_objs[window][chain]->doPosteriorDraws();
	// 	}
	// 	bvhar::MinnRecords mn_record = mn_objs[window][chain]->returnMinnRecords(num_burn, thinning);
	// 	forecaster[window][chain].reset(new bvhar::BvarForecaster(
	// 		mn_record, step, expand_y0[window], lag, include_mean
	// 	));
	// 	mn_objs[window][chain].reset();
	// };
	// if (num_chains == 1) {
	// #ifdef _OPENMP
	// 	#pragma omp parallel for num_threads(nthreads)
	// #endif
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		if (!use_fit || window != 0) {
	// 			run_conj(window, 0);
	// 		}
	// 		res[window][0] = forecaster[window][0]->forecastDensity().bottomRows(1);
	// 		forecaster[window][0].reset();
	// 	}
	// } else {
	// #ifdef _OPENMP
	// 	#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	// #endif
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		for (int chain = 0; chain < num_chains; ++chain) {
	// 			if (!use_fit || window != 0) {
	// 				run_conj(window, chain);
	// 			}
	// 			res[window][chain] = forecaster[window][chain]->forecastDensity().bottomRows(1);
	// 			forecaster[window][chain].reset();
	// 		}
	// 	}
	// }
	// return Rcpp::wrap(res);
}

//' Out-of-Sample Forecasting of BVAR based on Expanding Window
//' 
//' This function conducts an expanding window forecasting of BVAR with Flat prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag BVAR order
//' @param bayes_spec List, BVAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd expand_bvarflat(Eigen::MatrixXd y, int lag, Eigen::MatrixXd U, bool include_mean, int step, Eigen::MatrixXd y_test, int nthreads) {
// Rcpp::List expand_bvarflat(Eigen::MatrixXd y, int lag, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
// 													 Eigen::MatrixXd U, bool include_mean, int step, Eigen::MatrixXd y_test,
// 													 Eigen::MatrixXi seed_chain, int nthreads, int chunk_size) {
  // if (!bayes_spec.inherits("bvharspec")) {
  //   Rcpp::stop("'object' must be bvharspec object.");
  // }
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
	// std::vector<std::vector<std::unique_ptr<bvhar::MinnFlat>>> mn_objs(num_horizon);
	// for (auto &mn_chain : mn_objs) {
	// 	mn_chain.resize(num_chains);
	// 	for (auto &ptr : mn_chain) {
	// 		ptr = nullptr;
	// 	}
	// }
	// std::vector<std::vector<std::unique_ptr<bvhar::BvarForecaster>>> forecaster(num_horizon);
	// for (auto &mn_forecast : forecaster) {
	// 	mn_forecast.resize(num_chains);
	// 	for (auto &ptr : mn_forecast) {
	// 		ptr = nullptr;
	// 	}
	// }
	// bool use_fit = fit_record.size() > 0;
	// if (use_fit) {
	// 	Rcpp::List alpha_list = fit_record["alpha_record"];
	// 	Rcpp::List sig_list = fit_record["sigma_record"];
	// 	for (int i = 0; i < num_chains; ++i) {
	// 		bvhar::MinnRecords mn_record(
	// 			Rcpp::as<Eigen::MatrixXd>(alpha_list[i]),
	// 			Rcpp::as<Eigen::MatrixXd>(sig_list[i])
	// 		);
	// 		forecaster[0][i].reset(new bvhar::BvarForecaster(mn_record, step, expand_y0[0], lag, include_mean));
	// 	}
	// }
	// std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	// for (int window = 0; window < num_horizon; ++window) {
	// 	for (int chain = 0; chain < num_chains; ++chain) {
	// 		Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], lag, include_mean);
	// 		// 	dummy_response = build_ydummy(
	// 		// 	lag, spec._sigma,
	// 		// 	spec._lambda, spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
	// 		// 	const_term
	// 		// );
	// 		mn_objs[window][chain].reset(new bvhar::MinnFlat(num_iter, design, expand_y0[window], U, static_cast<unsigned int>(seed_chain(window, chain))));
	// 		mn_objs[window][chain]->computePosterior();
	// 	}
	// 	expand_mat[window].resize(0, 0);
	// }
	// auto run_conj = [&](int window, int chain) {
	// 	bvhar::bvharinterrupt();
	// 	for (int i = 0; i < num_iter; ++i) {
	// 		if (bvhar::bvharinterrupt::is_interrupted()) {
	// 			// 
	// 			break;
	// 		}
	// 		mn_objs[window][chain]->doPosteriorDraws();
	// 	}
	// 	bvhar::MinnRecords mn_record = mn_objs[window][chain]->returnMinnRecords(num_burn, thinning);
	// 	forecaster[window][chain].reset(new bvhar::BvarForecaster(
	// 		mn_record, step, expand_y0[window], lag, include_mean
	// 	));
	// 	mn_objs[window][chain].reset();
	// };
	// if (num_chains == 1) {
	// #ifdef _OPENMP
	// 	#pragma omp parallel for num_threads(nthreads)
	// #endif
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		if (!use_fit || window != 0) {
	// 			run_conj(window, 0);
	// 		}
	// 		res[window][0] = forecaster[window][0]->forecastDensity().bottomRows(1);
	// 		forecaster[window][0].reset();
	// 	}
	// } else {
	// #ifdef _OPENMP
	// 	#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	// #endif
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		for (int chain = 0; chain < num_chains; ++chain) {
	// 			if (!use_fit || window != 0) {
	// 				run_conj(window, chain);
	// 			}
	// 			res[window][chain] = forecaster[window][chain]->forecastDensity().bottomRows(1);
	// 			forecaster[window][chain].reset();
	// 		}
	// 	}
	// }
	// return Rcpp::wrap(res);

	std::vector<std::unique_ptr<bvhar::MinnFlat>> mn_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd x = bvhar::build_x0(expand_mat[i], lag, include_mean);
		mn_objs[i].reset(new bvhar::MinnFlat(x, expand_y0[i], U));
	}
	std::vector<std::unique_ptr<bvhar::BvarForecaster>> forecaster(num_horizon);
	std::vector<Eigen::MatrixXd> res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int window = 0; window < num_horizon; window++) {
		bvhar::MinnFit mn_fit = mn_objs[window]->returnMinnFit();
		forecaster[window].reset(new bvhar::BvarForecaster(mn_fit, step, expand_y0[window], lag, 1, include_mean));
		res[window] = forecaster[window]->returnPoint().bottomRows(1);
		mn_objs[window].reset(); // free the memory by making nullptr
		forecaster[window].reset(); // free the memory by making nullptr
	}
	return std::accumulate(
		res.begin() + 1, res.end(), res[0],
		[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
			Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
			concat_mat << acc,
										curr;
			return concat_mat;
		}
	);
	// 
  // Rcpp::Function fit("bvar_flat");
  // int window = y.rows();
  // int dim = y.cols();
  // int num_test = y_test.rows();
  // int num_iter = num_test - step + 1; // longest forecast horizon
  // Eigen::MatrixXd expand_mat(window + num_iter, dim); // train + h-step forecast points
  // expand_mat.block(0, 0, window, dim) = y;
  // Rcpp::List bvar_mod = fit(y, lag, bayes_spec, include_mean);
  // Rcpp::List bvar_pred = forecast_bvar(bvar_mod, step, 1);
  // Eigen::MatrixXd y_pred = bvar_pred["posterior_mean"]; // step x m
  // Eigen::MatrixXd res(num_iter, dim);
  // res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  // for (int i = 1; i < num_iter; i++) {
  //   expand_mat.row(window + i - 1) = y_test.row(i - 1); // expanding window
  //   bvar_mod = fit(
  //     expand_mat.block(0, 0, window + i, dim),
  //     lag, 
  //     bayes_spec, 
  //     include_mean
  //   );
  //   bvar_pred = forecast_bvar(bvar_mod, step, 1);
  //   y_pred = bvar_pred["posterior_mean"];
  //   res.row(i) = y_pred.row(step - 1);
  // }
  // return res;
}

//' Out-of-Sample Forecasting of BVHAR based on Expanding Window
//' 
//' This function conducts an expanding window forecasting of BVHAR with Minnesota prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param har `r lifecycle::badge("experimental")` Numeric vector for weekly and monthly order.
//' @param bayes_spec List, BVHAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd expand_bvhar(Eigen::MatrixXd y, int week, int month, Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test, int nthreads) {
// Rcpp::List expand_bvhar(Eigen::MatrixXd y, int week, int month, int num_chains, int num_iter, int num_burn, int thinning, Rcpp::List fit_record,
// 												Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test,
// 												Eigen::MatrixXi seed_chain, int nthreads, int chunk_size) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
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
	// std::vector<std::vector<std::unique_ptr<bvhar::Minnesota>>> mn_objs(num_horizon);
	// for (auto &mn_chain : mn_objs) {
	// 	mn_chain.resize(num_chains);
	// 	for (auto &ptr : mn_chain) {
	// 		ptr = nullptr;
	// 	}
	// }
	// // std::vector<std::unique_ptr<bvhar::MinnBvhar>> mn_objs(num_horizon);
	// std::vector<std::vector<std::unique_ptr<bvhar::BvharForecaster>>> forecaster(num_horizon);
	// for (auto &mn_forecast : forecaster) {
	// 	mn_forecast.resize(num_chains);
	// 	for (auto &ptr : mn_forecast) {
	// 		ptr = nullptr;
	// 	}
	// }
	// bool use_fit = fit_record.size() > 0;
	// if (use_fit) {
	// 	Rcpp::List phi_list = fit_record["phi_record"];
	// 	Rcpp::List sig_list = fit_record["sigma_record"];
	// 	for (int i = 0; i < num_chains; ++i) {
	// 		bvhar::MinnRecords mn_record(
	// 			Rcpp::as<Eigen::MatrixXd>(phi_list[i]),
	// 			Rcpp::as<Eigen::MatrixXd>(sig_list[i])
	// 		);
	// 		forecaster[0][i].reset(new bvhar::BvharForecaster(mn_record, step, expand_y0[0], har_trans, month, include_mean));
	// 	}
	// }
	// std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	// if (bayes_spec.containsElementNamed("delta")) {
	// 	// for (int i = 0; i < num_horizon; ++i) {
	// 	// 	mn_objs[i] = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharS(roll_mat[i], week, month, mn_spec, include_mean));
	// 	// }
	// 	bvhar::BvarSpec mn_spec(bayes_spec);
	// 	Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
	// 		Eigen::VectorXd::LinSpaced(3, 1, 3),
	// 		mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
	// 	);
	// 	Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
	// 		3, mn_spec._sigma, mn_spec._lambda,
	// 		mn_spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
	// 		include_mean
	// 	);
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		for (int chain = 0; chain < num_chains; ++chain) {
	// 			Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
	// 			mn_objs[window][chain].reset(new bvhar::Minnesota(num_iter, design, expand_y0[window], x_dummy, y_dummy, static_cast<unsigned int>(seed_chain(window, chain))));
	// 			mn_objs[window][chain]->computePosterior();
	// 		}
	// 		expand_mat[window].resize(0, 0);
	// 	}
	// } else {
	// 	bvhar::BvharSpec mn_spec(bayes_spec);
	// 	Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
	// 		Eigen::VectorXd::LinSpaced(3, 1, 3),
	// 		mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
	// 	);
	// 	Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
	// 		3, mn_spec._sigma, mn_spec._lambda,
	// 		mn_spec._daily, mn_spec._weekly, mn_spec._monthly,
	// 		include_mean
	// 	);
	// 	// for (int i = 0; i < num_horizon; ++i) {
	// 	// 	bvhar::BvharSpec mn_spec(bayes_spec);
	// 	// 	mn_objs[i] = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharL(roll_mat[i], week, month, mn_spec, include_mean));
	// 	// }
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		for (int chain = 0; chain < num_chains; ++chain) {
	// 			Eigen::MatrixXd design = bvhar::build_x0(expand_mat[window], month, include_mean) * har_trans.transpose();
	// 			mn_objs[window][chain].reset(new bvhar::Minnesota(num_iter, design, expand_y0[window], x_dummy, y_dummy, static_cast<unsigned int>(seed_chain(window, chain))));
	// 			mn_objs[window][chain]->computePosterior();
	// 		}
	// 		expand_mat[window].resize(0, 0);
	// 	}
	// }
	std::vector<std::unique_ptr<bvhar::MinnBvhar>> mn_objs(num_horizon);
	// bool minn_short = true;
	if (bayes_spec.containsElementNamed("delta")) {
		// minn_short = false;
		for (int i = 0; i < num_horizon; ++i) {
			bvhar::BvarSpec mn_spec(bayes_spec);
			mn_objs[i] = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharS(expand_mat[i], week, month, mn_spec, include_mean));
		}
	} else {
		for (int i = 0; i < num_horizon; ++i) {
			bvhar::BvharSpec mn_spec(bayes_spec);
			mn_objs[i] = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharL(expand_mat[i], week, month, mn_spec, include_mean));
		}
	}
	std::vector<std::unique_ptr<bvhar::BvharForecaster>> forecaster(num_horizon);
	std::vector<Eigen::MatrixXd> res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int window = 0; window < num_horizon; window++) {
		bvhar::MinnFit mn_fit = mn_objs[window]->returnMinnFit();
		forecaster[window].reset(new bvhar::BvharForecaster(mn_fit, step, expand_y0[window], har_trans, month, 1, include_mean));
		res[window] = forecaster[window]->returnPoint().bottomRows(1);
		mn_objs[window].reset(); // free the memory by making nullptr
		forecaster[window].reset(); // free the memory by making nullptr
	}
	return std::accumulate(
		res.begin() + 1, res.end(), res[0],
		[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
			Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
			concat_mat << acc,
										curr;
			return concat_mat;
		}
	);
	// auto run_conj = [&](int window, int chain) {
	// 	bvhar::bvharinterrupt();
	// 	for (int i = 0; i < num_iter; ++i) {
	// 		if (bvhar::bvharinterrupt::is_interrupted()) {
	// 			// 
	// 			break;
	// 		}
	// 		mn_objs[window][chain]->doPosteriorDraws();
	// 	}
	// 	bvhar::MinnRecords mn_record = mn_objs[window][chain]->returnMinnRecords(num_burn, thinning);
	// 	forecaster[window][chain].reset(new bvhar::BvharForecaster(
	// 		mn_record, step, expand_y0[window], har_trans, month, include_mean
	// 	));
	// 	mn_objs[window][chain].reset();
	// };
	// if (num_chains == 1) {
	// #ifdef _OPENMP
	// 	#pragma omp parallel for num_threads(nthreads)
	// #endif
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		if (!use_fit || window != 0) {
	// 			run_conj(window, 0);
	// 		}
	// 		res[window][0] = forecaster[window][0]->forecastDensity().bottomRows(1);
	// 		forecaster[window][0].reset();
	// 	}
	// } else {
	// #ifdef _OPENMP
	// 	#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthreads)
	// #endif
	// 	for (int window = 0; window < num_horizon; ++window) {
	// 		for (int chain = 0; chain < num_chains; ++chain) {
	// 			if (!use_fit || window != 0) {
	// 				run_conj(window, chain);
	// 			}
	// 			res[window][chain] = forecaster[window][chain]->forecastDensity().bottomRows(1);
	// 			forecaster[window][chain].reset();
	// 		}
	// 	}
	// }
	// return Rcpp::wrap(res);
}
