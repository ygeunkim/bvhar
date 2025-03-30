#include <bvhar/mniw>

//' BVAR(p) Point Estimates based on Minnesota Prior
//' 
//' Point estimates for posterior distribution
//' 
//' @param y Time series data
//' @param lag VAR order
//' @param bayes_spec BVAR Minnesota specification
//' @param include_mean Constant term
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_mn(Eigen::MatrixXd y, int lag, Rcpp::List bayes_spec, bool include_mean) {
	bvhar::BvarSpec mn_spec(bayes_spec);
	std::unique_ptr<bvhar::MinnBvar> mn_obj(new bvhar::MinnBvar(y, lag, mn_spec, include_mean));
	return mn_obj->returnMinnRes();
}

//' BVHAR Point Estimates based on Minnesota Prior
//' 
//' Point estimates for posterior distribution
//' 
//' @param y Time series data
//' @param week VHAR week order
//' @param month VHAR month order
//' @param bayes_spec BVHAR Minnesota specification
//' @param include_mean Constant term
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvhar_mn(Eigen::MatrixXd y, int week, int month, Rcpp::List bayes_spec, bool include_mean) {
	std::unique_ptr<bvhar::MinnBvhar> mn_obj;
	if (bayes_spec.containsElementNamed("delta")) {
		bvhar::BvarSpec bvhar_spec(bayes_spec);
		mn_obj.reset(new bvhar::MinnBvharS(y, week, month, bvhar_spec, include_mean));
	} else {
		bvhar::BvharSpec bvhar_spec(bayes_spec);
		mn_obj.reset(new bvhar::MinnBvharL(y, week, month, bvhar_spec, include_mean));
	}
	return mn_obj->returnMinnRes();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_mh(int num_chains, int num_iter, int num_burn, int thin,
														Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd x_dummy, Eigen::MatrixXd y_dummy,
														Rcpp::List param_prior, Rcpp::List param_init,
														Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
	std::vector<std::unique_ptr<bvhar::MhMinnesota>> mn_objs(num_chains);
	std::vector<Rcpp::List> res(num_chains);
	Rcpp::List lambda_spec = param_prior["lambda"];
	Rcpp::List psi_spec = param_prior["sigma"];
	bvhar::MhMinnSpec mn_spec(lambda_spec, psi_spec);
  for (int i = 0; i < num_chains; ++i) {
		Rcpp::List init_spec = param_init[i];
		bvhar::MhMinnInits mn_init(init_spec);
		mn_objs[i].reset(new bvhar::MhMinnesota(num_iter, mn_spec, mn_init, x, y, x_dummy, y_dummy, static_cast<unsigned int>(seed_chain[i])));
		mn_objs[i]->computePosterior();
	}
	auto run_mh = [&](int chain) {
		bvhar::bvharprogress bar(num_iter, display_progress);
		bvhar::bvharinterrupt();
		for (int i = 0; i < num_iter; i++) {
			if (bvhar::bvharinterrupt::is_interrupted()) {
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					res[chain] = mn_objs[chain]->returnRecords(0, 1);
				}
				break;
			}
			bar.increment();
			if (display_progress) {
				bar.update();
			}
			mn_objs[chain]->doPosteriorDraws();
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			res[chain] = mn_objs[chain]->returnRecords(num_burn, thin);
		}
	};
	if (num_chains == 1) {
		run_mh(0);
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int chain = 0; chain < num_chains; chain++) {
			run_mh(chain);
		}
	}
	return Rcpp::wrap(res);
}

//' BVAR(p) Point Estimates based on Nonhierarchical Matrix Normal Prior
//' 
//' Point estimates for Ghosh et al. (2018) nonhierarchical model for BVAR.
//' 
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param U Positive definite matrix, covariance matrix corresponding to the column of the model parameter B
//' 
//' @details
//' In Ghosh et al. (2018), there are many models for BVAR such as hierarchical or non-hierarchical.
//' Among these, this function chooses the most simple non-hierarchical matrix normal prior in Section 3.1.
//' 
//' @references
//' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). [https://doi:10.1080/01621459.2018.1437043](https://doi:10.1080/01621459.2018.1437043)
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_mn_flat(Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd U) {
  if (U.rows() != x.cols()) {
    Rcpp::stop("Wrong dimension: U");
  }
  if (U.cols() != x.cols()) {
    Rcpp::stop("Wrong dimension: U");
  }
	std::unique_ptr<bvhar::MinnFlat> mn_obj(new bvhar::MinnFlat(x, y, U));
	return mn_obj->returnMinnRes();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_mniw(int num_chains, int num_iter, int num_burn, int thin,
												 const Eigen::MatrixXd& mn_mean, const Eigen::MatrixXd& mn_prec,
												 const Eigen::MatrixXd& iw_scale, double iw_shape,
												 Eigen::VectorXi seed_chain, bool display_progress, int nthreads) {
	std::vector<std::unique_ptr<bvhar::McmcMniw>> mn_objs(num_chains);
	for (int i = 0; i < num_chains; ++i) {
		bvhar::MinnFit mn_fit(mn_mean, mn_prec, iw_scale, iw_shape);
		mn_objs[i].reset(new bvhar::McmcMniw(num_iter, mn_fit, static_cast<unsigned int>(seed_chain[i])));
	}
	std::vector<Rcpp::List> res(num_chains);
	auto run_conj = [&](int chain) {
		bvhar::bvharprogress bar(num_iter, display_progress);
		for (int i = 0; i < num_iter; ++i) {
			bar.increment();
			bar.update();
			mn_objs[chain]->doPosteriorDraws();
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			res[chain] = mn_objs[chain]->returnRecords(num_burn, thin);
		}
	};
	if (num_chains == 1) {
		run_conj(0);
	} else {
	#ifdef _OPENMP
		#pragma omp parallel for num_threads(nthreads)
	#endif
		for (int chain = 0; chain < num_chains; ++chain) {
			run_conj(chain);
		}
	}
	return Rcpp::wrap(res);
}

//' Forecasting BVAR(p)
//' 
//' @param object A `bvarmn` or `bvarflat` object
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
//' Karlsson, S. (2013). *Chapter 15 Forecasting with Bayesian Vector Autoregression*. Handbook of Economic Forecasting, 2, 791-897. doi:[10.1016/b978-0-444-62731-5.00015-4](https://doi.org/10.1016/B978-0-444-62731-5.00015-4)
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvar(Rcpp::List object, int step, int num_sim, unsigned int seed) {
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
	std::unique_ptr<bvhar::BvarForecaster> forecaster(new bvhar::BvarForecaster(mn_fit, step, response_mat, var_lag, num_sim, include_mean, seed));
	forecaster->forecastDensity();
	return forecaster->returnForecast();
}

//' Forecasting Bayesian VHAR
//' 
//' @param object A `bvharmn` object
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
//' @references Kim, Y. G., and Baek, C. (2024). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation, 94(6), 1139-1157.
//' @noRd
// [[Rcpp::export]]
Rcpp::List forecast_bvharmn(Rcpp::List object, int step, int num_sim, unsigned int seed) {
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
	std::unique_ptr<bvhar::BvharForecaster> forecaster(new bvhar::BvharForecaster(mn_fit, step, response_mat, HARtrans, month, num_sim, include_mean, seed));
	forecaster->forecastDensity();
	return forecaster->returnForecast();
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
Eigen::MatrixXd roll_bvar(Eigen::MatrixXd y, int lag, Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test, Eigen::VectorXi seed_forecast, int nthreads) {
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
		forecaster[window].reset(new bvhar::BvarForecaster(mn_fit, step, roll_y0[window], lag, 1, include_mean, static_cast<unsigned int>(seed_forecast[window])));
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
Eigen::MatrixXd roll_bvarflat(Eigen::MatrixXd y, int lag, Eigen::MatrixXd U, bool include_mean, int step, Eigen::MatrixXd y_test, Eigen::VectorXi seed_forecast, int nthreads) {
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
		forecaster[window].reset(new bvhar::BvarForecaster(mn_fit, step, roll_y0[window], lag, 1, include_mean, static_cast<unsigned int>(seed_forecast[window])));
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
Eigen::MatrixXd roll_bvhar(Eigen::MatrixXd y, int week, int month, Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test, Eigen::VectorXi seed_forecast, int nthreads) {
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
	std::vector<std::unique_ptr<bvhar::MinnBvhar>> mn_objs(num_horizon);
	if (bayes_spec.containsElementNamed("delta")) {
		bvhar::BvarSpec mn_spec(bayes_spec);
		for (int i = 0; i < num_horizon; ++i) {
			mn_objs[i].reset(new bvhar::MinnBvharS(roll_mat[i], week, month, mn_spec, include_mean));
		}
	} else {
		bvhar::BvharSpec mn_spec(bayes_spec);
		for (int i = 0; i < num_horizon; ++i) {
			bvhar::BvharSpec mn_spec(bayes_spec);
			mn_objs[i].reset(new bvhar::MinnBvharL(roll_mat[i], week, month, mn_spec, include_mean));
		}
	}
	std::vector<std::unique_ptr<bvhar::BvharForecaster>> forecaster(num_horizon);
	std::vector<Eigen::MatrixXd> res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int window = 0; window < num_horizon; window++) {
		bvhar::MinnFit mn_fit = mn_objs[window]->returnMinnFit();
		forecaster[window].reset(new bvhar::BvharForecaster(mn_fit, step, roll_y0[window], har_trans, month, 1, include_mean, static_cast<unsigned int>(seed_forecast[window])));
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
Eigen::MatrixXd expand_bvar(Eigen::MatrixXd y, int lag, Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test, Eigen::VectorXi seed_forecast, int nthreads) {
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
	std::vector<std::unique_ptr<bvhar::MinnBvar>> mn_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
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
		forecaster[window].reset(new bvhar::BvarForecaster(mn_fit, step, expand_y0[window], lag, 1, include_mean, static_cast<unsigned int>(seed_forecast[window])));
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
Eigen::MatrixXd expand_bvarflat(Eigen::MatrixXd y, int lag, Eigen::MatrixXd U, bool include_mean, int step, Eigen::MatrixXd y_test, Eigen::VectorXi seed_forecast, int nthreads) {
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
		forecaster[window].reset(new bvhar::BvarForecaster(mn_fit, step, expand_y0[window], lag, 1, include_mean, static_cast<unsigned int>(seed_forecast[window])));
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
Eigen::MatrixXd expand_bvhar(Eigen::MatrixXd y, int week, int month, Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test, Eigen::VectorXi seed_forecast, int nthreads) {
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
	std::vector<std::unique_ptr<bvhar::MinnBvhar>> mn_objs(num_horizon);
	if (bayes_spec.containsElementNamed("delta")) {
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
		forecaster[window].reset(new bvhar::BvharForecaster(mn_fit, step, expand_y0[window], har_trans, month, 1, include_mean, static_cast<unsigned int>(seed_forecast[window])));
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

//' Generalized Spillover of Minnesota prior
//' 
//' @param object varlse or vharlse object.
//' @param step Step to forecast.
//' @param num_iter Number to sample MNIW distribution
//' @param num_burn Number of burn-in
//' @param thin Thinning
//' @param seed Random seed for boost library
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List compute_mn_spillover(Rcpp::List object, int step, int num_iter, int num_burn, int thin, unsigned int seed) {
	if (!(object.inherits("bvarmn") || object.inherits("bvharmn"))) {
    Rcpp::stop("'object' must be bvarmn or bvharmn object.");
  }
	std::unique_ptr<bvhar::MinnSpillover> spillover;
	if (object.inherits("bvharmn")) {
		bvhar::MinnFit fit(Rcpp::as<Eigen::MatrixXd>(object["coefficients"]), Rcpp::as<Eigen::MatrixXd>(object["mn_prec"]), Rcpp::as<Eigen::MatrixXd>(object["covmat"]), object["iw_shape"]);
		spillover.reset(new bvhar::BvharSpillover(fit, step, num_iter, num_burn, thin, object["month"], Rcpp::as<Eigen::MatrixXd>(object["HARtrans"]), seed));
	} else {
		bvhar::MinnFit fit(Rcpp::as<Eigen::MatrixXd>(object["coefficients"]), Rcpp::as<Eigen::MatrixXd>(object["mn_prec"]), Rcpp::as<Eigen::MatrixXd>(object["covmat"]), object["iw_shape"]);
		spillover.reset(new bvhar::MinnSpillover(fit, step, num_iter, num_burn, thin, object["p"], seed));
	}
	spillover->updateMniw();
	spillover->computeSpillover();
	Eigen::VectorXd to_sp = spillover->returnTo();
	Eigen::VectorXd from_sp = spillover->returnFrom();
	return Rcpp::List::create(
		Rcpp::Named("connect") = spillover->returnSpillover(),
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = spillover->returnTot(),
		Rcpp::Named("net") = to_sp - from_sp,
		Rcpp::Named("net_pairwise") = spillover->returnNet()
	);
}

// Rcpp::List compute_bvarmn_spillover(int lag, int step, Eigen::MatrixXd alpha_record, Eigen::MatrixXd sig_record) {
// 	// if (!(object.inherits("bvarmn") || object.inherits("bvharmn"))) {
//   //   Rcpp::stop("'object' must be bvarmn or bvharmn object.");
//   // }
// 	bvhar::MinnRecords mn_record(alpha_record, sig_record);
// 	std::unique_ptr<bvhar::MinnSpillover> spillover(new bvhar::MinnSpillover(mn_record, step, lag));
// 	// std::unique_ptr<bvhar::MinnSpillover> spillover;
// 	// if (object.inherits("bvharmn")) {
// 	// 	bvhar::MinnFit fit(Rcpp::as<Eigen::MatrixXd>(object["coefficients"]), Rcpp::as<Eigen::MatrixXd>(object["mn_prec"]), Rcpp::as<Eigen::MatrixXd>(object["iw_scale"]), object["iw_shape"]);
// 	// 	spillover.reset(new bvhar::BvharSpillover(fit, step, num_iter, num_burn, thin, object["month"], Rcpp::as<Eigen::MatrixXd>(object["HARtrans"]), seed));
// 	// } else {
// 	// 	bvhar::MinnFit fit(Rcpp::as<Eigen::MatrixXd>(object["coefficients"]), Rcpp::as<Eigen::MatrixXd>(object["mn_prec"]), Rcpp::as<Eigen::MatrixXd>(object["iw_scale"]), object["iw_shape"]);
// 	// 	spillover.reset(new bvhar::MinnSpillover(fit, step, num_iter, num_burn, thin, object["p"], seed));
// 	// }
// 	// spillover->updateMniw();
// 	spillover->computeSpillover();
// 	Eigen::VectorXd to_sp = spillover->returnTo();
// 	Eigen::VectorXd from_sp = spillover->returnFrom();
// 	return Rcpp::List::create(
// 		Rcpp::Named("connect") = spillover->returnSpillover(),
// 		Rcpp::Named("to") = to_sp,
// 		Rcpp::Named("from") = from_sp,
// 		Rcpp::Named("tot") = spillover->returnTot(),
// 		Rcpp::Named("net") = to_sp - from_sp,
// 		Rcpp::Named("net_pairwise") = spillover->returnNet()
// 	);
// }

// Rcpp::List compute_bvharmn_spillover(int month, int step, Eigen::MatrixXd har_trans, Eigen::MatrixXd phi_record, Eigen::MatrixXd sig_record) {
// 	// if (!(object.inherits("bvarmn") || object.inherits("bvharmn"))) {
//   //   Rcpp::stop("'object' must be bvarmn or bvharmn object.");
//   // }
// 	bvhar::MinnRecords mn_record(phi_record, sig_record);
// 	std::unique_ptr<bvhar::BvharSpillover> spillover(new bvhar::BvharSpillover(mn_record, step, month, har_trans));
// 	// std::unique_ptr<bvhar::MinnSpillover> spillover;
// 	// if (object.inherits("bvharmn")) {
// 	// 	bvhar::MinnFit fit(Rcpp::as<Eigen::MatrixXd>(object["coefficients"]), Rcpp::as<Eigen::MatrixXd>(object["mn_prec"]), Rcpp::as<Eigen::MatrixXd>(object["iw_scale"]), object["iw_shape"]);
// 	// 	spillover.reset(new bvhar::BvharSpillover(fit, step, num_iter, num_burn, thin, object["month"], Rcpp::as<Eigen::MatrixXd>(object["HARtrans"]), seed));
// 	// } else {
// 	// 	bvhar::MinnFit fit(Rcpp::as<Eigen::MatrixXd>(object["coefficients"]), Rcpp::as<Eigen::MatrixXd>(object["mn_prec"]), Rcpp::as<Eigen::MatrixXd>(object["iw_scale"]), object["iw_shape"]);
// 	// 	spillover.reset(new bvhar::MinnSpillover(fit, step, num_iter, num_burn, thin, object["p"], seed));
// 	// }
// 	// spillover->updateMniw();
// 	spillover->computeSpillover();
// 	Eigen::VectorXd to_sp = spillover->returnTo();
// 	Eigen::VectorXd from_sp = spillover->returnFrom();
// 	return Rcpp::List::create(
// 		Rcpp::Named("connect") = spillover->returnSpillover(),
// 		Rcpp::Named("to") = to_sp,
// 		Rcpp::Named("from") = from_sp,
// 		Rcpp::Named("tot") = spillover->returnTot(),
// 		Rcpp::Named("net") = to_sp - from_sp,
// 		Rcpp::Named("net_pairwise") = spillover->returnNet()
// 	);
// }

//' Rolling-sample Total Spillover Index of BVAR
//' 
//' @param y Time series data of which columns indicate the variables
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param num_iter Number to sample MNIW distribution
//' @param num_burn Number of burn-in
//' @param thin Thinning
//' @param lag BVAR order
//' @param bayes_spec BVAR specification
//' @param include_mean Add constant term
//' @param seed_chain Random seed for each window
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List dynamic_bvar_spillover(Eigen::MatrixXd y, int window, int step, int num_iter, int num_burn, int thin,
																 	int lag, Rcpp::List bayes_spec, bool include_mean, Eigen::VectorXi seed_chain, int nthreads) {
// Rcpp::List dynamic_bvar_spillover(Eigen::MatrixXd y, int window, int step, int num_chains, int num_iter, int num_burn, int thin,
// 																 	int lag, Rcpp::List bayes_spec, bool include_mean, Eigen::MatrixXi seed_chain, int nthreads) {
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	// std::vector<std::vector<std::unique_ptr<bvhar::Minnesota>>> mn_objs(num_horizon);
	// for (auto &mn_chain : mn_objs) {
	// 	mn_chain.resize(num_chains);
	// 	for (auto &ptr : mn_chain) {
	// 		ptr = nullptr;
	// 	}
	// }
	std::vector<std::unique_ptr<bvhar::MinnBvar>> mn_objs(num_horizon);
	int dim = y.cols();
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		// Eigen::MatrixXd roll_y0 = bvhar::build_y0(roll_mat, lag, lag + 1);
		// Eigen::MatrixXd roll_x0 = bvhar::build_x0(roll_mat, lag, include_mean);
		bvhar::BvarSpec mn_spec(bayes_spec);
		mn_objs[i] = std::unique_ptr<bvhar::MinnBvar>(new bvhar::MinnBvar(roll_mat, lag, mn_spec, include_mean));
		// Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
		// 	lag, mn_spec._sigma, mn_spec._lambda,
		// 	mn_spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
		// 	include_mean
		// );
		// Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
		// 	Eigen::VectorXd::LinSpaced(lag, 1, lag),
		// 	mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
		// );
		// for (int j = 0; j < num_chains; ++j) {
		// 	mn_objs[i][j].reset(new bvhar::Minnesota(num_iter, roll_x0, roll_y0, x_dummy, y_dummy, static_cast<unsigned int>(seed_chain(i, j))));
		// 	mn_objs[i][j]->computePosterior();
		// }
	}
	// std::vector<std::vector<bvhar::MinnRecords>> mn_recs(num_horizon);
	// for (auto &rec_chain : mn_recs) {
	// 	rec_chain.resize(num_chains);
	// 	for (auto &ptr : rec_chain) {
	// 		ptr = nullptr;
	// 	}
	// }
	// std::vector<bvhar::MinnRecords> mn_rec(num_chains);
	std::vector<std::unique_ptr<bvhar::MinnSpillover>> spillover(num_horizon);
	// std::vector<bvhar::MinnRecords> mn_recs(num_chains);
	// std::vector<Eigen::MatrixXd> coef_record(num_chains);
	// std::vector<Eigen::MatrixXd> sig_record(num_chains);
	Eigen::VectorXd tot(num_horizon);
	Eigen::MatrixXd to_sp(num_horizon, dim);
	Eigen::MatrixXd from_sp(num_horizon, dim);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
	// for (int win = 0; win < num_horizon; ++win) {
		bvhar::MinnFit mn_fit = mn_objs[i]->returnMinnFit();
		spillover[i].reset(new bvhar::MinnSpillover(mn_fit, step, num_iter, num_burn, thin, lag, static_cast<unsigned int>(seed_chain[i])));
		spillover[i]->updateMniw();
		spillover[i]->computeSpillover();
		to_sp.row(i) = spillover[i]->returnTo();
		from_sp.row(i) = spillover[i]->returnFrom();
		tot[i] = spillover[i]->returnTot();
		mn_objs[i].reset(); // free the memory by making nullptr
		spillover[i].reset(); // free the memory by making nullptr

		// std::vector<Eigen::MatrixXd> coef_record(num_chains);
		// std::vector<Eigen::MatrixXd> sig_record(num_chains);
		// for (int chain = 0; chain < num_chains; ++chain) {
		// 	for (int i = 0; i < num_iter; ++i) {
		// 		mn_objs[win][chain]->doPosteriorDraws();
		// 	}
		// 	// mn_recs[i][j] = mn_objs[i][j]->returnMinnRecords(num_burn, thin);
		// 	bvhar::MinnRecords mn_rec = mn_objs[win][chain]->returnMinnRecords(num_burn, thin);
		// 	// mn_recs[j] = mn_objs[i][j]->returnMinnRecords(num_burn, thin);
		// 	coef_record[chain] = mn_rec.coef_record;
		// 	sig_record[chain] = mn_rec.sig_record;
		// 	mn_objs[win][chain].reset();
		// }
		// Eigen::MatrixXd coef = std::accumulate(
		// 	coef_record.begin() + 1, coef_record.end(), coef_record[0],
		// 	[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
		// 		Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
		// 		concat_mat << acc,
		// 									curr;
		// 		return concat_mat;
		// 	}
		// );
		// Eigen::MatrixXd sig = std::accumulate(
		// 	sig_record.begin() + 1, sig_record.end(), sig_record[0],
		// 	[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
		// 		Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
		// 		concat_mat << acc,
		// 									curr;
		// 		return concat_mat;
		// 	}
		// );
		// bvhar::MinnRecords mn_record(coef, sig);
		// spillover[win].reset(new bvhar::MinnSpillover(mn_record, step, lag));
		// // bvhar::MinnFit mn_fit = mn_objs[win]->returnMinnFit();
		// // spillover[win].reset(new bvhar::MinnSpillover(mn_fit, step, num_iter, num_burn, thin, lag, static_cast<unsigned int>(seed_chain[win])));
		// // spillover[win]->updateMniw();
		// spillover[win]->computeSpillover();
		// to_sp.row(win) = spillover[win]->returnTo();
		// from_sp.row(win) = spillover[win]->returnFrom();
		// tot[win] = spillover[win]->returnTot();
		// // mn_objs[win].reset(); // free the memory by making nullptr
		// spillover[win].reset(); // free the memory by making nullptr
	}
	return Rcpp::List::create(
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = tot,
		Rcpp::Named("net") = to_sp - from_sp
	);
}

//' Rolling-sample Total Spillover Index of BVHAR
//' 
//' @param y Time series data of which columns indicate the variables
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param num_iter Number to sample MNIW distribution
//' @param num_burn Number of burn-in
//' @param thin Thinning
//' @param week Week order
//' @param month Month order
//' @param bayes_spec BVHAR specification
//' @param include_mean Add constant term
//' @param seed_chain Random seed for each window
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List dynamic_bvhar_spillover(Eigen::MatrixXd y, int window, int step, int num_iter, int num_burn, int thin,
																	 int week, int month, Rcpp::List bayes_spec, bool include_mean, Eigen::VectorXi seed_chain, int nthreads) {
// Rcpp::List dynamic_bvhar_spillover(Eigen::MatrixXd y, int window, int step, int num_chains, int num_iter, int num_burn, int thin,
// 																	 int week, int month, Rcpp::List bayes_spec, bool include_mean, Eigen::MatrixXi seed_chain, int nthreads) {
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	int dim = y.cols();
	Eigen::MatrixXd har_trans = bvhar::build_vhar(dim, week, month, include_mean);
	// std::vector<std::vector<std::unique_ptr<bvhar::Minnesota>>> mn_objs(num_horizon);
	// for (auto &mn_chain : mn_objs) {
	// 	mn_chain.resize(num_chains);
	// 	for (auto &ptr : mn_chain) {
	// 		ptr = nullptr;
	// 	}
	// }
	// for (int i = 0; i < num_horizon; ++i) {
	// 	Eigen::MatrixXd roll_mat = y.middleRows(i, window);
	// 	Eigen::MatrixXd roll_y0 = bvhar::build_y0(roll_mat, month, month + 1);
	// 	Eigen::MatrixXd roll_x0 = bvhar::build_x0(roll_mat, month, include_mean) * har_trans.transpose();
		// bvhar::BvarSpec mn_spec(bayes_spec);
		// Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
		// 	lag, mn_spec._sigma, mn_spec._lambda,
		// 	mn_spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
		// 	include_mean
		// );
		// Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
		// 	Eigen::VectorXd::LinSpaced(lag, 1, lag),
		// 	mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
		// );
	// 	if (bayes_spec.containsElementNamed("delta")) {
	// 		// bvhar::BvarSpec bvhar_spec(bayes_spec);
	// 		bvhar::BvarSpec mn_spec(bayes_spec);
	// 		Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
	// 			3, mn_spec._sigma, mn_spec._lambda,
	// 			mn_spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
	// 			include_mean
	// 		);
	// 		Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
	// 			Eigen::VectorXd::LinSpaced(3, 1, 3),
	// 			mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
	// 		);
	// 		// mn_objs[i] = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharS(roll_mat, week, month, bvhar_spec, include_mean));
	// 		for (int j = 0; j < num_chains; ++j) {
	// 			mn_objs[i][j].reset(new bvhar::Minnesota(num_iter, roll_x0, roll_y0, x_dummy, y_dummy, static_cast<unsigned int>(seed_chain(i, j))));
	// 			mn_objs[i][j]->computePosterior();
	// 		}
	// 	} else {
	// 		// bvhar::BvharSpec bvhar_spec(bayes_spec);
	// 		bvhar::BvharSpec mn_spec(bayes_spec);
	// 		Eigen::MatrixXd y_dummy = bvhar::build_ydummy(
	// 			3, mn_spec._sigma, mn_spec._lambda,
	// 			mn_spec._daily, mn_spec._weekly, mn_spec._monthly,
	// 			include_mean
	// 		);
	// 		Eigen::MatrixXd x_dummy = bvhar::build_xdummy(
	// 			Eigen::VectorXd::LinSpaced(3, 1, 3),
	// 			mn_spec._lambda, mn_spec._sigma, mn_spec._eps, include_mean
	// 		);
	// 		// mn_objs[i] = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharL(roll_mat, week, month, bvhar_spec, include_mean));
	// 		for (int j = 0; j < num_chains; ++j) {
	// 			mn_objs[i][j].reset(new bvhar::Minnesota(num_iter, roll_x0, roll_y0, x_dummy, y_dummy, static_cast<unsigned int>(seed_chain(i, j))));
	// 			mn_objs[i][j]->computePosterior();
	// 		}
	// 	}
	// }
	std::vector<std::unique_ptr<bvhar::MinnBvhar>> mn_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		if (bayes_spec.containsElementNamed("delta")) {
			bvhar::BvarSpec bvhar_spec(bayes_spec);
			mn_objs[i] = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharS(roll_mat, week, month, bvhar_spec, include_mean));
		} else {
			bvhar::BvharSpec bvhar_spec(bayes_spec);
			mn_objs[i] = std::unique_ptr<bvhar::MinnBvhar>(new bvhar::MinnBvharL(roll_mat, week, month, bvhar_spec, include_mean));
		}
	}
	std::vector<std::unique_ptr<bvhar::BvharSpillover>> spillover(num_horizon);
	// std::vector<bvhar::MinnRecords> mn_recs(num_chains);
	// std::vector<Eigen::MatrixXd> coef_record(num_chains);
	// std::vector<Eigen::MatrixXd> sig_record(num_chains);
  Eigen::VectorXd tot(num_horizon);
	Eigen::MatrixXd to_sp(num_horizon, y.cols());
	Eigen::MatrixXd from_sp(num_horizon, y.cols());
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
	// for (int win = 0; win < num_horizon; ++win) {
		// std::vector<Eigen::MatrixXd> coef_record(num_chains);
		// std::vector<Eigen::MatrixXd> sig_record(num_chains);
		// for (int chain = 0; chain < num_chains; ++chain) {
		// 	for (int i = 0; i < num_iter; ++i) {
		// 		mn_objs[win][chain]->doPosteriorDraws();
		// 	}
		// 	bvhar::MinnRecords mn_rec = mn_objs[win][chain]->returnMinnRecords(num_burn, thin);
		// 	coef_record[chain] = mn_rec.coef_record;
		// 	sig_record[chain] = mn_rec.sig_record;
		// 	mn_objs[win][chain].reset();
		// }
		// Eigen::MatrixXd coef = std::accumulate(
		// 	coef_record.begin() + 1, coef_record.end(), coef_record[0],
		// 	[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
		// 		Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
		// 		concat_mat << acc,
		// 									curr;
		// 		return concat_mat;
		// 	}
		// );
		// Eigen::MatrixXd sig = std::accumulate(
		// 	sig_record.begin() + 1, sig_record.end(), sig_record[0],
		// 	[](const Eigen::MatrixXd& acc, const Eigen::MatrixXd& curr) {
		// 		Eigen::MatrixXd concat_mat(acc.rows() + curr.rows(), acc.cols());
		// 		concat_mat << acc,
		// 									curr;
		// 		return concat_mat;
		// 	}
		// );
		// bvhar::MinnRecords mn_record(coef, sig);
		bvhar::MinnFit mn_fit = mn_objs[i]->returnMinnFit();
		// spillover[i].reset(new bvhar::BvharSpillover(mn_fit, step, num_iter, num_burn, thin, month, har_trans, static_cast<unsigned int>(seed_chain[win])));
		spillover[i].reset(new bvhar::BvharSpillover(mn_fit, step, num_iter, num_burn, thin, month, har_trans, static_cast<unsigned int>(seed_chain[i])));
		spillover[i]->updateMniw();
		// spillover[win].reset(new bvhar::BvharSpillover(mn_record, step, month, har_trans));
		spillover[i]->computeSpillover();
		to_sp.row(i) = spillover[i]->returnTo();
		from_sp.row(i) = spillover[i]->returnFrom();
		tot[i] = spillover[i]->returnTot();
		mn_objs[i].reset(); // free the memory by making nullptr
		spillover[i].reset(); // free the memory by making nullptr
	}
	return Rcpp::List::create(
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = tot,
		Rcpp::Named("net") = to_sp - from_sp
	);
}
