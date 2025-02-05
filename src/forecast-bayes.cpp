#include <bvhar/forecast>

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
											 	 Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											 	 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
											 	 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	 bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<bvhar::McmcOutforecastRun<bvhar::RegForecaster>> {
		if (ggl) {
			return std::make_unique<bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster, true>>(
				y, lag, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, seed_chain, seed_forecast, display_progress, nthreads, true
			);
		}
		return std::make_unique<bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster, false>>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, seed_chain, seed_forecast, display_progress, nthreads, true
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
													bool sparse, double level, Rcpp::List fit_record,
											  	Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											  	Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
													bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  	bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<bvhar::McmcOutforecastRun<bvhar::RegForecaster>> {
		if (ggl) {
			return std::make_unique<bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster, true>>(
				y, week, month, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, seed_chain, seed_forecast, display_progress, nthreads, true
			);
		}
		return std::make_unique<bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster, false>>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, seed_chain, seed_forecast, display_progress, nthreads, true
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
												 	 bool sparse, double level, Rcpp::List fit_record,
											 	 	 Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											 	 	 Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
												 	 bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											 	 	 bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<bvhar::McmcOutforecastRun<bvhar::RegForecaster>> {
		if (ggl) {
			return std::make_unique<bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster, true>>(
				y, lag, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, seed_chain, seed_forecast, display_progress, nthreads, true
			);
		}
		return std::make_unique<bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster, false>>(
			y, lag, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, seed_chain, seed_forecast, display_progress, nthreads, true
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
														bool sparse, double level, Rcpp::List fit_record,
											  		Rcpp::List param_reg, Rcpp::List param_prior, Rcpp::List param_intercept, Rcpp::List param_init, int prior_type, bool ggl,
											  		Eigen::VectorXi grp_id, Eigen::VectorXi own_id, Eigen::VectorXi cross_id, Eigen::MatrixXi grp_mat,
														bool include_mean, bool stable, int step, Eigen::MatrixXd y_test,
											  		bool get_lpl, Eigen::MatrixXi seed_chain, Eigen::VectorXi seed_forecast, bool display_progress, int nthreads) {
	auto forecaster = [&]() -> std::unique_ptr<bvhar::McmcOutforecastRun<bvhar::RegForecaster>> {
		if (ggl) {
			return std::make_unique<bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster, true>>(
				y, week, month, num_chains, num_iter, num_burn, thinning,
				sparse, level, fit_record,
				param_reg, param_prior, param_intercept, param_init, prior_type,
				grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
				get_lpl, seed_chain, seed_forecast, display_progress, nthreads, true
			);
		}
		return std::make_unique<bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster, false>>(
			y, week, month, num_chains, num_iter, num_burn, thinning,
			sparse, level, fit_record,
			param_reg, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat, include_mean, stable, step, y_test,
			get_lpl, seed_chain, seed_forecast, display_progress, nthreads, true
		);
	}();
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
