#include <bvhar/ols>

//' Compute VAR(p) Coefficient Matrices and Fitted Values
//' 
//' This function fits VAR(p) given response and design matrices of multivariate time series.
//' 
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param method Method to solve linear equation system. 1: normal equation, 2: cholesky, 3: HouseholderQR.
//' @details
//' Given Y0 and Y0, the function estimate least squares
//' Y0 = X0 A + Z
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_var(Eigen::MatrixXd y, int lag, bool include_mean, int method) {
	// std::unique_ptr<baecon::bvhar::OlsVar> ols_obj(new bvhar::OlsVar(y, lag, include_mean, method));
	auto ols_obj = std::make_unique<baecon::bvhar::OlsVar>(y, lag, include_mean, method);
	return ols_obj->returnOlsRes();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_varx(Eigen::MatrixXd y, Eigen::MatrixXd exogen, int lag, int exogen_lag, bool include_mean, int method) {
	auto ols_obj = std::make_unique<baecon::bvhar::OlsVar>(y, exogen, lag, exogen_lag, include_mean, method);
	return ols_obj->returnOlsRes();
}

//' Compute Vector HAR Coefficient Matrices and Fitted Values
//' 
//' This function fits VHAR given response and design matrices of multivariate time series.
//' 
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param week Integer, order for weekly term
//' @param month Integer, order for monthly term
//' @param include_mean bool, Add constant term (Default: `true`) or not (`false`)
//' @param method Method to solve linear equation system. 1: normal equation, 2: cholesky, 3: HouseholderQR.
//' @details
//' Given Y0 and Y0, the function estimate least squares
//' \deqn{Y_0 = X_1 \Phi + Z}
//' 
//' @references
//' Baek, C. and Park, M. (2021). *Sparse vector heterogeneous autoregressive modeling for realized volatility*. J. Korean Stat. Soc. 50, 495-510. doi:[10.1007/s42952-020-00090-5](https://doi.org/10.1007/s42952-020-00090-5)
//' 
//' Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174-196. doi:[10.1093/jjfinec/nbp001](https://doi.org/10.1093/jjfinec/nbp001)
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_har(Eigen::MatrixXd y, int week, int month, bool include_mean, int method) {
	auto ols_obj = std::make_unique<baecon::bvhar::OlsVhar>(y, week, month, include_mean, method);
	return ols_obj->returnOlsRes();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_harx(Eigen::MatrixXd y, Eigen::MatrixXd exogen, int week, int month, int exogen_lag, bool include_mean, int method) {
	auto ols_obj = std::make_unique<baecon::bvhar::OlsVhar>(y, exogen, week, month, exogen_lag, include_mean, method);
	return ols_obj->returnOlsRes();
}

//' Covariance Estimate for Residual Covariance Matrix
//' 
//' Compute ubiased estimator for residual covariance.
//' 
//' @param z Matrix, residual
//' @param num_design Integer, Number of sample used (s = n - p)
//' @param dim_design Ingeger, Number of parameter for each dimension (k = mp + 1)
//' @details
//' See pp75 Lütkepohl (2007).
//' 
//' * s = n - p: sample used (`num_design`)
//' * k = mp + 1 (m: dimension, p: VAR lag): number of parameter for each dimension (`dim_design`)
//' 
//' Then an unbiased estimator for \eqn{\Sigma_e} is
//' 
//' \deqn{\hat{\Sigma}_e = \frac{1}{s - k} (Y_0 - \hat{A} X_0)^T (Y_0 - \hat{A} X_0)}
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_cov(Eigen::MatrixXd z, int num_design, int dim_design) {
  Eigen::MatrixXd cov_mat(z.cols(), z.cols());
  cov_mat = z.transpose() * z / (num_design - dim_design);
  return cov_mat;
}

//' Statistic for VAR
//' 
//' Compute partial t-statistics for inference in VAR model.
//' 
//' @param object A `varlse` object
//' @details
//' Partial t-statistic for H0: aij = 0
//' 
//' * For each variable (e.g. 1st variable)
//' * Standard error =  (1st) diagonal element of \eqn{\Sigma_e} estimator x diagonal elements of \eqn{(X_0^T X_0)^(-1)}
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Rcpp::List infer_var(Rcpp::List object) {
  if (!object.inherits("varlse")) {
    Rcpp::stop("'object' must be varlse object.");
  }
  int dim = object["m"]; // dimension of time series
  Eigen::MatrixXd cov_mat = object["covmat"]; // sigma
  Eigen::MatrixXd coef_mat = object["coefficients"]; // Ahat(mp, m) = [A1^T, A2^T, ..., Ap^T, c^T]^T
  Eigen::MatrixXd design_mat = object["design"]; // X0: n x mp
  int num_design = object["obs"];
  int dim_design = coef_mat.rows(); // mp(+1)
  int df = num_design - dim_design;
  Eigen::VectorXd XtX = (design_mat.transpose() * design_mat).inverse().diagonal(); // diagonal element of (XtX)^(-1)
  Eigen::MatrixXd res(dim_design * dim, 3); // stack estimate, std, and t stat
  Eigen::ArrayXd st_err(dim_design); // save standard error in for loop
  for (int i = 0; i < dim; i++) {
    res.block(i * dim_design, 0, dim_design, 1) = coef_mat.col(i);
    for (int j = 0; j < dim_design; j++) {
      st_err[j] = sqrt(XtX[j] * cov_mat(i, i)); // variable-covariance matrix element
    }
    res.block(i * dim_design, 1, dim_design, 1) = st_err;
    res.block(i * dim_design, 2, dim_design, 1) = coef_mat.col(i).array() / st_err;
  }
  return Rcpp::List::create(
    Rcpp::Named("df") = df,
    Rcpp::Named("summary_stat") = res
  );
}

//' Statistic for VHAR
//' 
//' Compute partial t-statistics for inference in VHAR model.
//' 
//' @param object A `vharlse` object
//' @details
//' Partial t-statistic for H0: \eqn{\phi_{ij} = 0}
//' 
//' * For each variable (e.g. 1st variable)
//' * Standard error =  (1st) diagonal element of \eqn{\Sigma_e} estimator x diagonal elements of \eqn{(X_1^T X_1)^(-1)}
//' @noRd
// [[Rcpp::export]]
Rcpp::List infer_vhar(Rcpp::List object) {
  if (!object.inherits("vharlse")) {
    Rcpp::stop("'object' must be vharlse object.");
  }
  int dim = object["m"]; // dimension of time series
  Eigen::MatrixXd cov_mat = object["covmat"]; // sigma
  Eigen::MatrixXd coef_mat = object["coefficients"]; // Phihat(mp, m) = [Phi(daily), Phi(weekly), Phi(monthly), c^T]^T
  Eigen::MatrixXd design_mat = object["design"]; // X0: n x mp
  Eigen::MatrixXd HARtrans = object["HARtrans"]; // HAR transformation
  Eigen::MatrixXd vhar_design = design_mat * HARtrans.transpose(); // X1 = X0 * C0^T
  int num_design = object["obs"];
  int num_har = coef_mat.rows(); // 3m(+1)
  int df = num_design - num_har;
  Eigen::VectorXd XtX = (vhar_design.transpose() * vhar_design).inverse().diagonal(); // diagonal element of (XtX)^(-1)
  Eigen::MatrixXd res(num_har * dim, 3); // stack estimate, std, and t stat
  Eigen::ArrayXd st_err(num_har); // save standard error in for loop
  for (int i = 0; i < dim; i++) {
    res.block(i * num_har, 0, num_har, 1) = coef_mat.col(i);
    for (int j = 0; j < num_har; j++) {
      st_err[j] = sqrt(XtX[j] * cov_mat(i, i)); // variable-covariance matrix element
    }
    res.block(i * num_har, 1, num_har, 1) = st_err;
    res.block(i * num_har, 2, num_har, 1) = coef_mat.col(i).array() / st_err;
  }
  return Rcpp::List::create(
    Rcpp::Named("df") = df,
    Rcpp::Named("summary_stat") = res
  );
}

//' Generate Multivariate Time Series Process Following VAR(p)
//' 
//' This function generates multivariate time series dataset that follows VAR(p).
//' 
//' @param num_sim Number to generated process
//' @param num_burn Number of burn-in
//' @param var_coef VAR coefficient. The format should be the same as the output of [coef()] from [var_lm()]
//' @param var_lag Lag of VAR
//' @param sig_error Variance matrix of the error term. Try `diag(dim)`.
//' @param init Initial y1, ..., yp matrix to simulate VAR model. Try `matrix(0L, nrow = var_lag, ncol = dim)`.
//' @param process Process type. 1: Gaussian. 2: student-t.
//' @param mvt_df DF of MVT
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_var_process(int num_sim, int num_burn,
                                Eigen::MatrixXd var_coef,
                                int var_lag,
                                Eigen::MatrixXd sig_error,
															  double mvt_df,
                                Eigen::MatrixXd init,
                                int process,
                                int method,
															  unsigned int seed) {
	// auto var_generator = std::make_unique<baecon::bvhar::OlsSimulator>(num_sim, num_burn, var_lag, init, var_coef, sig_error, method, seed);
	auto dgp_run = [&]() -> std::unique_ptr<baecon::bvhar::OlsSimulator> {
		if (process == 1) {
			return std::make_unique<baecon::bvhar::OlsSimulator>(num_sim, num_burn, var_lag, init, var_coef, sig_error, method, seed);
		}
		return std::make_unique<baecon::bvhar::OlsSimulator>(num_sim, num_burn, var_lag, init, var_coef, sig_error, method, seed, mvt_df);
	}();
	return dgp_run->returnDgp();
}

//' Generate Multivariate Time Series Process Following VHAR
//' 
//' This function generates multivariate time series dataset that follows VHAR.
//' 
//' @param num_sim Number to generated process
//' @param num_burn Number of burn-in
//' @param vhar_coef VHAR coefficient. The format should be the same as the output of [coef.vharlse()] from [vhar_lm()]
//' @param week Order for weekly term. Try `5L` by default.
//' @param month Order for monthly term. Try `22L` by default.
//' @param sig_error Variance matrix of the error term. Try `diag(dim)`.
//' @param init Initial y1, ..., y_month matrix to simulate VHAR model. Try `matrix(0L, nrow = month, ncol = dim)`.
//' @param process Process type. 1: Gaussian. 2: student-t.
//' @param mvt_df DF of MVT
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_vhar_process(int num_sim, int num_burn,
                                 Eigen::MatrixXd vhar_coef, 
                                 int week, int month,
                                 Eigen::MatrixXd sig_error,
															   double mvt_df,
                                 Eigen::MatrixXd init,
                                 int process,
                                 int method,
															   unsigned int seed) {
	auto dgp_run = [&]() -> std::unique_ptr<baecon::bvhar::OlsSimulator> {
		if (process == 1) {
			return std::make_unique<baecon::bvhar::OlsSimulator>(num_sim, num_burn, week, month, init, vhar_coef, sig_error, method, seed);
		}
		return std::make_unique<baecon::bvhar::OlsSimulator>(num_sim, num_burn, week, month, init, vhar_coef, sig_error, method, seed, mvt_df);
	}();
	return dgp_run->returnDgp();
}

//' Forecasting Vector Autoregression
//' 
//' @param object A `varlse` object
//' @param step Integer, Step to forecast
//' @details
//' n-step ahead forecasting using VAR(p) recursively, based on pp35 of Lütkepohl (2007).
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd forecast_var(Rcpp::List object, int step) {
  if (! object.inherits("varlse")) {
		Rcpp::stop("'object' must be varlse object.");
	}
  Eigen::MatrixXd response_mat = object["y"]; // Y0
  Eigen::MatrixXd coef_mat = object["coefficients"]; // bhat
  int var_lag = object["p"]; // VAR(p)
	bool include_mean = Rcpp::as<std::string>(object["type"]) == "const";
	// bvhar::OlsFit ols_fit(coef_mat, var_lag);
	// std::unique_ptr<baecon::bvhar::VarForecaster> forecaster(new bvhar::VarForecaster(ols_fit, step, response_mat, include_mean));
	// return forecaster->forecastPoint();
	auto forecaster = std::make_unique<baecon::bvhar::OlsForecastRun>(var_lag, step, response_mat, coef_mat, include_mean);
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd forecast_varx(Eigen::MatrixXd response, Eigen::MatrixXd coef_mat, int lag, int step,
															bool include_mean, Eigen::MatrixXd exogen, Eigen::MatrixXd exogen_coef, int exogen_lag) {
	auto forecaster = std::make_unique<baecon::bvhar::OlsForecastRun>(lag, step, response, coef_mat, include_mean, exogen_lag, exogen, exogen_coef);
	// auto forecaster = std::make_unique<baecon::bvhar::OlsForecastRun>(lag, step, response, coef_mat, include_mean, exogen, exogen_lag);
	return forecaster->returnForecast();
}

//' Forecasting Vector HAR
//' 
//' @param object A `vharlse` object
//' @param step Integer, Step to forecast
//' @details
//' n-step ahead forecasting using VHAR recursively.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd forecast_vhar(Rcpp::List object, int step) {
  if (!object.inherits("vharlse")) {
    Rcpp::stop("'object' must be vharlse object.");
  }
  Eigen::MatrixXd response_mat = object["y"]; // Y0
  Eigen::MatrixXd coef_mat = object["coefficients"]; // bhat
  // Eigen::MatrixXd HARtrans = object["HARtrans"]; // HAR transformation
	int week = object["week"];
  int month = object["month"];
	bool include_mean = Rcpp::as<std::string>(object["type"]) == "const";
	// bvhar::OlsFit ols_fit(coef_mat, month);
	// std::unique_ptr<baecon::bvhar::VharForecaster> forecaster(new bvhar::VharForecaster(ols_fit, step, response_mat, HARtrans, include_mean));
	// return forecaster->forecastPoint();
	auto forecaster = std::make_unique<baecon::bvhar::OlsForecastRun>(week, month, step, response_mat, coef_mat, include_mean);
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd forecast_harx(Eigen::MatrixXd response, Eigen::MatrixXd coef_mat, int week, int month, int step,
															bool include_mean, Eigen::MatrixXd exogen, Eigen::MatrixXd exogen_coef, int exogen_lag) {
	auto forecaster = std::make_unique<baecon::bvhar::OlsForecastRun>(week, month, step, response, coef_mat, include_mean, exogen_lag, exogen, exogen_coef);
	return forecaster->returnForecast();
}

//' Out-of-Sample Forecasting of VAR based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of VAR.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag VAR order
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param method Method to solve linear equation system. 1: normal equation, 2: cholesky, 3: HouseholderQR.
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_var(Eigen::MatrixXd y, int lag, bool include_mean, int step, Eigen::MatrixXd y_test, int method, int nthreads) {
	auto forecaster = std::make_unique<baecon::bvhar::VarOutforecastRun<baecon::bvhar::OlsRollforecastRun>>(
		y, lag, include_mean, step, y_test,
		method, nthreads
	);
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_varx(Eigen::MatrixXd y, int lag, bool include_mean,
												  int step, Eigen::MatrixXd y_test, int method, int nthreads,
												  Eigen::MatrixXd exogen, int exogen_lag) {
	auto forecaster = std::make_unique<baecon::bvhar::VarOutforecastRun<baecon::bvhar::OlsRollforecastRun>>(
		y, lag, include_mean, step, y_test,
		method, nthreads,
		exogen, exogen_lag
	);
	return forecaster->returnForecast();
}

//' Out-of-Sample Forecasting of VHAR based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of VHAR.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param week Integer, order for weekly term
//' @param month Integer, order for monthly term
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param method Method to solve linear equation system. 1: normal equation, 2: cholesky, 3: HouseholderQR.
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_vhar(Eigen::MatrixXd y, int week, int month, bool include_mean, int step, Eigen::MatrixXd y_test, int method, int nthreads) {
	auto forecaster = std::make_unique<baecon::bvhar::VharOutforecastRun<baecon::bvhar::OlsRollforecastRun>>(
		y, week, month, include_mean, step, y_test,
		method, nthreads
	);
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_vharx(Eigen::MatrixXd y, int week, int month, bool include_mean,
													 int step, Eigen::MatrixXd y_test, int method, int nthreads,
													 Eigen::MatrixXd exogen, int exogen_lag) {
	auto forecaster = std::make_unique<baecon::bvhar::VharOutforecastRun<baecon::bvhar::OlsRollforecastRun>>(
		y, week, month, include_mean, step, y_test,
		method, nthreads,
		exogen, exogen_lag
	);
	return forecaster->returnForecast();
}

//' Out-of-Sample Forecasting of VAR based on Expanding Window
//' 
//' This function conducts an expanding window forecasting of VAR.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag VAR order
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param method Method to solve linear equation system. 1: normal equation, 2: cholesky, 3: HouseholderQR.
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd expand_var(Eigen::MatrixXd y, int lag, bool include_mean, int step, Eigen::MatrixXd y_test, int method, int nthreads) {
	auto forecaster = std::make_unique<baecon::bvhar::VarOutforecastRun<baecon::bvhar::OlsExpandforecastRun>>(
		y, lag, include_mean, step, y_test,
		method, nthreads
	);
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd expand_varx(Eigen::MatrixXd y, int lag, bool include_mean,
														int step, Eigen::MatrixXd y_test, int method, int nthreads,
														Eigen::MatrixXd exogen, int exogen_lag) {
	auto forecaster = std::make_unique<baecon::bvhar::VarOutforecastRun<baecon::bvhar::OlsExpandforecastRun>>(
		y, lag, include_mean, step, y_test,
		method, nthreads,
		exogen, exogen_lag
	);
	return forecaster->returnForecast();
}

//' Out-of-Sample Forecasting of VHAR based on Expanding Window
//' 
//' This function conducts an expanding window forecasting of VHAR.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param week Integer, order for weekly term
//' @param month Integer, order for monthly term
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param method Method to solve linear equation system. 1: normal equation, 2: cholesky, 3: HouseholderQR.
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd expand_vhar(Eigen::MatrixXd y, int week, int month, bool include_mean, int step, Eigen::MatrixXd y_test, int method, int nthreads) {
	auto forecaster = std::make_unique<baecon::bvhar::VharOutforecastRun<baecon::bvhar::OlsExpandforecastRun>>(
		y, week, month, include_mean, step, y_test,
		method, nthreads
	);
	return forecaster->returnForecast();
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd expand_vharx(Eigen::MatrixXd y, int week, int month, bool include_mean,
														 int step, Eigen::MatrixXd y_test, int method, int nthreads,
														 Eigen::MatrixXd exogen, int exogen_lag) {
	auto forecaster = std::make_unique<baecon::bvhar::VharOutforecastRun<baecon::bvhar::OlsExpandforecastRun>>(
		y, week, month, include_mean, step, y_test,
		method, nthreads,
		exogen, exogen_lag
	);
	return forecaster->returnForecast();
}

//' Generalized Spillover of VAR
//' 
//' @param object varlse or vharlse object.
//' @param step Step to forecast.
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List compute_var_spillover(Eigen::MatrixXd coef_mat, int lag, Eigen::MatrixXd cov_mat, int step) {
	auto spillover = std::make_unique<baecon::bvhar::OlsSpilloverRun>(lag, step, coef_mat, cov_mat);
	return spillover->returnSpillover();
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List compute_vhar_spillover(Eigen::MatrixXd coef_mat, int week, int month, Eigen::MatrixXd cov_mat, int step) {
	auto spillover = std::make_unique<baecon::bvhar::OlsSpilloverRun>(week, month, step, coef_mat, cov_mat);
	return spillover->returnSpillover();
}

//' Rolling-sample Total Spillover Index of VAR
//' 
//' @param y Time series data of which columns indicate the variables
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param lag VAR order
//' @param include_mean Add constant term
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List dynamic_var_spillover(Eigen::MatrixXd y, int window, int step, int lag, bool include_mean, int method, int nthreads) {
	auto spillover = std::make_unique<baecon::bvhar::OlsDynamicSpillover>(y, window, step, lag, include_mean, method, nthreads);
	return spillover->returnSpillover();
}

//' Rolling-sample Total Spillover Index of VHAR
//' 
//' @param y Time series data of which columns indicate the variables
//' @param window Rolling window size
//' @param step forecast horizon for FEVD
//' @param har VHAR order
//' @param include_mean Add constant term
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List dynamic_vhar_spillover(Eigen::MatrixXd y, int window, int step, int week, int month, bool include_mean, int method, int nthreads) {
	auto spillover = std::make_unique<baecon::bvhar::OlsDynamicSpillover>(y, window, step, month, include_mean, method, nthreads, week);
	return spillover->returnSpillover();
}
