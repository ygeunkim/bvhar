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
	std::unique_ptr<bvhar::OlsVar> ols_obj(new bvhar::OlsVar(y, lag, include_mean, method));
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
	std::unique_ptr<bvhar::OlsVhar> ols_obj(new bvhar::OlsVhar(y, week, month, include_mean, method));
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
  Eigen::MatrixXd response_mat = object["y0"]; // Y0
  Eigen::MatrixXd coef_mat = object["coefficients"]; // bhat
  int var_lag = object["p"]; // VAR(p)
	bool include_mean = Rcpp::as<std::string>(object["type"]) == "const";
	bvhar::OlsFit ols_fit(coef_mat, var_lag);
	std::unique_ptr<bvhar::VarForecaster> forecaster(new bvhar::VarForecaster(ols_fit, step, response_mat, include_mean));
	return forecaster->forecastPoint();
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
  Eigen::MatrixXd response_mat = object["y0"]; // Y0
  Eigen::MatrixXd coef_mat = object["coefficients"]; // bhat
  Eigen::MatrixXd HARtrans = object["HARtrans"]; // HAR transformation
  int month = object["month"];
	bool include_mean = Rcpp::as<std::string>(object["type"]) == "const";
	bvhar::OlsFit ols_fit(coef_mat, month);
	std::unique_ptr<bvhar::VharForecaster> forecaster(new bvhar::VharForecaster(ols_fit, step, response_mat, HARtrans, include_mean));
	return forecaster->forecastPoint();
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
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
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
	std::vector<std::unique_ptr<bvhar::MultiOls>> ols_objs(num_horizon);
	switch(method) {
	case 1: {
		for (int i = 0; i < num_horizon; ++i) {
			Eigen::MatrixXd design = bvhar::build_x0(roll_mat[i], lag, include_mean);
			ols_objs[i] = std::unique_ptr<bvhar::MultiOls>(new bvhar::MultiOls(design, roll_y0[i]));
		}
	}
	case 2: {
		for (int i = 0; i < num_horizon; ++i) {
			Eigen::MatrixXd design = bvhar::build_x0(roll_mat[i], lag, include_mean);
			ols_objs[i] = std::unique_ptr<bvhar::MultiOls>(new bvhar::LltOls(design, roll_y0[i]));
		}
	}
	case 3: {
		for (int i = 0; i < num_horizon; ++i) {
			Eigen::MatrixXd design = bvhar::build_x0(roll_mat[i], lag, include_mean);
			ols_objs[i] = std::unique_ptr<bvhar::MultiOls>(new bvhar::QrOls(design, roll_y0[i]));
		}
	}
	}
	std::vector<std::unique_ptr<bvhar::VarForecaster>> forecaster(num_horizon);
	std::vector<Eigen::MatrixXd> res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int window = 0; window < num_horizon; window++) {
		bvhar::OlsFit ols_fit = ols_objs[window]->returnOlsFit(lag);
		forecaster[window].reset(new bvhar::VarForecaster(ols_fit, step, roll_y0[window], include_mean));
		res[window] = forecaster[window]->forecastPoint().bottomRows(1);
		ols_objs[window].reset(); // free the memory by making nullptr
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
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
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
	std::vector<std::unique_ptr<bvhar::MultiOls>> ols_objs(num_horizon);
	switch(method) {
	case 1: {
		for (int i = 0; i < num_horizon; ++i) {
			Eigen::MatrixXd design = bvhar::build_x0(roll_mat[i], month, include_mean) * har_trans.transpose();
			ols_objs[i] = std::unique_ptr<bvhar::MultiOls>(new bvhar::MultiOls(design, roll_y0[i]));
		}
	}
	case 2: {
		for (int i = 0; i < num_horizon; ++i) {
			Eigen::MatrixXd design = bvhar::build_x0(roll_mat[i], month, include_mean) * har_trans.transpose();
			ols_objs[i] = std::unique_ptr<bvhar::MultiOls>(new bvhar::LltOls(design, roll_y0[i]));
		}
	}
	case 3: {
		for (int i = 0; i < num_horizon; ++i) {
			Eigen::MatrixXd design = bvhar::build_x0(roll_mat[i], month, include_mean) * har_trans.transpose();
			ols_objs[i] = std::unique_ptr<bvhar::MultiOls>(new bvhar::QrOls(design, roll_y0[i]));
		}
	}
	}
	std::vector<std::unique_ptr<bvhar::VharForecaster>> forecaster(num_horizon);
	std::vector<Eigen::MatrixXd> res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int window = 0; window < num_horizon; window++) {
		bvhar::OlsFit ols_fit = ols_objs[window]->returnOlsFit(month);
		forecaster[window].reset(new bvhar::VharForecaster(ols_fit, step, roll_y0[window], har_trans, include_mean));
		res[window] = forecaster[window]->forecastPoint().bottomRows(1);
		ols_objs[window].reset(); // free the memory by making nullptr
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
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1; // longest forecast horizon
	Eigen::MatrixXd tot_mat(num_window + num_test, dim);
	tot_mat << y,
						y_test;
	std::vector<Eigen::MatrixXd> expand_mat(num_horizon);
	std::vector<Eigen::MatrixXd> expand_y0(num_horizon);
	for (int i = 0; i < num_horizon; i++) {
		expand_mat[i] = tot_mat.topRows(num_window + i);
		expand_y0[i] = bvhar::build_y0(expand_mat[i], lag, lag + 1);
	}
	std::vector<std::unique_ptr<bvhar::MultiOls>> ols_objs(num_horizon);
	switch(method) {
	case 1: {
		for (int i = 0; i < num_horizon; ++i) {
			Eigen::MatrixXd design = bvhar::build_x0(expand_mat[i], lag, include_mean);
			ols_objs[i] = std::unique_ptr<bvhar::MultiOls>(new bvhar::MultiOls(design, expand_y0[i]));
		}
	}
	case 2: {
		for (int i = 0; i < num_horizon; ++i) {
			Eigen::MatrixXd design = bvhar::build_x0(expand_mat[i], lag, include_mean);
			ols_objs[i] = std::unique_ptr<bvhar::MultiOls>(new bvhar::LltOls(design, expand_y0[i]));
		}
	}
	case 3: {
		for (int i = 0; i < num_horizon; ++i) {
			Eigen::MatrixXd design = bvhar::build_x0(expand_mat[i], lag, include_mean);
			ols_objs[i] = std::unique_ptr<bvhar::MultiOls>(new bvhar::QrOls(design, expand_y0[i]));
		}
	}
	}
	std::vector<std::unique_ptr<bvhar::VarForecaster>> forecaster(num_horizon);
	std::vector<Eigen::MatrixXd> res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int window = 0; window < num_horizon; window++) {
		bvhar::OlsFit ols_fit = ols_objs[window]->returnOlsFit(lag);
		forecaster[window].reset(new bvhar::VarForecaster(ols_fit, step, expand_y0[window], include_mean));
		res[window] = forecaster[window]->forecastPoint().bottomRows(1);
		ols_objs[window].reset(); // free the memory by making nullptr
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
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
#endif
	int num_window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1; // longest forecast horizon
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
	std::vector<std::unique_ptr<bvhar::MultiOls>> ols_objs(num_horizon);
	switch(method) {
	case 1: {
		for (int i = 0; i < num_horizon; ++i) {
			Eigen::MatrixXd design = bvhar::build_x0(expand_mat[i], month, include_mean) * har_trans.transpose();
			ols_objs[i] = std::unique_ptr<bvhar::MultiOls>(new bvhar::MultiOls(design, expand_y0[i]));
		}
	}
	case 2: {
		for (int i = 0; i < num_horizon; ++i) {
			Eigen::MatrixXd design = bvhar::build_x0(expand_mat[i], month, include_mean) * har_trans.transpose();
			ols_objs[i] = std::unique_ptr<bvhar::MultiOls>(new bvhar::LltOls(design, expand_y0[i]));
		}
	}
	case 3: {
		for (int i = 0; i < num_horizon; ++i) {
			Eigen::MatrixXd design = bvhar::build_x0(expand_mat[i], month, include_mean) * har_trans.transpose();
			ols_objs[i] = std::unique_ptr<bvhar::MultiOls>(new bvhar::QrOls(design, expand_y0[i]));
		}
	}
	}
	std::vector<std::unique_ptr<bvhar::VharForecaster>> forecaster(num_horizon);
	std::vector<Eigen::MatrixXd> res(num_horizon);
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int window = 0; window < num_horizon; window++) {
		bvhar::OlsFit ols_fit = ols_objs[window]->returnOlsFit(month);
		forecaster[window].reset(new bvhar::VharForecaster(ols_fit, step, expand_y0[window], har_trans, include_mean));
		res[window] = forecaster[window]->forecastPoint().bottomRows(1);
		ols_objs[window].reset(); // free the memory by making nullptr
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

//' Generalized Spillover of VAR
//' 
//' @param object varlse or vharlse object.
//' @param step Step to forecast.
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List compute_ols_spillover(Rcpp::List object, int step) {
	if (!(object.inherits("varlse") || object.inherits("vharlse"))) {
    Rcpp::stop("'object' must be varlse or vharlse object.");
  }
	std::unique_ptr<Eigen::MatrixXd> coef_mat;
	int ord;
	if (object.inherits("vharlse")) {
		coef_mat.reset(new Eigen::MatrixXd(Rcpp::as<Eigen::MatrixXd>(object["HARtrans"]).transpose() * Rcpp::as<Eigen::MatrixXd>(object["coefficients"])));
		ord = object["month"];
	} else {
		coef_mat.reset(new Eigen::MatrixXd(Rcpp::as<Eigen::MatrixXd>(object["coefficients"])));
		ord = object["p"];
	}
	bvhar::StructuralFit fit(*coef_mat, ord, step - 1, Rcpp::as<Eigen::MatrixXd>(object["covmat"]));
	std::unique_ptr<bvhar::OlsSpillover> spillover(new bvhar::OlsSpillover(fit));
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
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	std::vector<std::unique_ptr<bvhar::OlsVar>> ols_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		ols_objs[i] = std::unique_ptr<bvhar::OlsVar>(new bvhar::OlsVar(roll_mat, lag, include_mean, method));
	}
	std::vector<std::unique_ptr<bvhar::OlsSpillover>> spillover(num_horizon);
	Eigen::VectorXd tot(num_horizon);
	Eigen::MatrixXd to_sp(num_horizon, y.cols());
	Eigen::MatrixXd from_sp(num_horizon, y.cols());
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
		bvhar::StructuralFit ols_fit = ols_objs[i]->returnStructuralFit(step - 1);
		spillover[i].reset(new bvhar::OlsSpillover(ols_fit));
		spillover[i]->computeSpillover();
		to_sp.row(i) = spillover[i]->returnTo();
		from_sp.row(i) = spillover[i]->returnFrom();
		tot[i] = spillover[i]->returnTot();
		ols_objs[i].reset(); // free the memory by making nullptr
		spillover[i].reset(); // free the memory by making nullptr
	}
	return Rcpp::List::create(
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = tot,
		Rcpp::Named("net") = to_sp - from_sp
	);
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
  int num_horizon = y.rows() - window + 1; // number of windows = T - win + 1
	if (num_horizon <= 0) {
		Rcpp::stop("Window size is too large.");
	}
	std::vector<std::unique_ptr<bvhar::OlsVhar>> ols_objs(num_horizon);
	for (int i = 0; i < num_horizon; ++i) {
		Eigen::MatrixXd roll_mat = y.middleRows(i, window);
		ols_objs[i] = std::unique_ptr<bvhar::OlsVhar>(new bvhar::OlsVhar(roll_mat, week, month, include_mean, method));
	}
	std::vector<std::unique_ptr<bvhar::OlsSpillover>> spillover(num_horizon);
	Eigen::VectorXd tot(num_horizon);
	Eigen::MatrixXd to_sp(num_horizon, y.cols());
	Eigen::MatrixXd from_sp(num_horizon, y.cols());
#ifdef _OPENMP
	#pragma omp parallel for num_threads(nthreads)
#endif
	for (int i = 0; i < num_horizon; ++i) {
		bvhar::StructuralFit ols_fit = ols_objs[i]->returnStructuralFit(step - 1);
		spillover[i].reset(new bvhar::OlsSpillover(ols_fit));
		spillover[i]->computeSpillover();
		to_sp.row(i) = spillover[i]->returnTo();
		from_sp.row(i) = spillover[i]->returnFrom();
		tot[i] = spillover[i]->returnTot();
		ols_objs[i].reset(); // free the memory by making nullptr
		spillover[i].reset(); // free the memory by making nullptr
	}
	return Rcpp::List::create(
		Rcpp::Named("to") = to_sp,
		Rcpp::Named("from") = from_sp,
		Rcpp::Named("tot") = tot,
		Rcpp::Named("net") = to_sp - from_sp
	);
}
