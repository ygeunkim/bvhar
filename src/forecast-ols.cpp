#include "olsforecaster.h"

//' Forecasting Vector Autoregression
//' 
//' @param object `varlse` object
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
//' @param object `vharlse` object
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
