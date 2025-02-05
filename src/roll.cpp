#include <bvhar/forecast>

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
