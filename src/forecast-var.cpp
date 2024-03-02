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

//' Out-of-Sample Forecasting of VAR based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of VAR.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag VAR order
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_var(Eigen::MatrixXd y, 
                         int lag, 
                         bool include_mean, 
                         int step,
                         Eigen::MatrixXd y_test) {
  Rcpp::Function fit("var_lm");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd roll_mat = y; // same size as y
  Rcpp::List var_mod = fit(roll_mat, lag, include_mean);
  Eigen::MatrixXd y_pred = forecast_var(var_mod, step); // step x m
  Eigen::MatrixXd res(num_horizon, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_horizon; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows: move one period
    var_mod = fit(roll_mat, lag, include_mean);
    y_pred = forecast_var(var_mod, step);
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
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
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd expand_var(Eigen::MatrixXd y, 
                           int lag, 
                           bool include_mean, 
                           int step,
                           Eigen::MatrixXd y_test) {
  Rcpp::Function fit("var_lm");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_iter = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd expand_mat(window + num_iter, dim); // train + h-step forecast points
  expand_mat.block(0, 0, window, dim) = y;
  Rcpp::List var_mod = fit(y, lag, include_mean);
  Eigen::MatrixXd y_pred = forecast_var(var_mod, step); // step x m
  Eigen::MatrixXd res(num_iter, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_iter; i++) {
    expand_mat.row(window + i - 1) = y_test.row(i - 1); // expanding window
    var_mod = fit(
      expand_mat.block(0, 0, window + i, dim),
      lag,
      include_mean
    );
    y_pred = forecast_var(var_mod, step);
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}