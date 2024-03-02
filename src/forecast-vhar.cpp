#include "olsforecaster.h"

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

//' Out-of-Sample Forecasting of VHAR based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of VHAR.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param har `r lifecycle::badge("experimental")` Numeric vector for weekly and monthly order.
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_vhar(Eigen::MatrixXd y, 
                          Eigen::VectorXd har,
                          bool include_mean, 
                          int step,
                          Eigen::MatrixXd y_test) {
  Rcpp::Function fit("vhar_lm");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd roll_mat = y; // same size as y
  Rcpp::List vhar_mod = fit(roll_mat, har, include_mean);
  Eigen::MatrixXd y_pred = forecast_vhar(vhar_mod, step); // step x m
  Eigen::MatrixXd res(num_horizon, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_horizon; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    vhar_mod = fit(roll_mat, har, include_mean);
    y_pred = forecast_vhar(vhar_mod, step);
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}

//' Out-of-Sample Forecasting of VHAR based on Expanding Window
//' 
//' This function conducts an expanding window forecasting of VHAR.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param har `r lifecycle::badge("experimental")` Numeric vector for weekly and monthly order.
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd expand_vhar(Eigen::MatrixXd y, 
                            Eigen::VectorXd har,
                            bool include_mean, 
                            int step,
                            Eigen::MatrixXd y_test) {
  Rcpp::Function fit("vhar_lm");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_iter = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd expand_mat(window + num_iter, dim); // train + h-step forecast points
  expand_mat.block(0, 0, window, dim) = y;
  Rcpp::List vhar_mod = fit(y, har, include_mean);
  Eigen::MatrixXd y_pred = forecast_vhar(vhar_mod, step); // step x m
  Eigen::MatrixXd res(num_iter, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_iter; i++) {
    expand_mat.row(window + i - 1) = y_test.row(i - 1); // expanding window
    vhar_mod = fit(
      expand_mat.block(0, 0, window + i, dim),
      har,
      include_mean
    );
    y_pred = forecast_vhar(vhar_mod, step);
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}
