#include <RcppEigen.h>
#include "fitvar.h"

// [[Rcpp::depends(RcppEigen)]]

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
  int num_iter = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd roll_mat = y; // same size as y
  Rcpp::List var_mod = fit(roll_mat, lag, include_mean);
  Eigen::MatrixXd y_pred = forecast_var(var_mod, step); // step x m
  Eigen::MatrixXd res(num_iter, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_iter; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows: move one period
    var_mod = fit(roll_mat, lag, include_mean);
    y_pred = forecast_var(var_mod, step);
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}

//' Out-of-Sample Forecasting of VHAR based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of VHAR.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_vhar(Eigen::MatrixXd y, 
                          bool include_mean, 
                          int step,
                          Eigen::MatrixXd y_test) {
  Rcpp::Function fit("vhar_lm");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_iter = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd roll_mat = y; // same size as y
  Rcpp::List vhar_mod = fit(roll_mat, include_mean);
  Eigen::MatrixXd y_pred = forecast_vhar(vhar_mod, step); // step x m
  Eigen::MatrixXd res(num_iter, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_iter; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    vhar_mod = fit(roll_mat, include_mean);
    y_pred = forecast_vhar(vhar_mod, step);
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
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
Eigen::MatrixXd roll_bvar(Eigen::MatrixXd y, 
                          int lag, 
                          Rcpp::List bayes_spec,
                          bool include_mean, 
                          int step,
                          Eigen::MatrixXd y_test) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
  Rcpp::Function fit("bvar_minnesota");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_iter = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd roll_mat = y; // same size as y
  Rcpp::List bvar_mod = fit(roll_mat, lag, bayes_spec, include_mean);
  Rcpp::List bvar_pred = forecast_bvar(bvar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_iter, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_iter; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    bvar_mod = fit(roll_mat, lag, bayes_spec, include_mean);
    bvar_pred = forecast_bvar(bvar_mod, step, 1);
    y_pred = bvar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
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
Eigen::MatrixXd roll_bvarflat(Eigen::MatrixXd y, 
                              int lag, 
                              Rcpp::List bayes_spec,
                              bool include_mean, 
                              int step,
                              Eigen::MatrixXd y_test) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
  Rcpp::Function fit("bvar_flat");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_iter = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd roll_mat = y; // same size as y
  Rcpp::List bvar_mod = fit(roll_mat, lag, bayes_spec, include_mean);
  Rcpp::List bvar_pred = forecast_bvar(bvar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_iter, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_iter; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    bvar_mod = fit(roll_mat, lag, bayes_spec, include_mean);
    bvar_pred = forecast_bvar(bvar_mod, step, 1);
    y_pred = bvar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}

//' Out-of-Sample Forecasting of BVHAR based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVHAR with Minnesota prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param bayes_spec List, BVHAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_bvhar(Eigen::MatrixXd y, 
                           Rcpp::List bayes_spec,
                           bool include_mean, 
                           int step,
                           Eigen::MatrixXd y_test) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
  Rcpp::Function fit("bvhar_minnesota");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_iter = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd roll_mat = y; // same size as y
  Rcpp::List bvhar_mod = fit(roll_mat, bayes_spec, include_mean);
  Rcpp::List bvhar_pred = forecast_bvharmn(bvhar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvhar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_iter, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_iter; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    bvhar_mod = fit(roll_mat, bayes_spec, include_mean);
    bvhar_pred = forecast_bvharmn(bvhar_mod, step, 1);
    y_pred = bvhar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}
