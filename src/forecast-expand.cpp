#include <RcppEigen.h>
#include "fitvar.h"

// [[Rcpp::depends(RcppEigen)]]

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

//' Out-of-Sample Forecasting of VHAR based on Expanding Window
//' 
//' This function conducts an expanding window forecasting of VHAR.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd expand_vhar(Eigen::MatrixXd y, 
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
  Rcpp::List vhar_mod = fit(y, include_mean);
  Eigen::MatrixXd y_pred = forecast_vhar(vhar_mod, step); // step x m
  Eigen::MatrixXd res(num_iter, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_iter; i++) {
    expand_mat.row(window + i - 1) = y_test.row(i - 1); // expanding window
    vhar_mod = fit(
      expand_mat.block(0, 0, window + i, dim),
      include_mean
    );
    y_pred = forecast_vhar(vhar_mod, step);
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
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
Eigen::MatrixXd expand_bvar(Eigen::MatrixXd y, 
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
  Eigen::MatrixXd expand_mat(window + num_iter, dim); // train + h-step forecast points
  expand_mat.block(0, 0, window, dim) = y;
  Rcpp::List bvar_mod = fit(y, lag, bayes_spec, include_mean);
  Rcpp::List bvar_pred = forecast_bvar(bvar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_iter, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_iter; i++) {
    expand_mat.row(window + i - 1) = y_test.row(i - 1); // expanding window
    bvar_mod = fit(
      expand_mat.block(0, 0, window + i, dim),
      lag, 
      bayes_spec, 
      include_mean
    );
    bvar_pred = forecast_bvar(bvar_mod, step, 1);
    y_pred = bvar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
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
Eigen::MatrixXd expand_bvarflat(Eigen::MatrixXd y, 
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
  Eigen::MatrixXd expand_mat(window + num_iter, dim); // train + h-step forecast points
  expand_mat.block(0, 0, window, dim) = y;
  Rcpp::List bvar_mod = fit(y, lag, bayes_spec, include_mean);
  Rcpp::List bvar_pred = forecast_bvar(bvar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_iter, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_iter; i++) {
    expand_mat.row(window + i - 1) = y_test.row(i - 1); // expanding window
    bvar_mod = fit(
      expand_mat.block(0, 0, window + i, dim),
      lag, 
      bayes_spec, 
      include_mean
    );
    bvar_pred = forecast_bvar(bvar_mod, step, 1);
    y_pred = bvar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}

//' Out-of-Sample Forecasting of BVHAR based on Expanding Window
//' 
//' This function conducts an expanding window forecasting of BVHAR with Minnesota prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param bayes_spec List, BVHAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd expand_bvhar(Eigen::MatrixXd y, 
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
  Eigen::MatrixXd expand_mat(window + num_iter, dim); // train + h-step forecast points
  expand_mat.block(0, 0, window, dim) = y;
  Rcpp::List bvhar_mod = fit(y, bayes_spec, include_mean);
  Rcpp::List bvhar_pred = forecast_bvharmn(bvhar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvhar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_iter, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_iter; i++) {
    expand_mat.row(window + i - 1) = y_test.row(i - 1); // expanding window
    bvhar_mod = fit(
      expand_mat.block(0, 0, window + i, dim), 
      bayes_spec, 
      include_mean
    );
    bvhar_pred = forecast_bvharmn(bvhar_mod, step, 1);
    y_pred = bvhar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}
