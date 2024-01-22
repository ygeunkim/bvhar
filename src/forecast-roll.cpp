#include <bvharomp.h>
#include <RcppEigen.h>
#include "fitvar.h"

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
  int num_horizon = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd roll_mat = y; // same size as y
  Rcpp::List bvar_mod = fit(roll_mat, lag, bayes_spec, include_mean);
  Rcpp::List bvar_pred = forecast_bvar(bvar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_horizon, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_horizon; i++) {
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
  int num_horizon = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd roll_mat = y; // same size as y
  Rcpp::List bvar_mod = fit(roll_mat, lag, bayes_spec, include_mean);
  Rcpp::List bvar_pred = forecast_bvar(bvar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_horizon, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_horizon; i++) {
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
//' @param har `r lifecycle::badge("experimental")` Numeric vector for weekly and monthly order.
//' @param bayes_spec List, BVHAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_bvhar(Eigen::MatrixXd y, 
                           Eigen::VectorXd har,
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
  int num_horizon = num_test - step + 1; // longest forecast horizon
  Eigen::MatrixXd roll_mat = y; // same size as y
  Rcpp::List bvhar_mod = fit(roll_mat, har, bayes_spec, include_mean);
  Rcpp::List bvhar_pred = forecast_bvharmn(bvhar_mod, step, 1);
  Eigen::MatrixXd y_pred = bvhar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_horizon, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
  for (int i = 1; i < num_horizon; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    bvhar_mod = fit(roll_mat, har, bayes_spec, include_mean);
    bvhar_pred = forecast_bvharmn(bvhar_mod, step, 1);
    y_pred = bvhar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
  return res;
}

//' Out-of-Sample Forecasting of VAR-SV based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVHAR with Minnesota prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param har `r lifecycle::badge("experimental")` Numeric vector for weekly and monthly order.
//' @param bayes_spec List, BVHAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads_roll Number of threads when rolling windows
//' @param nthreads_mod Number of threads when fitting models
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_bvarsv(Eigen::MatrixXd y, int lag, int num_iter, int num_burn, int thinning, Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test, int nthreads_roll, int nthreads_mod) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
  Rcpp::Function fit("bvar_sv");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
  Eigen::MatrixXd roll_mat = y;
  Rcpp::List bvar_mod = fit(roll_mat, lag, num_iter, num_burn, thinning, bayes_spec, include_mean, false, nthreads_mod);
  Eigen::MatrixXd y_pred = forecast_bvarsv(bvar_mod["p"], step, bvar_mod["y0"], bvar_mod["coefficients"]);
  // Eigen::MatrixXd y_pred = bvhar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_horizon, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
#ifdef _OPENMP
  Eigen::MatrixXd tot_mat(window + num_test, dim); // entire data set = train + test for parallel
  tot_mat.topRows(window) = y;
  tot_mat.bottomRows(num_test) = y_test;
  // shared(res, num_horizon, tot_mat, window, dim, fit, lag, num_iter, num_burn, thinning, bayes_spec, include_mean, step) \
#pragma omp parallel for num_threads(nthreads_roll) private(roll_mat, bvar_mod, y_pred)
  for (int i = 1; i < num_horizon; i++) {
    // roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    // roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    roll_mat = tot_mat.block(i, 0, window, dim);
    bvar_mod = fit(roll_mat, lag, num_iter, num_burn, thinning, bayes_spec, include_mean, false, nthreads_mod);
    y_pred = forecast_bvarsv(bvar_mod["p"], step, bvar_mod["y0"], bvar_mod["coefficients"]);
    res.row(i) = y_pred.row(step - 1);
  }
#else
  for (int i = 1; i < num_horizon; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    bvar_mod = fit(roll_mat, lag, num_iter, num_burn, thinning, bayes_spec, include_mean, false, nthreads_mod);
    y_pred = forecast_bvarsv(bvar_mod["p"], step, bvar_mod["y0"], bvar_mod["coefficients"]);
    res.row(i) = y_pred.row(step - 1);
  }
#endif
  return res;
}

//' Out-of-Sample Forecasting of VHAR-SV based on Rolling Window
//' 
//' This function conducts an rolling window forecasting of BVHAR with Minnesota prior.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param har `r lifecycle::badge("experimental")` Numeric vector for weekly and monthly order.
//' @param bayes_spec List, BVHAR specification
//' @param include_mean Add constant term
//' @param step Integer, Step to forecast
//' @param y_test Evaluation time series data period after `y`
//' @param nthreads_roll Number of threads when rolling windows
//' @param nthreads_mod Number of threads when fitting models
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd roll_bvharsv(Eigen::MatrixXd y, Eigen::VectorXi har,
                             int num_iter, int num_burn, int thinning,
                             Rcpp::List bayes_spec, bool include_mean, int step, Eigen::MatrixXd y_test,
                             int nthreads_roll, int nthreads_mod) {
  if (!bayes_spec.inherits("bvharspec")) {
    Rcpp::stop("'object' must be bvharspec object.");
  }
  Rcpp::Function fit("bvhar_sv");
  int window = y.rows();
  int dim = y.cols();
  int num_test = y_test.rows();
  int num_horizon = num_test - step + 1;
  Eigen::MatrixXd roll_mat = y;
  Rcpp::List bvhar_mod = fit(roll_mat, har, num_iter, num_burn, thinning, bayes_spec, include_mean, false, nthreads_mod);
  Eigen::MatrixXd y_pred = forecast_bvharsv(bvhar_mod["month"], step, bvhar_mod["y0"], bvhar_mod["coefficients"], bvhar_mod["HARtrans"]);
  // Eigen::MatrixXd y_pred = bvhar_pred["posterior_mean"]; // step x m
  Eigen::MatrixXd res(num_horizon, dim);
  res.row(0) = y_pred.row(step - 1); // only need the last one (e.g. step = h => h-th row)
#ifdef _OPENMP
  Eigen::MatrixXd tot_mat(window + num_test, dim); // entire data set = train + test for parallel
  tot_mat.topRows(window) = y;
  tot_mat.bottomRows(num_test) = y_test;
#pragma omp parallel for num_threads(nthreads_roll) private(roll_mat, bvhar_mod, y_pred)
  for (int i = 1; i < num_horizon; i++) {
    // roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    // roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    roll_mat = tot_mat.block(i, 0, window, dim);
    bvhar_mod = fit(roll_mat, har, num_iter, num_burn, thinning, bayes_spec, include_mean, false, nthreads_mod);
    y_pred = forecast_bvharsv(bvhar_mod["month"], step, bvhar_mod["y0"], bvhar_mod["coefficients"], bvhar_mod["HARtrans"]);
    // y_pred = bvhar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
#else
  for (int i = 1; i < num_horizon; i++) {
    roll_mat.block(0, 0, window - 1, dim) = roll_mat.block(1, 0, window - 1, dim); // rolling windows
    roll_mat.row(window - 1) = y_test.row(i - 1); // rolling windows
    bvhar_mod = fit(roll_mat, har, num_iter, num_burn, thinning, bayes_spec, include_mean, false, nthreads_mod);
    y_pred = forecast_bvharsv(bvhar_mod["month"], step, bvhar_mod["y0"], bvhar_mod["coefficients"], bvhar_mod["HARtrans"]);
    // y_pred = bvhar_pred["posterior_mean"];
    res.row(i) = y_pred.row(step - 1);
  }
#endif
  return res;
}
