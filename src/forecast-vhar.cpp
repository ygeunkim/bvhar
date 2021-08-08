#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' Forecasting Vector HAR
//' 
//' @param object \code{varlse} object by \code{\link{vhar_lm}}
//' @param step Integer, Step to forecast
//' @details
//' n-step ahead forecasting using VHAR recursively.
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP forecast_vhar(Rcpp::List object, int step) {
  if (! object.inherits("vharlse")) Rcpp::stop("'object' must be vharlse object.");
  Eigen::MatrixXd response_mat = object["y0"]; // Y0
  Eigen::MatrixXd coef_mat = object["coefficients"]; // bhat
  int dim = object["m"]; // dimension of time series
  Eigen::MatrixXd HARtrans = object["HARtrans"]; // HAR transformation
  int num_design = object["obs"]; // s = n - p
  Eigen::MatrixXd last_pvec(1, 22 * dim + 1); // vectorize the last 22 observation and include 1
  Eigen::MatrixXd tmp_vec(1, 22 * dim);
  Eigen::MatrixXd res(step, dim); // h x m matrix
  for (int i = 0; i < 22; i++) {
    last_pvec.block(0, i * dim, 1, dim) = response_mat.block(num_design - 1 - i, 0, 1, dim);
  }
  last_pvec(0, 22 * dim) = 1.0;
  res.block(0, 0, 1, dim) = last_pvec * HARtrans.adjoint() * coef_mat; // y(n + 1)^T = [y(n)^T, ..., y(n - p + 1)^T, 1] %*% t(HARtrans) %*% Phihat
  if (step == 1) return Rcpp::wrap(res);
  // Next h - 1: recursively
  for (int i = 1; i < step; i++) {
    tmp_vec = last_pvec.block(0, 0, 1, 21 * dim); // remove the last m (except 1)
    last_pvec.block(0, dim, 1, 21 * dim) = tmp_vec;
    last_pvec.block(0, 0, 1, dim) = res.block(i - 1, 0, 1, dim);
    // y(n + 2)^T = [yhat(n + 1)^T, y(n)^T, ... y(n - p + 2)^T, 1] %*% t(HARtrans) %*% Phihat
    res.block(i, 0, 1, dim) = last_pvec * HARtrans.adjoint() * coef_mat;
  }
  return Rcpp::wrap(res);
}
