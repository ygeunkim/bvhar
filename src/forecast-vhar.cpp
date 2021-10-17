#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' Forecasting Vector HAR
//' 
//' @param object \code{varlse} object by \code{\link{vhar_lm}}
//' @param step Integer, Step to forecast
//' @details
//' n-step ahead forecasting using VHAR recursively.
//' 
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd forecast_vhar(Rcpp::List object, int step) {
  if (!object.inherits("vharlse")) Rcpp::stop("'object' must be vharlse object.");
  Eigen::MatrixXd response_mat = object["y0"]; // Y0
  Eigen::MatrixXd coef_mat = object["coefficients"]; // bhat
  int dim = object["m"]; // dimension of time series
  Eigen::MatrixXd HARtrans = object["HARtrans"]; // HAR transformation
  int num_design = object["obs"]; // s = n - p
  int dim_har = HARtrans.cols(); // 22m + 1 (const) or 22m (none)
  Eigen::MatrixXd last_pvec(1, dim_har); // vectorize the last 22 observation and include 1
  Eigen::MatrixXd tmp_vec(1, 21 * dim); // temporary vector to move first 21 observations of last_pvec
  Eigen::MatrixXd res(step, dim); // h x m matrix
  last_pvec(0, dim_har - 1) = 1.0;
  for (int i = 0; i < 22; i++) {
    last_pvec.block(0, i * dim, 1, dim) = response_mat.block(num_design - 1 - i, 0, 1, dim);
  }
  res.block(0, 0, 1, dim) = last_pvec * HARtrans.transpose() * coef_mat; // y(n + 1)^T = [y(n)^T, ..., y(n - p + 1)^T, 1] %*% t(HARtrans) %*% Phihat
  if (step == 1) return res;
  for (int i = 1; i < step; i++) { // Next h - 1: recursively
    tmp_vec = last_pvec.block(0, 0, 1, 21 * dim); // remove the last m (except 1)
    last_pvec.block(0, dim, 1, 21 * dim) = tmp_vec;
    last_pvec.block(0, 0, 1, dim) = res.block(i - 1, 0, 1, dim);
    res.block(i, 0, 1, dim) = last_pvec * HARtrans.transpose() * coef_mat; // y(n + 2)^T = [yhat(n + 1)^T, y(n)^T, ... y(n - p + 2)^T, 1] %*% t(HARtrans) %*% Phihat
  }
  return res;
}
