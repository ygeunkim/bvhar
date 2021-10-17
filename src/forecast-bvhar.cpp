#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' Forecasting Bayesian VHAR
//' 
//' @param object \code{bvharmn} object by \code{\link{vhar_lm}}
//' @param step Integer, Step to forecast
//' @details
//' n-step ahead forecasting using VHAR recursively.
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List forecast_bvharmn(Rcpp::List object, int step) {
  if (!object.inherits("bvharmn")) Rcpp::stop("'object' must be bvharmn object.");
  Eigen::MatrixXd response_mat = object["y0"]; // Y0
  Eigen::MatrixXd posterior_mean_mat = object["coefficients"]; // Phihat = posterior mean of MN
  Eigen::MatrixXd posterior_prec_mat = object["mn_prec"]; // Psihat = posterior precision of MN to compute SE
  Eigen::MatrixXd HARtrans = object["HARtrans"]; // HAR transformation
  Eigen::MatrixXd transformed_prec_mat = HARtrans.transpose() * posterior_prec_mat.inverse() * HARtrans; // to compute SE: play a role V in BVAR
  int dim = object["m"]; // dimension of time series
  int num_design = object["obs"]; // s = n - p
  int dim_har = HARtrans.cols(); // 22m + 1 (const) or 22m (none)
  Eigen::MatrixXd last_pvec(1, dim_har); // vectorize the last 22 observation and include 1
  Eigen::MatrixXd tmp_vec(1, 21 * dim);
  Eigen::MatrixXd res(step, dim); // h x m matrix
  Eigen::VectorXd sig_closed(step); // se^2 for each forecast (except Sigma2 part, i.e. closed form)
  for (int i = 0; i < step; i++) {
    sig_closed(i) = 1.0;
  }
  last_pvec(0, dim_har - 1) = 1.0;
  for (int i = 0; i < 22; i++) {
    last_pvec.block(0, i * dim, 1, dim) = response_mat.block(num_design - 1 - i, 0, 1, dim);
  }
  sig_closed.block(0, 0, 1, 1) += last_pvec * transformed_prec_mat * last_pvec.transpose();
  res.block(0, 0, 1, dim) = last_pvec * HARtrans.transpose() * posterior_mean_mat; // y(n + 1)^T = [y(n)^T, ..., y(n - p + 1)^T, 1] %*% t(HARtrans) %*% Phihat
  if (step == 1) {
    return Rcpp::List::create(
      Rcpp::Named("posterior_mean") = res,
      Rcpp::Named("posterior_var_closed") = sig_closed
    );
  }
  // Next h - 1: recursively
  for (int i = 1; i < step; i++) {
    tmp_vec = last_pvec.block(0, 0, 1, 21 * dim); // remove the last m (except 1)
    last_pvec.block(0, dim, 1, 21 * dim) = tmp_vec;
    last_pvec.block(0, 0, 1, dim) = res.block(i - 1, 0, 1, dim);
    sig_closed.block(i, 0, 1, 1) += last_pvec * transformed_prec_mat * last_pvec.transpose();
    // y(n + 2)^T = [yhat(n + 1)^T, y(n)^T, ... y(n - p + 2)^T, 1] %*% t(HARtrans) %*% Phihat
    res.block(i, 0, 1, dim) = last_pvec * HARtrans.transpose() * posterior_mean_mat;
  }
  return Rcpp::List::create(
    Rcpp::Named("posterior_mean") = res,
    Rcpp::Named("posterior_var_closed") = sig_closed
  );
}
