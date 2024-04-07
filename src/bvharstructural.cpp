#include "bvharstructural.h"

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd VARcoeftoVMA(Eigen::MatrixXd var_coef, int var_lag, int lag_max) {
  return bvhar::convert_var_to_vma(var_coef, var_lag, lag_max);
}

//' Convert VAR to VMA(infinite)
//' 
//' Convert VAR process to infinite vector MA process
//' 
//' @param object `varlse` object
//' @param lag_max Maximum lag for VMA
//' @details
//' Let VAR(p) be stable.
//' \deqn{Y_t = c + \sum_{j = 0} W_j Z_{t - j}}
//' For VAR coefficient \eqn{B_1, B_2, \ldots, B_p},
//' \deqn{I = (W_0 + W_1 L + W_2 L^2 + \cdots + ) (I - B_1 L - B_2 L^2 - \cdots - B_p L^p)}
//' Recursively,
//' \deqn{W_0 = I}
//' \deqn{W_1 = W_0 B_1 (W_1^T = B_1^T W_0^T)}
//' \deqn{W_2 = W_1 B_1 + W_0 B_2 (W_2^T = B_1^T W_1^T + B_2^T W_0^T)}
//' \deqn{W_j = \sum_{j = 1}^k W_{k - j} B_j (W_j^T = \sum_{j = 1}^k B_j^T W_{k - j}^T)}
//' @return VMA coefficient of k(lag-max + 1) x k dimension
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd VARtoVMA(Rcpp::List object, int lag_max) {
  if (!object.inherits("varlse")) {
    Rcpp::stop("'object' must be varlse object.");
  }
  Eigen::MatrixXd coef_mat = object["coefficients"]; // bhat(k, m) = [B1^T, B2^T, ..., Bp^T, c^T]^T
  int var_lag = object["p"];
  Eigen::MatrixXd ma = bvhar::convert_var_to_vma(coef_mat, var_lag, lag_max);
  return ma;
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_var_mse(Eigen::MatrixXd cov_mat, Eigen::MatrixXd var_coef, int var_lag, int step) {
  int dim = cov_mat.cols(); // dimension of time series
  Eigen::MatrixXd vma_mat = bvhar::convert_var_to_vma(var_coef, var_lag, step);
  Eigen::MatrixXd innov_account = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::MatrixXd mse = Eigen::MatrixXd::Zero(dim * step, dim);
  for (int i = 0; i < step; i++) {
    innov_account += vma_mat.block(i * dim, 0, dim, dim).transpose() * cov_mat * vma_mat.block(i * dim, 0, dim, dim);
    mse.block(i * dim, 0, dim, dim) = innov_account;
  }
  return mse;
}

//' Compute Forecast MSE Matrices
//' 
//' Compute the forecast MSE matrices using VMA coefficients
//' 
//' @param object `varlse` object
//' @param step Integer, Step to forecast
//' @details
//' See pp38 of Lütkepohl (2007).
//' Let \eqn{\Sigma} be the covariance matrix of VAR and let \eqn{W_j} be the VMA coefficients.
//' Recursively,
//' \deqn{\Sigma_y(1) = \Sigma}
//' \deqn{\Sigma_y(2) = \Sigma + W_1 \Sigma W_1^T}
//' \deqn{\Sigma_y(3) = \Sigma_y(2) + W_2 \Sigma W_2^T}
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_covmse(Rcpp::List object, int step) {
  if (!object.inherits("varlse")) {
    Rcpp::stop("'object' must be varlse object.");
  }
  return compute_var_mse(object["covmat"], object["coefficients"], object["p"], step);
}

//' Convert VAR to Orthogonalized VMA(infinite)
//' 
//' Convert VAR process to infinite orthogonalized vector MA process
//' 
//' @param var_coef VAR coefficient matrix
//' @param var_covmat VAR covariance matrix
//' @param var_lag VAR order
//' @param lag_max Maximum lag for VMA
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd VARcoeftoVMA_ortho(Eigen::MatrixXd var_coef, Eigen::MatrixXd var_covmat, int var_lag, int lag_max) {
  return bvhar::convert_vma_ortho(var_coef, var_covmat, var_lag, lag_max);
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd VHARcoeftoVMA(Eigen::MatrixXd vhar_coef, Eigen::MatrixXd HARtrans_mat, int lag_max, int month) {
  return bvhar::convert_vhar_to_vma(vhar_coef, HARtrans_mat, lag_max, month);
}

//' Convert VHAR to VMA(infinite)
//' 
//' Convert VHAR process to infinite vector MA process
//' 
//' @param object `vharlse` object
//' @param lag_max Maximum lag for VMA
//' @details
//' Let VAR(p) be stable
//' and let VAR(p) be
//' \eqn{Y_0 = X_0 B + Z}
//' 
//' VHAR is VAR(22) with
//' \deqn{Y_0 = X_1 B + Z = ((X_0 \tilde{T}^T)) \Phi + Z}
//' 
//' Observe that
//' \deqn{B = \tilde{T}^T \Phi}
//' @return VMA coefficient of k(lag-max + 1) x k dimension
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd VHARtoVMA(Rcpp::List object, int lag_max) {
  if (!object.inherits("vharlse")) {
    Rcpp::stop("'object' must be vharlse object.");
  }
  Eigen::MatrixXd har_mat = object["coefficients"]; // Phihat(3m + 1, m) = [Phi(d)^T, Phi(w)^T, Phi(m)^T, c^T]^T
  Eigen::MatrixXd hartrans_mat = object["HARtrans"]; // tilde(T): (3m + 1, 22m + 1)
  int month = object["month"];
  Eigen::MatrixXd ma = bvhar::convert_vhar_to_vma(har_mat, hartrans_mat, lag_max, month);
  return ma;
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_vhar_mse(Eigen::MatrixXd cov_mat,
                                 Eigen::MatrixXd vhar_coef,
                                 Eigen::MatrixXd har_trans,
                                 int month,
                                 int step) {
  int dim = cov_mat.cols(); // dimension of time series
  Eigen::MatrixXd vma_mat = bvhar::convert_vhar_to_vma(vhar_coef, har_trans, month, step);
  Eigen::MatrixXd mse(dim * step, dim);
  mse.block(0, 0, dim, dim) = cov_mat; // sig(y) = sig
  for (int i = 1; i < step; i++) {
    mse.block(i * dim, 0, dim, dim) = mse.block((i - 1) * dim, 0, dim, dim) + 
      vma_mat.block(i * dim, 0, dim, dim).transpose() * cov_mat * vma_mat.block(i * dim, 0, dim, dim);
  }
  return mse;
}

//' Compute Forecast MSE Matrices for VHAR
//' 
//' Compute the forecast MSE matrices using VMA coefficients
//' 
//' @param object \code{varlse} object by \code{\link{var_lm}}
//' @param step Integer, Step to forecast
//' @details
//' See pp38 of Lütkepohl (2007).
//' Let \eqn{\Sigma} be the covariance matrix of VHAR and let \eqn{W_j} be the VMA coefficients.
//' Recursively,
//' \deqn{\Sigma_y(1) = \Sigma}
//' \deqn{\Sigma_y(2) = \Sigma + W_1 \Sigma W_1^T}
//' \deqn{\Sigma_y(3) = \Sigma_y(2) + W_2 \Sigma W_2^T}
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_covmse_har(Rcpp::List object, int step) {
  if (!object.inherits("vharlse")) {
    Rcpp::stop("'object' must be vharlse object.");
  }
  return compute_vhar_mse(
    object["covmat"],
    object["coefficients"],
    object["HARtrans"],
    object["month"],
    step
  );
}

//' Orthogonal Impulse Response Functions of VHAR
//' 
//' Compute orthogonal impulse responses of VHAR
//' 
//' @param vhar_coef VHAR coefficient
//' @param vhar_covmat VHAR covariance matrix
//' @param HARtrans_mat HAR linear transformation matrix
//' @param lag_max Maximum lag for VMA
//' @param month Order for monthly term
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd VHARcoeftoVMA_ortho(Eigen::MatrixXd vhar_coef, 
                                    Eigen::MatrixXd vhar_covmat, 
                                    Eigen::MatrixXd HARtrans_mat, 
                                    int lag_max, 
                                    int month) {
  return bvhar::convert_vhar_vma_ortho(vhar_coef, vhar_covmat, HARtrans_mat, lag_max, month);
}

//' h-step ahead Forecast Error Variance Decomposition
//' 
//' [w_(h = 1, ij)^T, w_(h = 2, ij)^T, ...]
//'
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_fevd(Eigen::MatrixXd vma_coef, Eigen::MatrixXd cov_mat, bool normalize) {
  return bvhar::compute_vma_fevd(vma_coef, cov_mat, normalize);
}

//' h-step ahead Normalized Spillover
//'
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_spillover(Eigen::MatrixXd fevd) {
  return fevd.bottomRows(fevd.cols()) * 100;
}

//' To-others Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd compute_to_spillover(Eigen::MatrixXd spillover) {
  return bvhar::compute_to(spillover);
}

//' From-others Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd compute_from_spillover(Eigen::MatrixXd spillover) {
  return bvhar::compute_from(spillover);
}

//' Total Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
double compute_tot_spillover(Eigen::MatrixXd spillover) {
  return bvhar::compute_tot(spillover);
}

//' Net Pairwise Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_net_spillover(Eigen::MatrixXd spillover) {
  return bvhar::compute_net(spillover);
}
