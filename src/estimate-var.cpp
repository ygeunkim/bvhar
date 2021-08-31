#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' Compute VAR(p) Coefficient Matrices and Fitted Values
//' 
//' @param x X0 processed by \code{\link{build_design}}
//' @param y Y0 processed by \code{\link{build_y0}}
//' @details
//' Given Y0 and Y0, the function estimate least squares
//' Y0 = X0 B + Z
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
Rcpp::List estimate_var (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd B(x.cols(), y.cols()); // bhat
  Eigen::MatrixXd yhat(y.rows(), y.cols());
  B = (x.adjoint() * x).inverse() * x.adjoint() * y;
  yhat = x * B;
  return Rcpp::List::create(
    Rcpp::Named("bhat") = Rcpp::wrap(B),
    Rcpp::Named("fitted") = Rcpp::wrap(yhat)
  );
}

//' Covariance Estimate for Residual Covariance Matrix
//' 
//' Plausible estimator for residual covariance.
//' 
//' @param z Matrix, residual
//' @param num_design Integer, s = n - p
//' @param dim_design Ingeger, k = mp + 1
//' @details
//' See Lütkepohl (2007).
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @useDynLib bvhar
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd compute_cov (Eigen::MatrixXd z, int num_design, int dim_design) {
  Eigen::MatrixXd cov_mat(z.cols(), z.cols());
  cov_mat = z.adjoint() * z / (num_design - dim_design);
  return cov_mat;
}

//' @useDynLib bvhar
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd VARcoeftoVMA(Eigen::MatrixXd var_coef, int var_lag, int lag_max) {
  int dim = var_coef.cols(); // m
  if (lag_max < 1) Rcpp::stop("'lag_max' must larger than 0");
  int ma_rows = dim * (lag_max + 1);
  int num_full_brows = ma_rows;
  if (lag_max < var_lag) num_full_brows = dim * var_lag; // for VMA coefficient q < VAR(p)
  Eigen::MatrixXd FullB = Eigen::MatrixXd::Zero(num_full_brows, dim); // same size with VMA coefficient matrix
  FullB.block(0, 0, dim * var_lag, dim) = var_coef.block(0, 0, dim * var_lag, dim); // fill first mp row with VAR coefficient matrix
  Eigen::MatrixXd Im(dim, dim); // identity matrix
  Im.setIdentity(dim, dim);
  Eigen::MatrixXd ma = Eigen::MatrixXd::Zero(ma_rows, dim); // VMA [W1^T, W2^T, ..., W(lag_max)^T]^T, ma_rows = m * lag_max
  ma.block(0, 0, dim, dim) = Im; // W0 = Im
  ma.block(dim, 0, dim, dim) = FullB.block(0, 0, dim, dim) * ma.block(0, 0, dim, dim); // W1^T = B1^T * W1^T
  if (lag_max == 1) return ma;
  for (int i = 2; i < (lag_max + 1); i++) { // from W2: m-th row
    for (int k = 0; k < i; k++) {
      ma.block(i * dim, 0, dim, dim) += FullB.block(k * dim, 0, dim, dim) * ma.block((i - k - 1) * dim, 0, dim, dim); // Wi = sum(W(i - k)^T * Bk^T)
    }
  }
  return ma;
}

//' Convert VAR to VMA(infinite)
//' 
//' Convert VAR process to infinite vector MA process
//' 
//' @param object \code{varlse} object by \code{\link{var_lm}}
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
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @useDynLib bvhar
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd VARtoVMA(Rcpp::List object, int lag_max) {
  if (!object.inherits("varlse")) Rcpp::stop("'object' must be varlse object.");
  Eigen::MatrixXd coef_mat = object["coefficients"]; // bhat(k, m) = [B1^T, B2^T, ..., Bp^T, c^T]^T
  int var_lag = object["p"];
  Eigen::MatrixXd ma = VARcoeftoVMA(coef_mat, var_lag, lag_max);
  return ma;
}

//' Compute Forecast MSE Matrices
//' 
//' Compute the forecast MSE matrices using VMA coefficients
//' 
//' @param object \code{varlse} object by \code{\link{var_lm}}
//' @param step Integer, Step to forecast
//' @details
//' See pp38 of Lütkepohl (2007).
//' Let \eqn{\Sigma} be the covariance matrix of VAR and let \eqn{W_j} be the VMA coefficients.
//' Recursively,
//' \deqn{\Sigma_y(1) = \Sigma}
//' \deqn{\Sigma_y(2) = \Sigma + W_1 \Sigma W_1^T}
//' \deqn{\Sigma_y(3) = \Sigma_y(2) + W_2 \Sigma W_2^T}
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @useDynLib bvhar
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd compute_covmse(Rcpp::List object, int step) {
  if (!object.inherits("varlse")) Rcpp::stop("'object' must be varlse object.");
  int dim = object["m"]; // dimension of time series
  Eigen::MatrixXd cov_mat = object["covmat"]; // sigma
  Eigen::MatrixXd vma_mat = VARtoVMA(object, step);
  Eigen::MatrixXd mse(dim * step, dim);
  mse.block(0, 0, dim, dim) = cov_mat; // sig(y) = sig
  for (int i = 1; i < step; i++) {
    mse.block(i * dim, 0, dim, dim) = mse.block((i - 1) * dim, 0, dim, dim) + 
      vma_mat.block(i * dim, 0, dim, dim).adjoint() * cov_mat * vma_mat.block(i * dim, 0, dim, dim);
  }
  return mse;
}

