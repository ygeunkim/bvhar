#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' Building a Linear Transformation Matrix for Vector HAR
//' 
//' This function produces a linear transformation matrix for VHAR for given dimension.
//' 
//' @param dim Integer, dimension
//' @param week Integer, order for weekly term
//' @param month Integer, order for monthly term
//' @param include_mean bool, Add constant term (Default: `true`) or not (`false`)
//' @details
//' VHAR is linearly restricted VAR(month = 22) in \eqn{Y_0 = X_0 A + Z}.
//' \deqn{Y_0 = X_1 \Phi + Z = (X_0 C_{HAR}^T) \Phi + Z}
//' This function computes above \eqn{C_{HAR}}.
//' 
//' Default VHAR model sets `week` and `month` as `5` and `22`.
//' This function can change these numbers to get linear transformation matrix.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd scale_har(int dim, int week, int month, bool include_mean) {
  if (week > month) {
    Rcpp::stop("'month' should be larger than 'week'.");
  }
  Eigen::MatrixXd HAR = Eigen::MatrixXd::Zero(3, month);
  Eigen::MatrixXd HARtrans(3 * dim + 1, month * dim + 1); // 3m x (month * m)
  Eigen::MatrixXd Im = Eigen::MatrixXd::Identity(dim, dim);
  HAR(0, 0) = 1.0;
  for (int i = 0; i < week; i++) {
    HAR(1, i) = 1.0 / week;
  }
  for (int i = 0; i < month; i++) {
    HAR(2, i) = 1.0 / month;
  }
  // T otimes Im
  HARtrans.block(0, 0, 3 * dim, month * dim) = Eigen::kroneckerProduct(HAR, Im).eval();
  HARtrans.block(0, month * dim, 3 * dim, 1) = Eigen::MatrixXd::Zero(3 * dim, 1);
  HARtrans.block(3 * dim, 0, 1, month * dim) = Eigen::MatrixXd::Zero(1, month * dim);
  HARtrans(3 * dim, month * dim) = 1.0;
  if (include_mean) {
    return HARtrans;
  }
  return HARtrans.block(0, 0, 3 * dim, month * dim);
}

//' Compute Vector HAR Coefficient Matrices and Fitted Values
//' 
//' This function fits VHAR given response and design matrices of multivariate time series.
//' 
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param week Integer, order for weekly term
//' @param month Integer, order for monthly term
//' @param include_mean bool, Add constant term (Default: `true`) or not (`false`)
//' @param method Method to solve linear equation system. 1: normal equation, 2: cholesky, 3: HouseholderQR.
//' @details
//' Given Y0 and Y0, the function estimate least squares
//' \deqn{Y_0 = X_1 \Phi + Z}
//' 
//' @references
//' Baek, C. and Park, M. (2021). *Sparse vector heterogeneous autoregressive modeling for realized volatility*. J. Korean Stat. Soc. 50, 495–510. doi:[10.1007/s42952-020-00090-5](https://doi.org/10.1007/s42952-020-00090-5)
//' 
//' Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174–196. doi:[10.1093/jjfinec/nbp001](https://doi.org/10.1093/jjfinec/nbp001)
//' @importFrom Rcpp sourceCpp
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_har(Eigen::MatrixXd x, Eigen::MatrixXd y, int week, int month, bool include_mean, int method) {
  int dim = y.cols();
  Eigen::MatrixXd HARtrans = scale_har(dim, week, month, include_mean);
  Eigen::MatrixXd x1 = x * HARtrans.transpose();
  Eigen::MatrixXd coef_mat(HARtrans.rows(), dim);
  switch (method) {
  case 1:
    coef_mat = (x1.transpose() * x1).inverse() * x1.transpose() * y;
    break;
  case 2:
    coef_mat = (x1.transpose() * x1).llt().solve(x1.transpose() * y);
    break;
  case 3:
    coef_mat = x1.householderQr().solve(y);
    break;
  default:
    break;
  }
  Eigen::MatrixXd yhat = x1 * coef_mat;
  return Rcpp::List::create(
    Rcpp::Named("HARtrans") = HARtrans,
    Rcpp::Named("phihat") = coef_mat,
    Rcpp::Named("fitted") = yhat
  );
}

//' Statistic for VHAR
//' 
//' Compute partial t-statistics for inference in VHAR model.
//' 
//' @param object `vharlse` object
//' @details
//' Partial t-statistic for H0: \eqn{\phi_{ij} = 0}
//' 
//' * For each variable (e.g. 1st variable)
//' * Standard error =  (1st) diagonal element of \eqn{\Sigma_e} estimator x diagonal elements of \eqn{(X_1^T X_1)^(-1)}
//' @noRd
// [[Rcpp::export]]
Rcpp::List infer_vhar(Rcpp::List object) {
  if (!object.inherits("vharlse")) {
    Rcpp::stop("'object' must be vharlse object.");
  }
  int dim = object["m"]; // dimension of time series
  Eigen::MatrixXd cov_mat = object["covmat"]; // sigma
  Eigen::MatrixXd coef_mat = object["coefficients"]; // Phihat(mp, m) = [Phi(daily), Phi(weekly), Phi(monthly), c^T]^T
  Eigen::MatrixXd design_mat = object["design"]; // X0: n x mp
  Eigen::MatrixXd HARtrans = object["HARtrans"]; // HAR transformation
  Eigen::MatrixXd vhar_design = design_mat * HARtrans.transpose(); // X1 = X0 * C0^T
  int num_design = object["obs"];
  int num_har = coef_mat.rows(); // 3m(+1)
  int df = num_design - num_har;
  Eigen::VectorXd XtX = (vhar_design.transpose() * vhar_design).inverse().diagonal(); // diagonal element of (XtX)^(-1)
  Eigen::MatrixXd res(num_har * dim, 3); // stack estimate, std, and t stat
  Eigen::ArrayXd st_err(num_har); // save standard error in for loop
  for (int i = 0; i < dim; i++) {
    res.block(i * num_har, 0, num_har, 1) = coef_mat.col(i);
    for (int j = 0; j < num_har; j++) {
      st_err[j] = sqrt(XtX[j] * cov_mat(i, i)); // variable-covariance matrix element
    }
    res.block(i * num_har, 1, num_har, 1) = st_err;
    res.block(i * num_har, 2, num_har, 1) = coef_mat.col(i).array() / st_err;
  }
  return Rcpp::List::create(
    Rcpp::Named("df") = df,
    Rcpp::Named("summary_stat") = res
  );
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd VHARcoeftoVMA(Eigen::MatrixXd vhar_coef, Eigen::MatrixXd HARtrans_mat, int lag_max, int month) {
  int dim = vhar_coef.cols(); // dimension of time series
  Eigen::MatrixXd coef_mat = HARtrans_mat.transpose() * vhar_coef; // bhat = tilde(T)^T * Phi
  if (lag_max < 1) {
    Rcpp::stop("'lag_max' must larger than 0");
  }
  int ma_rows = dim * (lag_max + 1);
  int num_full_arows = ma_rows;
  if (lag_max < month) num_full_arows = month * dim; // for VMA coefficient q < VAR(p)
  Eigen::MatrixXd FullA = Eigen::MatrixXd::Zero(num_full_arows, dim); // same size with VMA coefficient matrix
  FullA.block(0, 0, month * dim, dim) = coef_mat.block(0, 0, month * dim, dim); // fill first mp row with VAR coefficient matrix
  Eigen::MatrixXd Im = Eigen::MatrixXd::Identity(dim, dim); // identity matrix
  Eigen::MatrixXd ma = Eigen::MatrixXd::Zero(ma_rows, dim); // VMA [W1^T, W2^T, ..., W(lag_max)^T]^T, ma_rows = m * lag_max
  ma.block(0, 0, dim, dim) = Im; // W0 = Im
  ma.block(dim, 0, dim, dim) = FullA.block(0, 0, dim, dim) * ma.block(0, 0, dim, dim); // W1^T = B1^T * W1^T
  if (lag_max == 1) return ma;
  for (int i = 2; i < (lag_max + 1); i++) { // from W2: m-th row
    for (int k = 0; k < i; k++) {
      ma.block(i * dim, 0, dim, dim) += FullA.block(k * dim, 0, dim, dim) * ma.block((i - k - 1) * dim, 0, dim, dim); // Wi = sum(W(i - k)^T * Bk^T)
    }
  }
  return ma;
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
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd VHARtoVMA(Rcpp::List object, int lag_max) {
  if (!object.inherits("vharlse")) {
    Rcpp::stop("'object' must be vharlse object.");
  }
  Eigen::MatrixXd har_mat = object["coefficients"]; // Phihat(3m + 1, m) = [Phi(d)^T, Phi(w)^T, Phi(m)^T, c^T]^T
  Eigen::MatrixXd hartrans_mat = object["HARtrans"]; // tilde(T): (3m + 1, 22m + 1)
  int month = object["month"];
  Eigen::MatrixXd ma = VHARcoeftoVMA(har_mat, hartrans_mat, lag_max, month);
  return ma;
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
  int dim = object["m"]; // dimension of time series
  Eigen::MatrixXd cov_mat = object["covmat"]; // sigma
  Eigen::MatrixXd vma_mat = VHARtoVMA(object, step);
  Eigen::MatrixXd mse(dim * step, dim);
  mse.block(0, 0, dim, dim) = cov_mat; // sig(y) = sig
  for (int i = 1; i < step; i++) {
    mse.block(i * dim, 0, dim, dim) = mse.block((i - 1) * dim, 0, dim, dim) + 
      vma_mat.block(i * dim, 0, dim, dim).transpose() * cov_mat * vma_mat.block(i * dim, 0, dim, dim);
  }
  return mse;
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
  int dim = vhar_covmat.cols(); // num_rows = num_cols
  if ((dim != vhar_covmat.rows()) && (dim != vhar_coef.cols())) {
    Rcpp::stop("Wrong covariance matrix format: `vhar_covmat`.");
  }
  if ((vhar_coef.rows() != 3 * dim + 1) && (vhar_coef.rows() != 3 * dim)) {
    Rcpp::stop("Wrong VAR coefficient format: `vhar_coef`.");
  }
  Eigen::MatrixXd ma = VHARcoeftoVMA(vhar_coef, HARtrans_mat, lag_max, month);
  Eigen::MatrixXd res(ma.rows(), dim);
  Eigen::LLT<Eigen::MatrixXd> lltOfcovmat(Eigen::Map<Eigen::MatrixXd>(vhar_covmat.data(), dim, dim)); // cholesky decomposition for Sigma
  Eigen::MatrixXd chol_covmat = lltOfcovmat.matrixU();
  for (int i = 0; i < lag_max + 1; i++) {
    res.block(i * dim, 0, dim, dim) = chol_covmat * ma.block(i * dim, 0, dim, dim);
  }
  return res;
}
