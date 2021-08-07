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
SEXP estimate_var (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd B(x.cols(), y.cols()); // bhat
  Eigen::MatrixXd yhat(y.rows(), y.cols());
  B = (x.adjoint() * x).inverse() * x.adjoint() * y;
  yhat = x * B;
  return Rcpp::List::create(
    Rcpp::Named("bhat") = Rcpp::wrap(B),
    Rcpp::Named("fitted") = Rcpp::wrap(yhat)
  );
}

//' Convert VAR to VMA(infinite)
//' 
//' Convert VAR process to infinite vector MA process
//' 
//' @param var_coef Matrix (mp + 1 x m), VAR matrix augmented
//' @param p Integer, VAR lag
//' @details
//' Let VAR(p) be stable.
//' \deqn{Y_t = c + \sum_{j = 0} W_j Z_{t - j}}
//' For VAR coefficient \eqn{B_1, B_2, \ldots, B_p},
//' \deqn{I = (W_0 + W_1 L + W_2 L^2 + \cdots + ) (I - B_1 L - B_2 L^2 - \cdots - B_p L^p)}
//' Recursively,
//' \deqn{W_0 = I}
//' \deqn{W_1 = W_0 B_1}
//' \deqn{W_2 = W_1 B_1 + W_0 B_2}
//' \deqn{W_j = \sum_{j = 1}^k W_{k - j} B_j}
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP VARtoVMA(Eigen::MatrixXd var_coef, int p) {
  int num_vars = var_coef.cols(); // m
  int ma_rows = num_vars * p;
  Eigen::MatrixXd Im(num_vars, num_vars); // identity matrix
  Im.setIdentity(num_vars, num_vars);
  Eigen::MatrixXd ma = Eigen::MatrixXd::Zero(ma_rows, num_vars); // VMA [W1^T, W2^T, ..., W(lag_max)^T], lag_max = p
  ma.block(0, 0, num_vars, num_vars) = Im; // W0 = Im
  ma.block(num_vars, 0, num_vars, num_vars) = ma.block(0, 0, num_vars, num_vars) * var_coef.block(0, 0, num_vars, num_vars); // W1 = W1 B1
  // for (int i = 2; i < p; i++) {
  //   for (int k = 1; k < i; k++) {
  //     ma.block(i * num_vars, 0, num_vars, num_vars) += ma.block((i - k) * num_vars, 0, num_vars, num_vars) * var_coef.block((k - 1) * num_vars, 0, num_vars, num_vars);
  //   }
  // }
  for (int i = 1; i < p; i++) {
    for (int k = 1; k < i; k++) {
      ma.block(i * num_vars, 0, num_vars, num_vars) += ma.block((i - 1) * num_vars, 0, num_vars, num_vars) * var_coef.block((k - 1) * num_vars, 0, num_vars, num_vars) +
        ma.block((i - 2) * num_vars, 0, num_vars, num_vars) * var_coef.block(k * num_vars, 0, num_vars, num_vars);
    }
  }
  return Rcpp::wrap(ma);
}
