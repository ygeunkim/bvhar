#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' VAR(1) Representation Given VAR Coefficient Matrix
//' 
//' Compute the VAR(1) coefficient matrix form
//' 
//' @param x VAR without constant coefficient matrix form
//' @details
//' Each VAR(p) process can be represented by mp-dim VAR(1).
//' 
//' \deqn{Y_t = A Y_{t - 1} + C + U_t}
//' 
//' where
//' 
//' \deqn{
//'     A = \begin{bmatrix}
//'       A_1 & A_2 & \cdots A_{p - 1} & A_p \        \
//'       I_m & 0 & \cdots & 0 & 0 \                  \
//'       0 & I_m & \cdots & 0 & 0 \                  \
//'       \vdots & \vdots & \vdots & \vdots & \vdots \\
//'       0 & 0 & \cdots & I_m & 0
//'     \end{bmatrix}
//' }
//' 
//' \deqn{C = (c, 0, \ldots, 0)^T}
//' 
//' and
//' 
//' \deqn{U_t = (\epsilon_t, 0, \ldots, 0)^T}
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_stablemat(Eigen::MatrixXd x) {
  int dim = x.cols(); // m
  int var_lag = x.rows() / dim; // p
  Eigen::MatrixXd Im(dim, dim); // identity matrix
  Im.setIdentity();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim * var_lag, dim * var_lag);
  res.block(0, 0, dim, dim * var_lag) = x.transpose();
  for (int i = 1; i < var_lag; i++) {
    res.block(dim * i, dim * (i - 1), dim, dim) = Im;
  }
  return res;
}

//' VAR(1) Representation of VAR(p)
//' 
//' Compute the coefficient matrix of VAR(1) form
//' 
//' @param object Model fit
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_var_stablemat(Rcpp::List object) {
  if (!object.inherits("varlse") && !object.inherits("bvarmn") && !object.inherits("bvarflat")) {
    Rcpp::stop("'object' must be varlse object.");
  }
  int dim = object["m"]; // m
  int var_lag = object["p"]; // p
  Eigen::MatrixXd coef_mat = object["coefficients"]; // Ahat
  Eigen::MatrixXd coef_without_const = coef_mat.block(0, 0, dim * var_lag, dim);
  Eigen::MatrixXd res = compute_stablemat(coef_without_const);
  return res;
}

//' VAR(1) Representation of VHAR
//'
//' Compute the coefficient matrix of VAR(1) form of VHAR
//'
//' @param object Model fit
//' @details
//' Note that \eqn{A^T = \Phi^T T_{HAR}}.
//' This gives the VAR(1) form of constrained VAR(22).
//'
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_vhar_stablemat(Rcpp::List object) {
  if (!object.inherits("vharlse") && !object.inherits("bvharmn")) {
    Rcpp::stop("'object' must be varlse object.");
  }
  int dim = object["m"]; // m
  Eigen::MatrixXd coef_mat = object["coefficients"]; // Phihat
  Eigen::MatrixXd hartrans_mat = object["HARtrans"]; // HAR transformation: (3m + 1, 22m + 1)
  Eigen::MatrixXd coef_without_const = coef_mat.block(0, 0, 3 * dim, dim);
  Eigen::MatrixXd hartrans_without_const = hartrans_mat.block(0, 0, 3 * dim, 22 * dim); // 3m x 22m
  Eigen::MatrixXd res = compute_stablemat(hartrans_without_const.transpose() * coef_without_const);
  return res;
}
