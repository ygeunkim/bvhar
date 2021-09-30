#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' VAR(1) Representation of VAR(p)
//' 
//' Compute the coefficient matrix of VAR(1) form
//' 
//' @param object Model fit
//' @details
//' Each VAR(p) process can be represented by mp-dim VAR(1).
//' 
//' \deqn{Y_t = B Y_{t - 1} + C + U_t}
//' 
//' where
//' 
//' \deqn{
//'     B = \begin{bmatrix}
//'       B_1 & B_2 & \cdots B_{p - 1} & B_p \        \
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
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd compute_stablemat(Rcpp::List object) {
  int dim = object["m"]; // m
  int var_lag = object["p"]; // p
  Eigen::MatrixXd coef_mat = object["coefficients"]; // bhat
  Eigen::MatrixXd Im(dim, dim); // identity matrix
  Im.setIdentity(dim, dim);
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim * var_lag, dim * var_lag);
  res.block(0, 0, dim, dim * var_lag) = coef_mat.block(0, 0, dim * var_lag, dim).adjoint();
  for (int i = 1; i < var_lag; i++) {
    res.block(dim * i, dim * (i - 1), dim, dim) = Im;
  }
  return res;
}
