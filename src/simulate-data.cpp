#include <RcppEigen.h>
#include "bvharmisc.h"

// [[Rcpp::depends(RcppEigen)]]

//' Generate Multivariate Time Series Process Following VAR(p)
//' 
//' This function generates multivariate time series dataset that follows VAR(p).
//' 
//' @param num_sim Number to generated process
//' @param var_coef VAR coefficient. The format should be the same as the output of \code{\link{var_lm}}
//' @param var_lag Lag of VAR
//' @param sig_error Variance matrix of the error term. Try \code{diag(dim)}.
//' @param init Initial y1, ..., yp matrix to simulate VAR model. Try \code{matrix(0L, nrow = var_lag, ncol = dim)}.
//' @details
//' Recall the relation between stable VAR(p) and VMA.
//' 
//' @seealso 
//' \code{\link{VARtoVMA}} computes VMA representation.
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @useDynLib bvhar
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_var(int num_sim, Eigen::MatrixXd var_coef, int var_lag, Eigen::MatrixXd sig_error, Eigen::MatrixXd init) {
  int dim = sig_error.cols(); // m: dimension of time series
  int dim_design = dim * var_lag + 1; // k = mp + 1 if const_term exists
  if (num_sim < 2) Rcpp::stop("Generate more than 1 series");
  if (var_coef.cols() != dim) Rcpp::stop("Wrong VAR coefficient format or Variance matrix");
  if (var_coef.rows() != dim_design) Rcpp::stop("`var_coef` is not VAR coefficient. Check its dimension.");
  if (!(init.rows() == var_lag && init.cols() == dim)) Rcpp::stop("`init` is `var_lag` x `dim` matrix in order of y1, y2, ..., yp.");
  Eigen::MatrixXd obs_p(1, dim_design); // row vector of X0: yp^T, ..., y1^T, 1
  obs_p(0, dim_design - 1) = 1.0; // for constant term
  for (int i = 0; i < var_lag; i++) {
    obs_p.block(0, i * dim, 1, dim) = init.row(var_lag - i - 1);
  }
  Eigen::MatrixXd res(num_sim, dim); // Output: from y(p + 1)^T to y(n + p)^T
  Eigen::MatrixXd error_term = sim_mgaussian(num_sim, sig_error); // simulated error term: num_sim x m
  res.row(0) = obs_p * var_coef + error_term.row(0);
  for (int i = 1; i < num_sim; i++) {
    for (int t = 1; t < var_lag; t++) {
      obs_p.block(0, t * dim, 1, dim) = obs_p.block(0, (t - 1) * dim, 1, dim);
    }
    obs_p.block(0, 0, 1, dim) = res.row(i - 1);
    res.row(i) = obs_p * var_coef + error_term.row(i);
  }
  return res;
}

//' Generate Multivariate Time Series Process Following VHAR
//' 
//' This function generates multivariate time series dataset that follows VHAR.
//' 
//' @param num_sim Number to generated process
//' @param vhar_coef VHAR coefficient. The format should be the same as the output of \code{\link{vhar_lm}}
//' @param sig_error Variance matrix of the error term. Try \code{diag(dim)}.
//' @param init Initial y1, ..., yp matrix to simulate VAR model. Try \code{matrix(0L, nrow = 22L, ncol = dim)}.
//' @details
//' Recall the relation between stable VHAR and VMA.
//' 
//' @seealso 
//' \code{\link{VHARtoVMA}} computes VMA representation.
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' @useDynLib bvhar
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_vhar(int num_sim, Eigen::MatrixXd vhar_coef, Eigen::MatrixXd sig_error, Eigen::MatrixXd init) {
  int dim = sig_error.cols(); // m: dimension of time series
  int dim_design = 3 * dim + 1;
  int var_design = 22 * dim + 1;
  if (vhar_coef.cols() != dim) Rcpp::stop("Wrong VHAR coefficient format or Variance matrix");
  if (num_sim < 2) Rcpp::stop("Generate more than 1 series");
  if (vhar_coef.rows() != dim_design) Rcpp::stop("`vhar_coef` is not VHAR coefficient. Check its dimension.");
  if (!(init.rows() == 22 && init.cols() == dim)) Rcpp::stop("`init` is 22 x `dim` matrix in order of y1, y2, ..., y22.");
  Eigen::MatrixXd hartrans_mat = scale_har(dim);
  Eigen::MatrixXd obs_p(1, var_design); // row vector of X0: y22^T, ..., y1^T, 1
  obs_p(0, var_design - 1) = 1.0; // for constant term
  for (int i = 0; i < 22; i++) {
    obs_p.block(0, i * dim, 1, dim) = init.row(21 - i);
  }
  Eigen::MatrixXd res(num_sim, dim); // Output: from y(23)^T to y(n + 22)^T
  Eigen::MatrixXd error_term = sim_mgaussian(num_sim, sig_error); // simulated error term: num_sim x m
  res.row(0) = obs_p * hartrans_mat.adjoint() * vhar_coef + error_term.row(0);
  for (int i = 1; i < num_sim; i++) {
    for (int t = 1; t < 22; t++) {
      obs_p.block(0, t * dim, 1, dim) = obs_p.block(0, (t - 1) * dim, 1, dim);
    }
    obs_p.block(0, 0, 1, dim) = res.row(i - 1);
    res.row(i) = obs_p * hartrans_mat.adjoint() * vhar_coef + error_term.row(i);
  }
  return res;
}
