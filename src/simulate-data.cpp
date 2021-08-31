#include <RcppEigen.h>
#include "bvharmisc.h"

// [[Rcpp::depends(RcppEigen)]]

//' Generate Multivariate Time Series Process Following VAR(p)
//' 
//' This function generates multivariate time series dataset that follows VAR(p).
//' 
//' @param num_sim Number to generated process
//' @param var_coef VAR coefficient. The format should be the same as the output of \code{\link{var_lm}}
//' @param const_term constand term
//' @param var_lag Lag of VAR
//' @param sig_error Variance matrix of the error term
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
Eigen::MatrixXd sim_var(int num_sim, Eigen::MatrixXd var_coef, Eigen::VectorXd const_term, int var_lag, Eigen::MatrixXd sig_error) {
  int dim = sig_error.cols(); // m: dimension of time series
  if (var_coef.cols() != dim) Rcpp::stop("Wrong VAR coefficient format or Variance matrix");
  if (const_term.size() != dim) Rcpp::stop("Wrong `const_term` length");
  if (var_coef.rows() != (dim * var_lag) && var_coef.rows() != (dim * var_lag + 1)) Rcpp::stop("`var_coef` is not VAR coefficient. Check its dimension.");
  Eigen::MatrixXd mat_var(dim, dim);
  mat_var.setIdentity(dim, dim); // identity matrix
  for (int i = 0; i < var_lag; i++) {
    mat_var -= var_coef.block(dim * i, 0, dim, dim).adjoint();
  }
  Eigen::VectorXd sig_mean(dim); // mean of Yt
  sig_mean = mat_var.inverse() * const_term; // mu = (I - B1 - ... Bp)^{-1} * c
  Eigen::MatrixXd error_term = sim_mgaussian(num_sim, sig_error); // simulated error term: num_sim x m
  if (num_sim < 2) Rcpp::stop("Generate more than 1 series");
  Eigen::MatrixXd ma_coef = VARcoeftoVMA(var_coef, var_lag, num_sim - 1); // VMA representation up to VMA(num_sim - 1)
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(error_term.rows(), error_term.cols()); // Yt = sum(j = 0) Wj epsilon(t - j)
  res.row(0) = ma_coef.block(0, 0, dim, dim) * error_term.row(0).adjoint();
  for (int i = 1; i < num_sim; i++) {
    res.row(i) += sig_mean;
    for (int j = 0; j <= i; j++) {
      res.row(i) += ma_coef.block(dim * (i - j), 0, dim, dim) * error_term.row(j).adjoint();
    }
  }
  return res;
}

//' Generate Multivariate Time Series Process Following VHAR
//' 
//' This function generates multivariate time series dataset that follows VHAR.
//' 
//' @param num_sim Number to generated process
//' @param vhar_coef VHAR coefficient. The format should be the same as the output of \code{\link{vhar_lm}}
//' @param const_term constand term
//' @param sig_error Variance matrix of the error term
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
Eigen::MatrixXd sim_vhar(int num_sim, Eigen::MatrixXd vhar_coef, Eigen::VectorXd const_term, Eigen::MatrixXd sig_error) {
  int dim = sig_error.cols(); // m: dimension of time series
  if (vhar_coef.cols() != dim) Rcpp::stop("Wrong VHAR coefficient format or Variance matrix");
  if (const_term.size() != dim) Rcpp::stop("Wrong `const_term` length");
  if (vhar_coef.rows() != (3 * dim) && vhar_coef.rows() != (3 * dim + 1)) Rcpp::stop("`vhar_coef` is not VHAR coefficient. Check its dimension.");
  Eigen::MatrixXd mat_var(dim, dim);
  mat_var.setIdentity(dim, dim); // identity matrix
  for (int i = 0; i < 3; i++) {
    mat_var -= vhar_coef.block(dim * i, 0, dim, dim).adjoint();
  }
  Eigen::VectorXd sig_mean(dim);
  sig_mean = mat_var.inverse() * const_term;
  Eigen::MatrixXd hartrans_mat = scale_har(dim);
  Eigen::MatrixXd error_term = sim_mgaussian(num_sim, sig_error); // simulated error term: num_sim x m
  if (num_sim < 2) Rcpp::stop("Generate more than 1 series");
  Eigen::MatrixXd ma_coef = VHARcoeftoVMA(vhar_coef, hartrans_mat, num_sim - 1); // VMA representation up to VMA(num_sim - 1)
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(error_term.rows(), error_term.cols()); // Yt = sum(j = 0) Wj epsilon(t - j)
  res.row(0) = ma_coef.block(0, 0, dim, dim) * error_term.row(0).adjoint();
  for (int i = 1; i < num_sim; i++) {
    res.row(i) += sig_mean;
    for (int j = 0; j <= i; j++) {
      res.row(i) += ma_coef.block(dim * (i - j), 0, dim, dim) * error_term.row(j).adjoint();
    }
  }
  return res;
}
