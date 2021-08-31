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
Eigen::MatrixXd sim_var(int num_sim, Eigen::MatrixXd var_coef, int var_lag, Eigen::MatrixXd sig_error) {
  int dim = sig_error.cols(); // m: dimension of time series
  if (var_coef.cols() != dim) Rcpp::stop("Wrong VAR coefficient format or Variance matrix");
  Eigen::MatrixXd error_term = sim_mgaussian(num_sim, sig_error); // simulated error term: num_sim x m
  if (num_sim < 2) Rcpp::stop("Generate more than 1 series");
  Eigen::MatrixXd ma_coef = VARcoeftoVMA(var_coef, var_lag, num_sim - 1); // VMA representation up to VMA(num_sim - 1)
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(error_term.rows(), error_term.cols()); // Yt = sum(j = 0) Wj epsilon(t - j)
  res.row(0) = ma_coef.block(0, 0, dim, dim) * error_term.row(0).adjoint();
  for (int i = 1; i < num_sim; i++) {
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
Eigen::MatrixXd sim_vhar(int num_sim, Eigen::MatrixXd vhar_coef, Eigen::MatrixXd sig_error) {
  int dim = sig_error.cols(); // m: dimension of time series
  if (vhar_coef.cols() != dim) Rcpp::stop("Wrong VHAR coefficient format or Variance matrix");
  Eigen::MatrixXd hartrans_mat = scale_har(dim);
  Eigen::MatrixXd error_term = sim_mgaussian(num_sim, sig_error); // simulated error term: num_sim x m
  if (num_sim < 2) Rcpp::stop("Generate more than 1 series");
  Eigen::MatrixXd ma_coef = VHARcoeftoVMA(vhar_coef, hartrans_mat, num_sim - 1); // VMA representation up to VMA(num_sim - 1)
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(error_term.rows(), error_term.cols()); // Yt = sum(j = 0) Wj epsilon(t - j)
  res.row(0) = ma_coef.block(0, 0, dim, dim) * error_term.row(0).adjoint();
  for (int i = 1; i < num_sim; i++) {
    for (int j = 0; j <= i; j++) {
      res.row(i) += ma_coef.block(dim * (i - j), 0, dim, dim) * error_term.row(j).adjoint();
    }
  }
  return res;
}
