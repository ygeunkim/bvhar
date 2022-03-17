#include <RcppEigen.h>
#include "bvharmisc.h"

// [[Rcpp::depends(RcppEigen)]]

//' Generate Multivariate Time Series Process Following VAR(p)
//' 
//' This function generates multivariate time series dataset that follows VAR(p).
//' 
//' @param num_sim Number to generated process
//' @param num_burn Number of burn-in
//' @param var_coef VAR coefficient. The format should be the same as the output of [var_lm()]
//' @param var_lag Lag of VAR
//' @param sig_error Variance matrix of the error term. Try `diag(dim)`.
//' @param init Initial y1, ..., yp matrix to simulate VAR model. Try `matrix(0L, nrow = var_lag, ncol = dim)`.
//' @details
//' 1. Generate \eqn{\epsilon_1, \epsilon_n \sim N(0, \Sigma)}
//' 2. For i = 1, ... n,
//' \deqn{y_{p + i} = (y_{p + i - 1}^T, \ldots, y_i^T, 1)^T B + \epsilon_i}
//' 3. Then the output is \eqn{(y_{p + 1}, \ldots, y_{n + p})^T}
//' 
//' Initial values might be set to be zero vector or \eqn{(I_m - A_1 - \cdots - A_p)^{-1} c}.
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_var(int num_sim, 
                        int num_burn, 
                        Eigen::MatrixXd var_coef, 
                        int var_lag, 
                        Eigen::MatrixXd sig_error, 
                        Eigen::MatrixXd init) {
  int dim = sig_error.cols(); // m: dimension of time series
  if (num_sim < 2) {
    Rcpp::stop("Generate more than 1 series");
  }
  if (var_coef.rows() != dim * var_lag + 1 && var_coef.rows() != dim * var_lag) {
    Rcpp::stop("'var_coef' is not VAR coefficient. Check its dimension.");
  }
  int dim_design = var_coef.rows(); // k = mp + 1 (const) or mp (none)
  if (var_coef.cols() != dim) {
    Rcpp::stop("Wrong VAR coefficient format or Variance matrix");
  }
  if (!(init.rows() == var_lag && init.cols() == dim)) {
    Rcpp::stop("'init' is (var_lag, dim) matrix in order of y1, y2, ..., yp.");
  }
  int num_rand = num_sim + num_burn; // sim + burnin
  Eigen::MatrixXd obs_p(1, dim_design); // row vector of X0: yp^T, ..., y1^T, (1)
  obs_p(0, dim_design - 1) = 1.0; // for constant term if exists
  for (int i = 0; i < var_lag; i++) {
    obs_p.block(0, i * dim, 1, dim) = init.row(var_lag - i - 1);
  }
  Eigen::MatrixXd res(num_rand, dim); // Output: from y(p + 1)^T to y(n + p)^T
  // epsilon ~ N(0, sig_error)
  Eigen::VectorXd sig_mean = Eigen::VectorXd::Zero(dim); // zero mean
  Eigen::MatrixXd error_term = sim_mgaussian(num_rand, sig_mean, sig_error); // simulated error term: num_rand x m
  res.row(0) = obs_p * var_coef + error_term.row(0);
  for (int i = 1; i < num_rand; i++) {
    for (int t = 1; t < var_lag; t++) {
      obs_p.block(0, t * dim, 1, dim) = obs_p.block(0, (t - 1) * dim, 1, dim);
    }
    obs_p.block(0, 0, 1, dim) = res.row(i - 1);
    res.row(i) = obs_p * var_coef + error_term.row(i);
  }
  return res.bottomRows(num_rand - num_burn);
}

//' Generate Multivariate Time Series Process Following VHAR
//' 
//' This function generates multivariate time series dataset that follows VHAR.
//' 
//' @param num_sim Number to generated process
//' @param num_burn Number of burn-in
//' @param vhar_coef VHAR coefficient. The format should be the same as the output of [vhar_lm()]
//' @param week Order for weekly term. Try `5` by default.
//' @param month Order for monthly term. Try `22` by default.
//' @param sig_error Variance matrix of the error term. Try `diag(dim)`.
//' @param init Initial y1, ..., yp matrix to simulate VAR model. Try `matrix(0L, nrow = month, ncol = dim)`.
//' @details
//' 1. Generate \eqn{\epsilon_1, \epsilon_n \sim N(0, \Sigma)}
//' 2. For i = 1, ... n,
//' \deqn{y_{22 + i} = (y_{21 + i}^T, \ldots, y_i^T, 1)^T T_{HAR}^T \Phi + \epsilon_i}
//' 3. Then the output is \eqn{(y_{23}, \ldots, y_{n + 22})^T}
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_vhar(int num_sim, 
                         int num_burn, 
                         Eigen::MatrixXd vhar_coef, 
                         int week,
                         int month,
                         Eigen::MatrixXd sig_error, 
                         Eigen::MatrixXd init) {
  int dim = sig_error.cols(); // m: dimension of time series
  if (num_sim < 2) {
    Rcpp::stop("Generate more than 1 series");
  }
  if (vhar_coef.rows() != 3 * dim + 1 && vhar_coef.rows() != 3 * dim) {
    Rcpp::stop("'vhar_coef' is not VHAR coefficient. Check its dimension.");
  }
  int num_har = vhar_coef.rows(); // 3m + 1 (const) or 3m (none)
  int dim_har = month * dim + 1; // 22m + 1 (const)
  if (num_har == 3 * dim) dim_har -= 1; // 22m (none)
  if (vhar_coef.cols() != dim) {
    Rcpp::stop("Wrong VHAR coefficient format or Variance matrix");
  }
  if (!(init.rows() == month && init.cols() == dim)) {
    Rcpp::stop("'init' is (22, dim) matrix in order of y1, y2, ..., y22.");
  }
  int num_rand = num_sim + num_burn; // sim + burnin
  Eigen::MatrixXd hartrans_mat = scale_har(dim, week, month).block(0, 0, num_har, dim_har);
  Eigen::MatrixXd obs_p(1, dim_har); // row vector of X0: y22^T, ..., y1^T, 1
  obs_p(0, dim_har - 1) = 1.0; // for constant term if exists
  for (int i = 0; i < month; i++) {
    obs_p.block(0, i * dim, 1, dim) = init.row(month - 1 - i);
  }
  Eigen::MatrixXd res(num_rand, dim); // Output: from y(23)^T to y(n + 22)^T
  // epsilon ~ N(0, sig_error)
  Eigen::VectorXd sig_mean = Eigen::VectorXd::Zero(dim); // zero mean
  Eigen::MatrixXd error_term = sim_mgaussian(num_rand, sig_mean, sig_error); // simulated error term: num_rand x m
  res.row(0) = obs_p * hartrans_mat.transpose() * vhar_coef + error_term.row(0);
  for (int i = 1; i < num_rand; i++) {
    for (int t = 1; t < month; t++) {
      obs_p.block(0, t * dim, 1, dim) = obs_p.block(0, (t - 1) * dim, 1, dim);
    }
    obs_p.block(0, 0, 1, dim) = res.row(i - 1);
    res.row(i) = obs_p * hartrans_mat.transpose() * vhar_coef + error_term.row(i);
  }
  return res.bottomRows(num_rand - num_burn);
}
