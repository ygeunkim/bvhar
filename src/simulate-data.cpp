#include <bvharsim.h>
#include <bvhardesign.h>

//' Generate Multivariate Time Series Process Following VAR(p)
//' 
//' This function generates multivariate time series dataset that follows VAR(p).
//' 
//' @param num_sim Number to generated process
//' @param num_burn Number of burn-in
//' @param var_coef VAR coefficient. The format should be the same as the output of [coef.varlse()] from [var_lm()]
//' @param var_lag Lag of VAR
//' @param sig_error Variance matrix of the error term. Try `diag(dim)`.
//' @param init Initial y1, ..., yp matrix to simulate VAR model. Try `matrix(0L, nrow = var_lag, ncol = dim)`.
//' @param process Process type. 1: Gaussian. 2: student-t.
//' @param mvt_df DF of MVT
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_var_eigen(int num_sim, 
                              int num_burn, 
                              Eigen::MatrixXd var_coef, 
                              int var_lag, 
                              Eigen::MatrixXd sig_error, 
                              Eigen::MatrixXd init,
                              int process,
                              double mvt_df) {
  int dim = sig_error.cols(); // m: dimension of time series
  int dim_design = var_coef.rows(); // k = mp + 1 (const) or mp (none)
  int num_rand = num_sim + num_burn; // sim + burnin
  Eigen::MatrixXd obs_p(1, dim_design); // row vector of X0: yp^T, ..., y1^T, (1)
  obs_p(0, dim_design - 1) = 1.0; // for constant term if exists
  for (int i = 0; i < var_lag; i++) {
    obs_p.block(0, i * dim, 1, dim) = init.row(var_lag - i - 1);
  }
  Eigen::MatrixXd res(num_rand, dim); // Output: from y(p + 1)^T to y(n + p)^T
  // epsilon ~ N(0, sig_error)
  Eigen::VectorXd sig_mean = Eigen::VectorXd::Zero(dim); // zero mean
  // Eigen::MatrixXd error_term = sim_mgaussian(num_rand, sig_mean, sig_error); // simulated error term: num_rand x m
  Eigen::MatrixXd error_term(num_rand, dim);
  switch (process) {
  case 1:
    error_term = sim_mgaussian(num_rand, sig_mean, sig_error);
    break;
  case 2:
    error_term = sim_mstudent(num_rand, mvt_df, sig_mean, sig_error * (mvt_df - 2) / mvt_df, 1);
    break;
  default:
    Rcpp::stop("Invalid 'process' option.");
  }
  res.row(0) = obs_p * var_coef + error_term.row(0); // y(p + 1) = [yp^T, ..., y1^T, 1] A + eps(T)
  for (int i = 1; i < num_rand; i++) {
    for (int t = 1; t < var_lag; t++) {
      obs_p.block(0, t * dim, 1, dim) = obs_p.block(0, (t - 1) * dim, 1, dim);
    }
    obs_p.block(0, 0, 1, dim) = res.row(i - 1);
    res.row(i) = obs_p * var_coef + error_term.row(i); // yi = [y(i-1), ..., y(i-p), 1] A + eps(i)
  }
  return res.bottomRows(num_rand - num_burn);
}

//' Generate Multivariate Time Series Process Following VAR(p) using Cholesky Decomposition
//' 
//' This function generates VAR(p) using Cholesky Decomposition.
//' 
//' @param num_sim Number to generated process
//' @param num_burn Number of burn-in
//' @param var_coef VAR coefficient. The format should be the same as the output of [coef.varlse()] from [var_lm()]
//' @param var_lag Lag of VAR
//' @param sig_error Variance matrix of the error term. Try `diag(dim)`.
//' @param init Initial y1, ..., yp matrix to simulate VAR model. Try `matrix(0L, nrow = var_lag, ncol = dim)`.
//' @param process Process type. 1: Gaussian. 2: student-t.
//' @param mvt_df DF of MVT
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_var_chol(int num_sim, 
                             int num_burn, 
                             Eigen::MatrixXd var_coef, 
                             int var_lag, 
                             Eigen::MatrixXd sig_error, 
                             Eigen::MatrixXd init,
                             int process,
                             double mvt_df) {
  int dim = sig_error.cols();
  int dim_design = var_coef.rows();
  int num_rand = num_sim + num_burn;
  Eigen::MatrixXd obs_p(1, dim_design);
  obs_p(0, dim_design - 1) = 1.0;
  for (int i = 0; i < var_lag; i++) {
    obs_p.block(0, i * dim, 1, dim) = init.row(var_lag - i - 1);
  }
  Eigen::MatrixXd res(num_rand, dim);
  Eigen::VectorXd sig_mean = Eigen::VectorXd::Zero(dim);
  // Eigen::MatrixXd error_term = sim_mgaussian_chol(num_rand, sig_mean, sig_error); // normal using cholesky
  Eigen::MatrixXd error_term(num_rand, dim);
  switch (process) {
  case 1:
    error_term = sim_mgaussian_chol(num_rand, sig_mean, sig_error);
    break;
  case 2:
    error_term = sim_mstudent(num_rand, mvt_df, sig_mean, sig_error * (mvt_df - 2) / mvt_df, 2);
    break;
  default:
    Rcpp::stop("Invalid 'process' option.");
  }
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
//' @param vhar_coef VHAR coefficient. The format should be the same as the output of [coef.vharlse()] from [vhar_lm()]
//' @param week Order for weekly term. Try `5L` by default.
//' @param month Order for monthly term. Try `22L` by default.
//' @param sig_error Variance matrix of the error term. Try `diag(dim)`.
//' @param init Initial y1, ..., y_month matrix to simulate VHAR model. Try `matrix(0L, nrow = month, ncol = dim)`.
//' @param process Process type. 1: Gaussian. 2: student-t.
//' @param mvt_df DF of MVT
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_vhar_eigen(int num_sim, 
                               int num_burn, 
                               Eigen::MatrixXd vhar_coef, 
                               int week,
                               int month,
                               Eigen::MatrixXd sig_error, 
                               Eigen::MatrixXd init,
                               int process,
                               double mvt_df) {
  int dim = sig_error.cols(); // m: dimension of time series
  int num_har = vhar_coef.rows(); // 3m + 1 (const) or 3m (none)
  int dim_har = month * dim + 1; // 22m + 1 (const)
  bool include_mean = true;
  if (num_har == 3 * dim) {
    dim_har -= 1;
    include_mean = false;
  } // 22m (none)
  int num_rand = num_sim + num_burn; // sim + burnin
  Eigen::MatrixXd hartrans_mat = scale_har(dim, week, month, include_mean).block(0, 0, num_har, dim_har);
  Eigen::MatrixXd obs_p(1, dim_har); // row vector of X0: y22^T, ..., y1^T, 1
  obs_p(0, dim_har - 1) = 1.0; // for constant term if exists
  for (int i = 0; i < month; i++) {
    obs_p.block(0, i * dim, 1, dim) = init.row(month - 1 - i);
  }
  Eigen::MatrixXd res(num_rand, dim); // Output: from y(23)^T to y(n + 22)^T
  // epsilon ~ N(0, sig_error)
  Eigen::VectorXd sig_mean = Eigen::VectorXd::Zero(dim); // zero mean
  // Eigen::MatrixXd error_term = sim_mgaussian(num_rand, sig_mean, sig_error); // simulated error term: num_rand x m
  Eigen::MatrixXd error_term(num_rand, dim);
  switch (process) {
  case 1:
    error_term = sim_mgaussian(num_rand, sig_mean, sig_error);
    break;
  case 2:
    error_term = sim_mstudent(num_rand, mvt_df, sig_mean, sig_error * (mvt_df - 2) / mvt_df, 1);
    break;
  default:
    Rcpp::stop("Invalid 'process' option.");
  }
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

//' Generate Multivariate Time Series Process Following VHAR using Cholesky Decomposition
//' 
//' This function generates multivariate time series dataset that follows VHAR.
//' 
//' @param num_sim Number to generated process
//' @param num_burn Number of burn-in
//' @param vhar_coef VHAR coefficient. The format should be the same as the output of [coef.vharlse()] from [vhar_lm()]
//' @param week Order for weekly term. Try `5L` by default.
//' @param month Order for monthly term. Try `22L` by default.
//' @param sig_error Variance matrix of the error term. Try `diag(dim)`.
//' @param init Initial y1, ..., y_month matrix to simulate VHAR model. Try `matrix(0L, nrow = month, ncol = dim)`.
//' @param process Process type. 1: Gaussian. 2: student-t.
//' @param mvt_df DF of MVT
//' @details
//' Let \eqn{M} be the month order, e.g. \eqn{M = 22}.
//' 
//' 1. Generate \eqn{\epsilon_1, \epsilon_n \sim N(0, \Sigma)}
//' 2. For i = 1, ... n,
//' \deqn{y_{M + i} = (y_{M + i - 1}^T, \ldots, y_i^T, 1)^T C_{HAR}^T \Phi + \epsilon_i}
//' 3. Then the output is \eqn{(y_{M + 1}, \ldots, y_{n + M})^T}
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_vhar_chol(int num_sim, 
                              int num_burn, 
                              Eigen::MatrixXd vhar_coef, 
                              int week,
                              int month,
                              Eigen::MatrixXd sig_error, 
                              Eigen::MatrixXd init,
                              int process,
                              double mvt_df) {
  int dim = sig_error.cols(); // m: dimension of time series
  int num_har = vhar_coef.rows(); // 3m + 1 (const) or 3m (none)
  int dim_har = month * dim + 1; // 22m + 1 (const)
  bool include_mean = true;
  if (num_har == 3 * dim) {
    dim_har -= 1;
    include_mean = false;
  } // 22m (none)
  int num_rand = num_sim + num_burn; // sim + burnin
  Eigen::MatrixXd hartrans_mat = scale_har(dim, week, month, include_mean).block(0, 0, num_har, dim_har);
  Eigen::MatrixXd obs_p(1, dim_har); // row vector of X0: y22^T, ..., y1^T, 1
  obs_p(0, dim_har - 1) = 1.0; // for constant term if exists
  for (int i = 0; i < month; i++) {
    obs_p.block(0, i * dim, 1, dim) = init.row(month - 1 - i);
  }
  Eigen::MatrixXd res(num_rand, dim); // Output: from y(23)^T to y(n + 22)^T
  // epsilon ~ N(0, sig_error)
  Eigen::VectorXd sig_mean = Eigen::VectorXd::Zero(dim); // zero mean
  // Eigen::MatrixXd error_term = sim_mgaussian_chol(num_rand, sig_mean, sig_error); // simulated error term: num_rand x m
  Eigen::MatrixXd error_term(num_rand, dim);
  switch (process) {
  case 1:
    error_term = sim_mgaussian_chol(num_rand, sig_mean, sig_error);
    break;
  case 2:
    error_term = sim_mstudent(num_rand, mvt_df, sig_mean, sig_error * (mvt_df - 2) / mvt_df, 2);
    break;
  default:
    Rcpp::stop("Invalid 'process' option.");
  }
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