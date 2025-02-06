// #include "bvharsim.h"
#include <bvhar/utils>

// [[Rcpp::export]]
int get_maxomp() {
	return omp_get_max_threads();
}

// [[Rcpp::export]]
void check_omp() {
#ifdef _OPENMP
  Rcpp::Rcout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
	Rcpp::Rcout << "OpenMP not available in this machine." << "\n";
#endif
}

// [[Rcpp::export]]
bool is_omp() {
#ifdef _OPENMP
  return true;
#else
	return false;
#endif
}

//' Build Response Matrix of VAR(p)
//' 
//' This function constructs response matrix of multivariate regression model formulation of VAR(p).
//' 
//' @param y Matrix, multivariate time series data.
//' @param var_lag Integer, VAR lag.
//' @param index Integer, Starting index to extract
//' 
//' @details
//' Let s = n - p.
//' \deqn{Y_j = (y_j, y_{j + 1}, \ldots, y_{j + s - 1})^T}
//' is the s x m matrix.
//' 
//' In case of response matrix, t = p + 1 (i.e. \eqn{Y_0 = Y_{p + 1}}).
//' This function is also used when constructing design matrix.
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd build_response(Eigen::MatrixXd y, int var_lag, int index) {
	return bvhar::build_y0(y, var_lag, index);
}

//' Build Design Matrix of VAR(p)
//' 
//' This function constructs design matrix of multivariate regression model formulation of VAR(p).
//' 
//' @param y Matrix, time series data
//' @param var_lag VAR lag
//' @param include_mean bool, Add constant term (Default: `true`) or not (`false`)
//' 
//' @details
//' X0 is
//' \deqn{X_0 = [Y_p, \ldots, Y_1, 1]}
//' i.e. (n - p) x (mp + 1) matrix
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd build_design(Eigen::MatrixXd y, int var_lag, bool include_mean) {
	return bvhar::build_x0(y, var_lag, include_mean);
}

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
	return bvhar::build_vhar(dim, week, month, include_mean);
}

//' Construct Dummy response for Minnesota Prior
//' 
//' Define dummy Y observations to add for Minnesota moments.
//' 
//' @param p Integer, VAR lag. For VHAR, put 3.
//' @param sigma Vector, standard error of each variable
//' @param lambda Double, tightness of the prior around a random walk or white noise
//' @param daily Vector, prior belief about white noise (Litterman sets 1)
//' @param weekly Vector, this was zero in the original Minnesota design
//' @param monthly Vector, this was zero in the original Minnesota design
//' @param include_mean bool, Add constant term (Default: `true`) or not (`false`)
//' 
//' @details
//' Bańbura et al. (2010) defines dummy observation and augment to the original data matrix to construct Litterman (1986) prior.
//' 
//' @references
//' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
//' 
//' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd build_ydummy_export(int p, Eigen::VectorXd sigma, double lambda, Eigen::VectorXd daily, Eigen::VectorXd weekly, Eigen::VectorXd monthly, bool include_mean) {
	return bvhar::build_ydummy(p, sigma, lambda, daily, weekly, monthly, include_mean);
}

//' Construct Dummy design matrix for Minnesota Prior
//' 
//' Define dummy X observation to add for Minnesota moments.
//' 
//' @param lag_seq Vector, sequence to build Jp = diag(1, ... p) matrix inside Xp.
//' @param sigma Vector, standard error of each variable
//' @param lambda Double, tightness of the prior around a random walk or white noise
//' @param eps Double, very small number
//' 
//' @details
//' Bańbura et al. (2010) defines dummy observation and augment to the original data matrix to construct Litterman (1986) prior.
//' 
//' @references
//' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
//' 
//' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd build_xdummy_export(Eigen::VectorXd lag_seq, double lambda, Eigen::VectorXd sigma, double eps, bool include_mean) {
	return bvhar::build_xdummy(lag_seq, lambda, sigma, eps, include_mean);
}

//' Parameters of Normal Inverted Wishart Prior
//' 
//' Given dummy observations, compute parameters of Normal-IW prior for Minnesota.
//' 
//' @param x_dummy Matrix, dummy observation for X0
//' @param y_dummy Matrix, dummy observation for Y0
//' 
//' @details
//' Minnesota prior give prior to parameters \eqn{B} (VAR matrices) and \eqn{\Sigma_e} (residual covariance) the following distributions
//' 
//' \deqn{B \mid \Sigma_e, Y_0 \sim MN(B_0, \Omega_0, \Sigma_e)}
//' \deqn{\Sigma_e \mid Y_0 \sim IW(S_0, \alpha_0)}
//' (MN: [matrix normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution), IW: [inverse-wishart](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution))
//' 
//' Bańbura et al. (2010) provides the formula how to find each matrix to match Minnesota moments.
//' 
//' @references
//' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
//' 
//' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List minnesota_prior(Eigen::MatrixXd x_dummy, Eigen::MatrixXd y_dummy) {
  int dim = y_dummy.cols(); // m
  int dim_design = x_dummy.cols(); // k
  Eigen::MatrixXd prior_mean(dim_design, dim); // prior mn mean
  Eigen::MatrixXd prior_prec(dim_design, dim_design); // prior mn precision
  Eigen::MatrixXd prior_scale(dim, dim); // prior iw scale
  int prior_shape = x_dummy.rows() - dim_design + 2;
  prior_prec = x_dummy.transpose() * x_dummy;
  prior_mean = prior_prec.inverse() * x_dummy.transpose() * y_dummy;
  prior_scale = (y_dummy - x_dummy * prior_mean).transpose() * (y_dummy - x_dummy * prior_mean);
  return Rcpp::List::create(
    Rcpp::Named("prior_mean") = prior_mean,
    Rcpp::Named("prior_prec") = prior_prec,
    Rcpp::Named("prior_scale") = prior_scale,
    Rcpp::Named("prior_shape") = prior_shape
  );
}

//' Generate Multivariate Normal Random Vector
//' 
//' This function samples n x muti-dimensional normal random matrix.
//' 
//' @param num_sim Number to generate process
//' @param mu Mean vector
//' @param sig Variance matrix
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_mgaussian(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig) {
  int dim = sig.cols();
  if (sig.rows() != dim) {
    Rcpp::stop("Invalid 'sig' dimension.");
  }
  if (dim != mu.size()) {
    Rcpp::stop("Invalid 'mu' size.");
  }
  Eigen::MatrixXd standard_normal(num_sim, dim);
  Eigen::MatrixXd res(num_sim, dim); // result: each column indicates variable
  for (int i = 0; i < num_sim; i++) {
    for (int j = 0; j < standard_normal.cols(); j++) {
      standard_normal(i, j) = norm_rand();
    }
  }
  res = standard_normal * sig.sqrt(); // epsilon(t) = Sigma^{1/2} Z(t)
  res.rowwise() += mu.transpose();
  return res;
}

//' Generate Multivariate Normal Random Vector using Cholesky Decomposition
//' 
//' This function samples n x muti-dimensional normal random matrix with using Cholesky decomposition.
//' 
//' @param num_sim Number to generate process
//' @param mu Mean vector
//' @param sig Variance matrix
//' @details
//' This function computes \eqn{\Sigma^{1/2}} by choleksy decomposition.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_mgaussian_chol_export(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig) {
	int dim = sig.cols();
	if (sig.rows() != dim) {
    Rcpp::stop("Invalid 'sig' dimension.");
  }
  if (dim != mu.size()) {
    Rcpp::stop("Invalid 'mu' size.");
  }
  return bvhar::sim_mgaussian_chol(num_sim, mu, sig);
}

//' Generate Multivariate t Random Vector
//' 
//' This function samples n x muti-dimensional normal random matrix.
//' 
//' @param num_sim Number to generate process
//' @param df Degrees of freedom
//' @param mu Location vector
//' @param sig Scale matrix
//' @param method Method to compute \eqn{\Sigma^{1/2}}. 1: spectral decomposition, 2: Cholesky.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_mstudent(int num_sim, double df, Eigen::VectorXd mu, Eigen::MatrixXd sig, int method) {
  int dim = sig.cols();
  if (sig.rows() != dim) {
    Rcpp::stop("Invalid 'sig' dimension.");
  }
  if (dim != mu.size()) {
    Rcpp::stop("Invalid 'mu' size.");
  }
  Eigen::MatrixXd res(num_sim, dim);
  switch (method) {
  case 1:
    res = sim_mgaussian(num_sim, Eigen::VectorXd::Zero(dim), sig);
    break;
  case 2:
    res = bvhar::sim_mgaussian_chol(num_sim, Eigen::VectorXd::Zero(dim), sig);
    break;
  default:
    Rcpp::stop("Invalid 'method' option.");
  }
  for (int i = 0; i < num_sim; i++) {
    res.row(i) *= sqrt(df / bvhar::chisq_rand(df));
  }
  res.rowwise() += mu.transpose();
  return res;
}

//' Generate Matrix Normal Random Matrix
//' 
//' This function samples one matrix gaussian matrix.
//' 
//' @param mat_mean Mean matrix
//' @param mat_scale_u First scale matrix
//' @param mat_scale_v Second scale matrix
//' @param u_prec If `TRUE`, use `mat_scale_u` as its inverse.
//' @details
//' Consider n x k matrix \eqn{Y_1, \ldots, Y_n \sim MN(M, U, V)} where M is n x k, U is n x n, and V is k x k.
//' 
//' 1. Lower triangular Cholesky decomposition: \eqn{U = P P^T} and \eqn{V = L L^T}
//' 2. Standard normal generation: s x m matrix \eqn{Z_i = [z_{ij} \sim N(0, 1)]} in row-wise direction.
//' 3. \eqn{Y_i = M + P Z_i L^T}
//' 
//' This function only generates one matrix, i.e. \eqn{Y_1}.
//' @return One n x k matrix following MN distribution.
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_matgaussian(Eigen::MatrixXd mat_mean, Eigen::MatrixXd mat_scale_u, Eigen::MatrixXd mat_scale_v, bool u_prec) {
  if (mat_scale_u.rows() != mat_scale_u.cols()) {
    Rcpp::stop("Invalid 'mat_scale_u' dimension.");
  }
  if (mat_mean.rows() != mat_scale_u.rows()) {
    Rcpp::stop("Invalid 'mat_scale_u' dimension.");
  }
  if (mat_scale_v.rows() != mat_scale_v.cols()) {
    Rcpp::stop("Invalid 'mat_scale_v' dimension.");
  }
  if (mat_mean.cols() != mat_scale_v.rows()) {
    Rcpp::stop("Invalid 'mat_scale_v' dimension.");
  }
	return bvhar::sim_mn(mat_mean, mat_scale_u, mat_scale_v, u_prec);
}

//' Generate Inverse-Wishart Random Matrix
//' 
//' This function samples one matrix IW matrix.
//' 
//' @param mat_scale Scale matrix
//' @param shape Shape
//' @details
//' Consider \eqn{\Sigma \sim IW(\Psi, \nu)}.
//' 
//' 1. Upper triangular Bartlett decomposition: k x k matrix \eqn{Q = [q_{ij}]} upper triangular with
//'     1. \eqn{q_{ii}^2 \chi_{\nu - i + 1}^2}
//'     2. \eqn{q_{ij} \sim N(0, 1)} with i < j (upper triangular)
//' 2. Lower triangular Cholesky decomposition: \eqn{\Psi = L L^T}
//' 3. \eqn{A = L (Q^{-1})^T}
//' 4. \eqn{\Sigma = A A^T \sim IW(\Psi, \nu)}
//' @return One k x k matrix following IW distribution
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_iw(Eigen::MatrixXd mat_scale, double shape) {
  // Eigen::MatrixXd chol_res = bvhar::sim_iw_tri(mat_scale, shape);
  // Eigen::MatrixXd res = chol_res * chol_res.transpose(); // dim x dim
  // return res;
	return bvhar::sim_inv_wishart(mat_scale, shape);
}

//' Generate Normal-IW Random Family
//' 
//' This function samples normal inverse-wishart matrices.
//' 
//' @param num_sim Number to generate
//' @param mat_mean Mean matrix of MN
//' @param mat_scale_u First scale matrix of MN
//' @param mat_scale Scale matrix of IW
//' @param shape Shape of IW
//' @param prec If true, use mat_scale_u as its inverse
//' @noRd
// [[Rcpp::export]]
Rcpp::List sim_mniw_export(int num_sim, Eigen::MatrixXd mat_mean, Eigen::MatrixXd mat_scale_u, Eigen::MatrixXd mat_scale, double shape, bool prec) {
	std::vector<std::vector<Eigen::MatrixXd>> res(num_sim, std::vector<Eigen::MatrixXd>(2));
	for (int i = 0; i < num_sim; i++) {
		res[i] = bvhar::sim_mn_iw(mat_mean, mat_scale_u, mat_scale, shape, prec);
  }
	return Rcpp::wrap(res);
}

//' Generate Multivariate Time Series Process Following VAR(p)
//' 
//' This function generates multivariate time series dataset that follows VAR(p).
//' 
//' @param num_sim Number to generated process
//' @param num_burn Number of burn-in
//' @param var_coef VAR coefficient. The format should be the same as the output of [coef()] from [var_lm()]
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
//' @param var_coef VAR coefficient. The format should be the same as the output of [coef()] from [var_lm()]
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
  // Eigen::MatrixXd error_term = bvhar::sim_mgaussian_chol(num_rand, sig_mean, sig_error); // normal using cholesky
  Eigen::MatrixXd error_term(num_rand, dim);
  switch (process) {
  case 1:
    error_term = bvhar::sim_mgaussian_chol(num_rand, sig_mean, sig_error);
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
  Eigen::MatrixXd hartrans_mat = bvhar::build_vhar(dim, week, month, include_mean).block(0, 0, num_har, dim_har);
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
  Eigen::MatrixXd hartrans_mat = bvhar::build_vhar(dim, week, month, include_mean).block(0, 0, num_har, dim_har);
  Eigen::MatrixXd obs_p(1, dim_har); // row vector of X0: y22^T, ..., y1^T, 1
  obs_p(0, dim_har - 1) = 1.0; // for constant term if exists
  for (int i = 0; i < month; i++) {
    obs_p.block(0, i * dim, 1, dim) = init.row(month - 1 - i);
  }
  Eigen::MatrixXd res(num_rand, dim); // Output: from y(23)^T to y(n + 22)^T
  // epsilon ~ N(0, sig_error)
  Eigen::VectorXd sig_mean = Eigen::VectorXd::Zero(dim); // zero mean
  // Eigen::MatrixXd error_term = bvhar::sim_mgaussian_chol(num_rand, sig_mean, sig_error); // simulated error term: num_rand x m
  Eigen::MatrixXd error_term(num_rand, dim);
  switch (process) {
  case 1:
    error_term = bvhar::sim_mgaussian_chol(num_rand, sig_mean, sig_error);
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

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd VARcoeftoVMA(Eigen::MatrixXd var_coef, int var_lag, int lag_max) {
  return bvhar::convert_var_to_vma(var_coef, var_lag, lag_max);
}

//' Convert VAR to VMA(infinite)
//' 
//' Convert VAR process to infinite vector MA process
//' 
//' @param object A `varlse` object
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
//' @return VMA coefficient of k(lag-max + 1) x k dimension
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd VARtoVMA(Rcpp::List object, int lag_max) {
  if (!object.inherits("varlse")) {
    Rcpp::stop("'object' must be varlse object.");
  }
  Eigen::MatrixXd coef_mat = object["coefficients"]; // bhat(k, m) = [B1^T, B2^T, ..., Bp^T, c^T]^T
  int var_lag = object["p"];
  Eigen::MatrixXd ma = bvhar::convert_var_to_vma(coef_mat, var_lag, lag_max);
  return ma;
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_var_mse(Eigen::MatrixXd cov_mat, Eigen::MatrixXd var_coef, int var_lag, int step) {
  int dim = cov_mat.cols(); // dimension of time series
  Eigen::MatrixXd vma_mat = bvhar::convert_var_to_vma(var_coef, var_lag, step);
  Eigen::MatrixXd innov_account = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::MatrixXd mse = Eigen::MatrixXd::Zero(dim * step, dim);
  for (int i = 0; i < step; i++) {
    innov_account += vma_mat.block(i * dim, 0, dim, dim).transpose() * cov_mat * vma_mat.block(i * dim, 0, dim, dim);
    mse.block(i * dim, 0, dim, dim) = innov_account;
  }
  return mse;
}

//' Compute Forecast MSE Matrices
//' 
//' Compute the forecast MSE matrices using VMA coefficients
//' 
//' @param object A `varlse` object
//' @param step Integer, Step to forecast
//' @details
//' See pp38 of Lütkepohl (2007).
//' Let \eqn{\Sigma} be the covariance matrix of VAR and let \eqn{W_j} be the VMA coefficients.
//' Recursively,
//' \deqn{\Sigma_y(1) = \Sigma}
//' \deqn{\Sigma_y(2) = \Sigma + W_1 \Sigma W_1^T}
//' \deqn{\Sigma_y(3) = \Sigma_y(2) + W_2 \Sigma W_2^T}
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_covmse(Rcpp::List object, int step) {
  if (!object.inherits("varlse")) {
    Rcpp::stop("'object' must be varlse object.");
  }
  return compute_var_mse(object["covmat"], object["coefficients"], object["p"], step);
}

//' Convert VAR to Orthogonalized VMA(infinite)
//' 
//' Convert VAR process to infinite orthogonalized vector MA process
//' 
//' @param var_coef VAR coefficient matrix
//' @param var_covmat VAR covariance matrix
//' @param var_lag VAR order
//' @param lag_max Maximum lag for VMA
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd VARcoeftoVMA_ortho(Eigen::MatrixXd var_coef, Eigen::MatrixXd var_covmat, int var_lag, int lag_max) {
  return bvhar::convert_vma_ortho(var_coef, var_covmat, var_lag, lag_max);
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd VHARcoeftoVMA(Eigen::MatrixXd vhar_coef, Eigen::MatrixXd HARtrans_mat, int lag_max, int month) {
  return bvhar::convert_vhar_to_vma(vhar_coef, HARtrans_mat, lag_max, month);
}

//' Convert VHAR to VMA(infinite)
//' 
//' Convert VHAR process to infinite vector MA process
//' 
//' @param object A `vharlse` object
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
//' @return VMA coefficient of k(lag-max + 1) x k dimension
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd VHARtoVMA(Rcpp::List object, int lag_max) {
  if (!object.inherits("vharlse")) {
    Rcpp::stop("'object' must be vharlse object.");
  }
  Eigen::MatrixXd har_mat = object["coefficients"]; // Phihat(3m + 1, m) = [Phi(d)^T, Phi(w)^T, Phi(m)^T, c^T]^T
  Eigen::MatrixXd hartrans_mat = object["HARtrans"]; // tilde(T): (3m + 1, 22m + 1)
  int month = object["month"];
  Eigen::MatrixXd ma = bvhar::convert_vhar_to_vma(har_mat, hartrans_mat, lag_max, month);
  return ma;
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_vhar_mse(Eigen::MatrixXd cov_mat,
                                 Eigen::MatrixXd vhar_coef,
                                 Eigen::MatrixXd har_trans,
                                 int month,
                                 int step) {
  int dim = cov_mat.cols(); // dimension of time series
  Eigen::MatrixXd vma_mat = bvhar::convert_vhar_to_vma(vhar_coef, har_trans, month, step);
  Eigen::MatrixXd mse(dim * step, dim);
  mse.block(0, 0, dim, dim) = cov_mat; // sig(y) = sig
  for (int i = 1; i < step; i++) {
    mse.block(i * dim, 0, dim, dim) = mse.block((i - 1) * dim, 0, dim, dim) + 
      vma_mat.block(i * dim, 0, dim, dim).transpose() * cov_mat * vma_mat.block(i * dim, 0, dim, dim);
  }
  return mse;
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
  return compute_vhar_mse(
    object["covmat"],
    object["coefficients"],
    object["HARtrans"],
    object["month"],
    step
  );
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
  return bvhar::convert_vhar_vma_ortho(vhar_coef, vhar_covmat, HARtrans_mat, lag_max, month);
}

//' h-step ahead Forecast Error Variance Decomposition
//' 
//' [w_(h = 1, ij)^T, w_(h = 2, ij)^T, ...]
//'
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_fevd(Eigen::MatrixXd vma_coef, Eigen::MatrixXd cov_mat, bool normalize) {
  return bvhar::compute_vma_fevd(vma_coef, cov_mat, normalize);
}

//' h-step ahead Normalized Spillover
//'
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_spillover(Eigen::MatrixXd fevd) {
  return fevd.bottomRows(fevd.cols()) * 100;
}

//' To-others Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd compute_to_spillover(Eigen::MatrixXd spillover) {
  return bvhar::compute_to(spillover);
}

//' From-others Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd compute_from_spillover(Eigen::MatrixXd spillover) {
  return bvhar::compute_from(spillover);
}

//' Total Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
double compute_tot_spillover(Eigen::MatrixXd spillover) {
  return bvhar::compute_tot(spillover);
}

//' Net Pairwise Spillovers
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_net_spillover(Eigen::MatrixXd spillover) {
  return bvhar::compute_net(spillover);
}

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
//'       A_1 & A_2 & \cdots A_{p - 1} & A_p \\
//'       I_m & 0 & \cdots & 0 & 0 \\
//'       0 & I_m & \cdots & 0 & 0 \\
//'       \vdots & \vdots & \vdots & \vdots & \vdots \\
//'       0 & 0 & \cdots & I_m & 0
//'     \end{bmatrix}
//' }
//' 
//' \deqn{C = (c, 0, \ldots, 0)^T}
//' 
//' and
//' 
//' \deqn{U_t = (\epsilon_t^T, 0^T, \ldots, 0^T)^T}
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_stablemat(Eigen::MatrixXd x) {
  return bvhar::build_companion(x);
}

//' VAR(1) Representation of VAR(p)
//' 
//' Compute the coefficient matrix of VAR(1) form
//' 
//' @param object Model fit
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
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
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
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

//' Log of Multivariate Gamma Function
//' 
//' Compute log of multivariate gamma function numerically
//' 
//' @param x Double, non-negative argument
//' @param p Integer, dimension
//' @noRd
// [[Rcpp::export]]
double log_mgammafn(double x, int p) {
  if (p < 1) {
    Rcpp::stop("'p' should be larger than or same as 1.");
  }
  if (x <= 0) {
    Rcpp::stop("'x' should be larger than 0.");
  }
  if (p == 1) {
    return bvhar::lgammafn(x);
  }
  if (2 * x < p) {
    Rcpp::stop("'x / 2' should be larger than 'p'.");
  }
  return bvhar::lmgammafn(x, p);
}

//' Numerically Stable Log ML Excluding Constant Term of BVAR and BVHAR
//' 
//' This function computes log of ML stable,
//' in purpose of objective function.
//' 
//' @param object Bayesian Model Fit
//' 
//' @noRd
// [[Rcpp::export]]
double logml_stable(Rcpp::List object) {
  if (!object.inherits("bvarmn") && !object.inherits("bvharmn")) {
    Rcpp::stop("'object' must be bvarmn or bvharmn object.");
  }
  return bvhar::compute_logml(object["m"], object["obs"], object["prior_precision"], object["prior_scale"], object["mn_prec"], object["covmat"], object["iw_shape"]);
}

//' AIC of VAR(p) using RSS
//' 
//' Compute AIC using RSS
//' 
//' @param object A `varlse` or `vharlse` object
//' 
//' @noRd
// [[Rcpp::export]]
double compute_aic(Rcpp::List object) {
  if (!object.inherits("varlse") && !object.inherits("vharlse")) {
    Rcpp::stop("'object' must be varlse or vharlse object.");
  }
  double dim = object["m"]; // m
  double dim_design = object["df"]; // k
  double num_design = object["obs"]; // s
  Eigen::MatrixXd cov_lse = object["covmat"]; // crossprod(COV) / (s - k)
  double sig_det = cov_lse.determinant() * pow((num_design - dim_design) / num_design, dim); // det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  // penalty = (2 / s) * number of freely estimated parameters
  return log(sig_det) + 2 / num_design * dim * dim_design;
}

//' BIC of VAR(p) using RSS
//' 
//' Compute BIC using RSS
//' 
//' @param object A `varlse` or `vharlse` object
//' 
//' @noRd
// [[Rcpp::export]]
double compute_bic(Rcpp::List object) {
  if (!object.inherits("varlse") && !object.inherits("vharlse")) {
    Rcpp::stop("'object' must be varlse or vharlse object.");
  }
  double dim = object["m"]; // m
  double dim_design = object["df"]; // k
  double num_design = object["obs"]; // s
  Eigen::MatrixXd cov_lse = object["covmat"]; // crossprod(COV) / (s - k)
  double sig_det = cov_lse.determinant() * pow((num_design - dim_design) / num_design, dim); // det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  // penalty = replace 2 / s with log(s) / s
  return log(sig_det) + log(num_design) / num_design * dim * dim_design;
}

//' HQ of VAR(p) using RSS
//' 
//' Compute HQ using RSS
//' 
//' @param object A `varlse` or `vharlse` object
//' 
//' @noRd
// [[Rcpp::export]]
double compute_hq(Rcpp::List object) {
  if (!object.inherits("varlse") && !object.inherits("vharlse")) {
    Rcpp::stop("'object' must be varlse or vharlse object.");
  }
  double dim = object["m"]; // m
  double dim_design = object["df"]; // k
  double num_design = object["obs"]; // s
  Eigen::MatrixXd cov_lse = object["covmat"]; // crossprod(COV) / (s - k)
  double sig_det = cov_lse.determinant() * pow((num_design - dim_design) / num_design, dim); // det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  // penalty = replace log(s) / s with 2 * log(log(s)) / s
  return log(sig_det) + 2 * log(log(num_design)) / num_design * dim * dim_design;
}

//' FPE of VAR(p) using RSS
//' 
//' Compute FPE using RSS
//' 
//' @param object A `varlse` or `vharlse` object
//' 
//' @noRd
// [[Rcpp::export]]
double compute_fpe(Rcpp::List object) {
  if (!object.inherits("varlse") && !object.inherits("vharlse")) {
    Rcpp::stop("'object' must be varlse or vharlse object.");
  }
  double dim = object["m"]; // m
  double dim_design = object["df"]; // k
  double num_design = object["obs"]; // s
  Eigen::MatrixXd cov_lse = object["covmat"]; // crossprod(COV) / (s - k)
  // FPE = ((s + k) / (s - k))^m * det = ((s + k) / s)^m * det(crossprod(resid) / (s - k))
  return pow((num_design + dim_design) / num_design, dim) * cov_lse.determinant();
}

//' Choose the Best VAR based on Information Criteria
//' 
//' This function computes AIC, FPE, BIC, and HQ up to p = `lag_max` of VAR model.
//' 
//' @param y Time series data of which columns indicate the variables
//' @param lag_max Maximum Var lag to explore
//' @param include_mean Add constant term
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd tune_var(Eigen::MatrixXd y, int lag_max, bool include_mean) {
  Rcpp::Function fit("var_lm");
  Eigen::MatrixXd ic_res(lag_max, 4); // matrix including information criteria: AIC-BIC-HQ-FPE
  Rcpp::List var_mod;
  for (int i = 0; i < lag_max; i++) {
    var_mod = fit(y, i + 1, include_mean);
    ic_res(i, 0) = compute_aic(var_mod);
    ic_res(i, 1) = compute_bic(var_mod);
    ic_res(i, 2) = compute_hq(var_mod);
    ic_res(i, 3) = compute_fpe(var_mod);
  }
  return ic_res;
}
