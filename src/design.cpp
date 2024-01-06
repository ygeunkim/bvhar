#include <RcppEigen.h>

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
Eigen::MatrixXd build_ydummy(int p, Eigen::VectorXd sigma, double lambda, Eigen::VectorXd daily, Eigen::VectorXd weekly, Eigen::VectorXd monthly, bool include_mean) {
  int dim = sigma.size();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim * p + dim + 1, dim); // Yp
  // first block------------------------
  res.block(0, 0, dim, dim).diagonal() = daily.array() * sigma.array(); // deltai * sigma or di * sigma
  if (p > 1) {
    // avoid error when p = 1
    res.block(dim, 0, dim, dim).diagonal() = weekly.array() * sigma.array(); // wi * sigma
    res.block(2 * dim, 0, dim, dim).diagonal() = monthly.array() * sigma.array(); // mi * sigma
  }
  // second block-----------------------
  res.block(dim * p, 0, dim, dim).diagonal() = sigma;
  if (include_mean) {
    return res;
  }
  return res.topRows(dim * p + dim);
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
Eigen::MatrixXd build_xdummy(Eigen::VectorXd lag_seq, double lambda, Eigen::VectorXd sigma, double eps, bool include_mean) {
  int dim = sigma.size();
  int var_lag = lag_seq.size();
  Eigen::MatrixXd Sig = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim * var_lag + dim + 1, dim * var_lag + 1);
  // first block------------------
  Eigen::MatrixXd Jp = Eigen::MatrixXd::Zero(var_lag, var_lag);
  Jp.diagonal() = lag_seq;
  Sig.diagonal() = sigma / lambda;
  res.block(0, 0, dim * var_lag, dim * var_lag) = Eigen::kroneckerProduct(Jp, Sig);
  // third block------------------
  res(dim * var_lag + dim, dim * var_lag) = eps;
  if (include_mean) {
    return res;
  }
  return res.block(0, 0, dim * var_lag + dim, dim * var_lag);
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
