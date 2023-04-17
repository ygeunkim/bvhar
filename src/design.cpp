#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

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
Eigen::MatrixXd build_y0(Eigen::MatrixXd y, int var_lag, int index) {
  int num_design = y.rows() - var_lag; // s = n - p
  int dim = y.cols(); // m: dimension of the multivariate time series
  Eigen::MatrixXd res(num_design, dim); // Yj (or Y0)
  for (int i = 0; i < num_design; i++) {
    res.row(i) = y.row(index + i - 1);
  }
  return res;
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
  int num_design = y.rows() - var_lag; // s = n - p
  int dim = y.cols(); // m: dimension of the multivariate time series
  int dim_design = dim * var_lag + 1; // k = mp + 1
  Eigen::MatrixXd res(num_design, dim_design); // X0 = [Yp, ... Y1, 1]: s x k
  for (int t = 0; t < var_lag; t++) {
    res.block(0, t * dim, num_design, dim) = build_y0(y, var_lag, var_lag - t); // Yp to Y1
  }
  if (!include_mean) {
    return res.block(0, 0, num_design, dim_design - 1);
  }
  for (int i = 0; i < num_design; i++) {
    res(i, dim_design - 1) = 1.0; // the last column for constant term
  }
  return res;
}

//' Diagonal Matrix
//' 
//' Construct a diagonal matrix.
//' 
//' @param Vector
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd diag_misc(Eigen::VectorXd x) {
  int n = x.size();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++) {
    res(i, i) = x[i];
  }
  return res;
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
Eigen::MatrixXd build_ydummy(int p, 
                             Eigen::VectorXd sigma, 
                             double lambda, 
                             Eigen::VectorXd daily,
                             Eigen::VectorXd weekly,
                             Eigen::VectorXd monthly,
                             bool include_mean) {
  int dim = sigma.size();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim * p + dim + 1, dim); // Yp
  Eigen::VectorXd weight_day(dim); // deltai * sigma or di * sigma
  Eigen::VectorXd weight_week(dim); // wi * sigma
  Eigen::VectorXd weight_month(dim); // mi * sigma
  for (int i = 0; i < dim; i++) {
    weight_day[i] = daily[i] * sigma[i] / lambda;
  }
  for (int i = 0; i < dim; i++) {
    weight_week[i] = weekly[i] * sigma[i] / lambda;
  }
  for (int i = 0; i < dim; i++) {
    weight_month[i] = monthly[i] * sigma[i] / lambda;
  }
  // first block------------------------
  res.block(0, 0, dim, dim) = diag_misc(weight_day);
  res.block(dim, 0, dim, dim) = diag_misc(weight_week);
  res.block(2 * dim, 0, dim, dim) = diag_misc(weight_month);
  // second block-----------------------
  res.block(dim * p, 0, dim, dim) = diag_misc(sigma);
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
  Eigen::MatrixXd Sig(dim, dim);
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim * var_lag + dim + 1, dim * var_lag + 1);
  // first block------------------
  Eigen::MatrixXd Jp = diag_misc(lag_seq);
  Sig = diag_misc(sigma) / lambda;
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
  int prior_shape = x_dummy.rows() - dim_design;
  prior_prec = x_dummy.transpose() * x_dummy;
  prior_mean = prior_prec.inverse() * x_dummy.transpose() * y_dummy;
  prior_scale = (y_dummy - x_dummy * prior_mean).transpose() * (y_dummy - x_dummy * prior_mean);
  return Rcpp::List::create(
    Rcpp::Named("prior_mean") = prior_mean,
    Rcpp::Named("prior_prec") = prior_prec,
    Rcpp::Named("prior_scale") = prior_scale,
    Rcpp::Named("prior_shape") = prior_shape + 2
  );
}
