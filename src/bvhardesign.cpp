#include "bvhardesign.h"

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
