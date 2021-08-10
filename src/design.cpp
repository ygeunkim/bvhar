#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' Build Y0 matrix in VAR(p)
//' 
//' @param x Matrix, time series data
//' @param p VAR lag
//' @param t starting index to extract
//' 
//' @details
//' Given data Y,
//' \deqn{Y0 = [y_t^T, y_{t + 1}^T, \ldots, y_{t + n - p - 1}^T]^T}
//' is the (n - p) x m matrix.
//' 
//' In case of Y0, t = p + 1.
//' This function is used when constructing X0.
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP build_y0(Eigen::MatrixXd x, int p, int t) {
  int s = x.rows() - p;
  int m = x.cols();
  Eigen::MatrixXd res(s, m); // Y0
  for (int i = 0; i < s; i++) {
    res.row(i) = x.row(t + i - 1);
  }
  return Rcpp::wrap(res);
}

//' Build X0 matrix in VAR(p)
//' 
//' @param x Matrix, time series data
//' @param p VAR lag
//' 
//' @details
//' X0 is
//' \deqn{X0 = [Y_p, \ldots, Y_1, 1]}
//' i.e. (n - p) x (mp + 1) matrix
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP build_design(Eigen::MatrixXd x, int p) {
  int s = x.rows() - p;
  int m = x.cols();
  int k = m * p + 1;
  Eigen::MatrixXd res(s, k); // X0
  for (int t = 0; t < p; t++) {
    res.block(0, t * m, s, m) = Rcpp::as<Eigen::MatrixXd>(build_y0(x, p, p - t));
  }
  for (int i = 0; i < s; i++) {
    res(i, k - 1) = 1.0;
  }
  return Rcpp::wrap(res);
}

//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP diag_misc(Eigen::VectorXd x) {
  int n = x.size();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++) {
    res(i, i) = x[i];
  }
  return Rcpp::wrap(res);
}

//' Construct Dummy response for Minnesota Prior
//' 
//' Define dummy Y observations to add for Minnesota moments.
//' 
//' @param p integer, VAR lag. For VHAR, put 3.
//' @param sigma vector, standard error of each variable
//' @param lambda double, tightness of the prior around a random walk or white noise
//' @param delta vector, prior belief about white noise (Litterman sets 1)
//' 
//' @details
//' Bańbura et al. (2010) defines dummy observation and augment to the original data matrix to construct Litterman (1986) prior.
//' 
//' @references
//' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
//' 
//' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP build_ydummy(int p, Eigen::VectorXd sigma, double lambda, Eigen::VectorXd delta) {
  int m = sigma.size();
  Eigen::MatrixXd res(m * p + m + 1, m); // Yp
  Eigen::VectorXd wt(m); // delta * sigma
  for (int i = 0; i < m; i++) {
    wt[i] = delta[i] * sigma[i] / lambda;
  }
  // first block
  res.block(0, 0, m, m) = Rcpp::as<Eigen::MatrixXd>(diag_misc(wt));
  res.block(m, 0, m * (p - 1), m) = Eigen::MatrixXd::Zero(m * (p - 1), m);
  // second block
  res.block(m * p, 0, m, m) = Rcpp::as<Eigen::MatrixXd>(diag_misc(sigma));
  // third block
  res.block(m * p + m, 0, 1, m) = Eigen::MatrixXd::Zero(1, m);
  return Rcpp::wrap(res);
}

//' Construct Dummy design matrix for Minnesota Prior
//' 
//' Define dummy X observation to add for Minnesota moments.
//' 
//' @param p integer, VAR lag. For VHAR, put 3.
//' @param sigma vector, standard error of each variable
//' @param lambda double, tightness of the prior around a random walk or white noise
//' @param eps double, very small number
//' 
//' @details
//' Bańbura et al. (2010) defines dummy observation and augment to the original data matrix to construct Litterman (1986) prior.
//' 
//' @references
//' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
//' 
//' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP build_xdummy(int p, double lambda, Eigen::VectorXd sigma, double eps) {
  int m = sigma.size();
  Eigen::VectorXd p_seq(p);
  Eigen::MatrixXd Jp(p, p);
  Eigen::MatrixXd Sig(m, m);
  Eigen::MatrixXd res(m * p + m + 1, m * p + 1);
  for (int i = 0; i < p; i++) {
    p_seq[i] = i + 1;
  }
  // first block
  Jp = Rcpp::as<Eigen::MatrixXd>(diag_misc(p_seq));
  Sig = Rcpp::as<Eigen::MatrixXd>(diag_misc(sigma)) / lambda;
  res.block(0, 0, m * p, m * p) = Eigen::kroneckerProduct(Jp, Sig);
  res.block(0, m * p, m * p, 1) = Eigen::MatrixXd::Zero(m * p, 1);
  // second block
  res.block(m * p, 0, m, m * p) = Eigen::MatrixXd::Zero(m, m * p);
  res.block(m * p, m * p, m, 1) = Eigen::MatrixXd::Zero(m, 1);
  // third block
  res.block(m * p + m, 0, 1, m * p) = Eigen::MatrixXd::Zero(1, m * p);
  res(m * p + m, m * p) = eps;
  return Rcpp::wrap(res);
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
//' (MN: \href{https://en.wikipedia.org/wiki/Matrix_normal_distribution}{matrix normal}, IW: \href{https://en.wikipedia.org/wiki/Inverse-Wishart_distribution}{inverse-wishart})
//' 
//' Bańbura et al. (2010) provides the formula how to find each matrix to match Minnesota moments.
//' 
//' @references
//' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
//' 
//' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP minnesota_prior (Eigen::MatrixXd x_dummy, Eigen::MatrixXd y_dummy) {
  int m = y_dummy.cols();
  int k = x_dummy.cols();
  Eigen::MatrixXd B(k, m); // location of matrix normal
  Eigen::MatrixXd Omega(k, k); // scale 1 of matrix normal
  Eigen::MatrixXd Sigma(m, m); // scale 2 of matrix normal
  Eigen::MatrixXd S(m, m); // scale of inverse wishart
  double a;
  Omega = (x_dummy.adjoint() * x_dummy).inverse();
  B = Omega * x_dummy.adjoint() * y_dummy;
  S = (y_dummy - x_dummy * B).adjoint() * (y_dummy - x_dummy * B);
  a = y_dummy.rows() - k;
  return Rcpp::List::create(
    Rcpp::Named("B0") = Rcpp::wrap(B),
    Rcpp::Named("Omega0") = Rcpp::wrap(Omega),
    Rcpp::Named("S0") = Rcpp::wrap(S),
    Rcpp::Named("alpha0") = Rcpp::wrap(a)
  );
}

//' Construct Dummy response for Second Version of BVHAR Minnesota Prior
//' 
//' Define dummy Y observations to add for Minnesota moments.
//' This function also fills zero matrix in the first block for applying to VHAR.
//' 
//' @param sigma vector, standard error of each variable
//' @param lambda double, tightness of the prior around a random walk or white noise
//' @param daily vector, instead of delta vector in the original Minnesota design (Litterman sets 1).
//' @param weekly vector, this was zero in the original Minnesota design
//' @param monthly vector, this was zero in the original Minnesota design
//' 
//' @details
//' Bańbura et al. (2010) defines dummy observation and augment to the original data matrix to construct Litterman (1986) prior.
//' 
//' @references
//' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
//' 
//' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP build_ydummy_bvhar(Eigen::VectorXd sigma, double lambda, Eigen::VectorXd daily, Eigen::VectorXd weekly, Eigen::VectorXd monthly) {
  int m = sigma.size();
  Eigen::MatrixXd res(3 * m + m + 1, m); // Yp
  Eigen::VectorXd wt1(m); // daily * sigma
  Eigen::VectorXd wt2(m); // weekly * sigma
  Eigen::VectorXd wt3(m); // monthly * sigma
  for (int i = 0; i < m; i++) {
    wt1[i] = daily[i] * sigma[i] / lambda;
  }
  for (int i = 0; i < m; i++) {
    wt2[i] = weekly[i] * sigma[i] / lambda;
  }
  for (int i = 0; i < m; i++) {
    wt3[i] = monthly[i] * sigma[i] / lambda;
  }
  // first block
  res.block(0, 0, m, m) = Rcpp::as<Eigen::MatrixXd>(diag_misc(wt1));
  res.block(m, 0, m, m) = Rcpp::as<Eigen::MatrixXd>(diag_misc(wt2));
  res.block(2 * m, 0, m, m) = Rcpp::as<Eigen::MatrixXd>(diag_misc(wt3));
  // second block
  res.block(3 * m, 0, m, m) = Rcpp::as<Eigen::MatrixXd>(diag_misc(sigma));
  // third block
  res.block(3 * m + m, 0, 1, m) = Eigen::MatrixXd::Zero(1, m);
  return Rcpp::wrap(res);
}
