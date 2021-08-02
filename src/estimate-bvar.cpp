#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' BVAR(p) Point Estimates based on Minnesota Prior
//' 
//' Point estimates for posterior distribution
//' 
//' @param x Matrix, X0
//' @param y Matrix, Y0
//' @param x_dummy Matrix, dummy X0
//' @param y_dummy Matrix, dummy Y0
//' 
//' @details
//' Augment originally processed data and dummy observation.
//' OLS from this set give the result.
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
SEXP estimate_bvar_mn (Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd x_dummy, Eigen::MatrixXd y_dummy) {
  int s = y.rows();
  int m = y.cols();
  int k = x.cols();
  int Tp = x_dummy.rows();
  int T = s + Tp;
  // initialize
  Eigen::MatrixXd ystar(T, m);
  Eigen::MatrixXd xstar(T, k);
  Eigen::MatrixXd Bhat(k, m); // MN location
  Eigen::MatrixXd Uhat(k, k); // MN scale 1
  Eigen::MatrixXd yhat(T, m);
  Eigen::MatrixXd Sighat(m, m); // iw scale
  // augment
  ystar.block(0, 0, s, m) = y;
  ystar.block(s, 0, Tp, m) = y_dummy;
  xstar.block(0, 0, s, k) = x;
  xstar.block(s, 0, Tp, k) = x_dummy;
  // point estimation
  Uhat = (xstar.adjoint() * xstar).inverse();
  Bhat = Uhat * xstar.adjoint() * ystar;
  yhat = xstar * Bhat;
  Sighat = (ystar - yhat).adjoint() * (ystar - yhat);
  return Rcpp::List::create(
    Rcpp::Named("bhat") = Rcpp::wrap(Bhat),
    Rcpp::Named("mnscale") = Rcpp::wrap(Uhat),
    Rcpp::Named("fitted") = Rcpp::wrap(yhat),
    Rcpp::Named("iwscale") = Rcpp::wrap(Sighat)
  );
}
