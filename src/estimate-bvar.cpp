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

//' BVAR(p) Point Estimates based on Nonhierarchical Matrix Normal Prior
//' 
//' Point estimates for Ghosh et al. (2018) nonhierarchical model for BVAR.
//' 
//' @param x Matrix, X0
//' @param y Matrix, Y0
//' @param U Positive definite matrix, covariance matrix corresponding to the column of the model parameter B
//' 
//' @details
//' In Ghosh et al. (2018), there are many models for BVAR such as hierarchical or non-hierarchical.
//' Among these, this function chooses the most simple non-hierarchical matrix normal prior.
//' 
//' @references
//' Ghosh, S., Khare, K., & Michailidis, G. (2018). \emph{High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models}. Journal of the American Statistical Association, 114(526). \url{https://doi:10.1080/01621459.2018.1437043}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP estimate_bvar_ghosh (Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd U) {
  int s = y.rows();
  int m = y.cols();
  int k = x.cols();
  Eigen::MatrixXd Bhat(k, m); // MN mean
  Eigen::MatrixXd Uhat(k, k); // MN scale 1
  Eigen::MatrixXd Sighat(m, m); // IW scale
  Eigen::MatrixXd Is(s, s);
  Is.setIdentity(s, s);
  Uhat = (x.adjoint() * x + U).inverse();
  Bhat = Uhat * x.adjoint() * y;
  Sighat = y.adjoint() * (Is - x * Uhat * x.adjoint()) * y;
  return Rcpp::List::create(
    Rcpp::Named("bhat") = Rcpp::wrap(Bhat),
    Rcpp::Named("mnscale") = Rcpp::wrap(Uhat),
    Rcpp::Named("iwscale") = Rcpp::wrap(Sighat)
  );
}
