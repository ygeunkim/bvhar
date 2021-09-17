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
//' @export
// [[Rcpp::export]]
Rcpp::List estimate_bvar_mn (Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd x_dummy, Eigen::MatrixXd y_dummy) {
  int num_design = y.rows(); // s = n - p
  int dim = y.cols(); // m
  int dim_design = x_dummy.cols(); // k = mp (+ 1)
  int num_dummy = x_dummy.rows(); // Tp = mp + m (+ 1)
  int num_augment = num_design + num_dummy; // T = s + Tp
  if (num_dummy != y_dummy.rows()) Rcpp::stop("Wrong dimension: x_dummy and y_dummy");
  if (dim_design != x.cols()) Rcpp::stop("Wrong dimension: x and x_dummy");
  if (y_dummy.cols() != dim) Rcpp::stop("Wrong dimension: y_dummy");
  // prior-----------------------------------------------
  Eigen::MatrixXd prior_mean(dim_design, dim); // prior mn mean
  Eigen::MatrixXd prior_prec(dim_design, dim_design); // prior mn precision
  Eigen::MatrixXd prior_scale(dim, dim); // prior iw scale
  prior_prec = x_dummy.adjoint() * x_dummy;
  prior_mean = prior_prec.inverse() * x_dummy.adjoint() * y_dummy;
  prior_scale = (y_dummy - x_dummy * prior_mean).adjoint() * (y_dummy - x_dummy * prior_mean);
  int prior_shape = num_dummy - dim_design; // prior iw shape
  // posterior-------------------------------------------
  // initialize posteriors
  Eigen::MatrixXd ystar(num_augment, dim); // [Y0, Yp]
  Eigen::MatrixXd xstar(num_augment, dim_design); // [X0, Xp]
  Eigen::MatrixXd Bhat(dim_design, dim); // MN mean
  Eigen::MatrixXd Uhat(dim_design, dim_design); // MN precision
  Eigen::MatrixXd yhat(num_design, dim); // x %*% bhat
  Eigen::MatrixXd yhat_star(num_augment, dim); // xstar %*% bhat
  Eigen::MatrixXd Sighat(dim, dim); // IW scale
  // augment
  ystar.block(0, 0, num_design, dim) = y;
  ystar.block(num_design, 0, num_dummy, dim) = y_dummy;
  xstar.block(0, 0, num_design, dim_design) = x;
  xstar.block(num_design, 0, num_dummy, dim_design) = x_dummy;
  // point estimation
  Uhat = (xstar.adjoint() * xstar); // precision hat
  Bhat = Uhat.inverse() * xstar.adjoint() * ystar;
  yhat = x * Bhat;
  yhat_star = xstar * Bhat;
  Sighat = (ystar - yhat_star).adjoint() * (ystar - yhat_star);
  return Rcpp::List::create(
    Rcpp::Named("prior_mean") = prior_mean,
    Rcpp::Named("prior_prec") = prior_prec,
    Rcpp::Named("prior_scale") = prior_scale,
    Rcpp::Named("prior_shape") = prior_shape,
    Rcpp::Named("bhat") = Bhat,
    Rcpp::Named("mnprec") = Uhat,
    Rcpp::Named("fitted") = yhat,
    Rcpp::Named("iwscale") = Sighat
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
//' Among these, this function chooses the most simple non-hierarchical matrix normal prior in Section 3.1.
//' 
//' @references
//' Ghosh, S., Khare, K., & Michailidis, G. (2018). \emph{High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models}. Journal of the American Statistical Association, 114(526). \url{https://doi:10.1080/01621459.2018.1437043}
//' 
//' @export
// [[Rcpp::export]]
Rcpp::List estimate_mn_flat (Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd U) {
  int s = y.rows();
  int m = y.cols();
  int k = x.cols();
  if (U.rows() != x.cols()) Rcpp::stop("Wrong dimension: U");
  if (U.cols() != x.cols()) Rcpp::stop("Wrong dimension: U");
  Eigen::MatrixXd Bhat(k, m); // MN mean
  Eigen::MatrixXd Uhat(k, k); // MN precision
  Eigen::MatrixXd Uhat_inv(k, k);
  Eigen::MatrixXd Sighat(m, m); // IW scale
  Eigen::MatrixXd yhat(s, m); // x %*% bhat
  Eigen::MatrixXd Is(s, s);
  Is.setIdentity(s, s);
  Uhat = (x.adjoint() * x + U);
  Uhat_inv = Uhat.inverse();
  Bhat = Uhat_inv * x.adjoint() * y;
  yhat = x * Bhat;
  Sighat = y.adjoint() * (Is - x * Uhat_inv * x.adjoint()) * y;
  return Rcpp::List::create(
    Rcpp::Named("bhat") = Bhat,
    Rcpp::Named("mnprec") = Uhat,
    Rcpp::Named("fitted") = yhat,
    Rcpp::Named("iwscale") = Sighat,
    Rcpp::Named("iwshape") = s - m - 1
  );
}
