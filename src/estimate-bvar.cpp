#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' BVAR(p) Point Estimates based on Minnesota Prior
//' 
//' Point estimates for posterior distribution
//' 
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param x_dummy Dummy observations Xp for design matrix X0
//' @param y_dummy Dummy observations Yp for design matrix Y0
//' 
//' @details
//' Augment originally processed data and dummy observation.
//' OLS from this set gives the result.
//' 
//' @references
//' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
//' 
//' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_mn(Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd x_dummy, Eigen::MatrixXd y_dummy) {
  int num_design = y.rows(); // s = n - p
  int dim = y.cols(); // m
  int dim_design = x_dummy.cols(); // k = mp (+ 1)
  int num_dummy = x_dummy.rows(); // Tp = mp + m (+ 1)
  int num_augment = num_design + num_dummy; // T = s + Tp
  if (num_dummy != y_dummy.rows()) {
    Rcpp::stop("Wrong dimension: x_dummy and y_dummy");
  }
  if (dim_design != x.cols()) {
    Rcpp::stop("Wrong dimension: x and x_dummy");
  }
  if (y_dummy.cols() != dim) {
    Rcpp::stop("Wrong dimension: y_dummy");
  }
  // prior-----------------------------------------------
  Eigen::MatrixXd prior_prec = x_dummy.transpose() * x_dummy; // prior mn precision
  Eigen::MatrixXd prior_mean = prior_prec.inverse() * x_dummy.transpose() * y_dummy; // prior mn mean
  Eigen::MatrixXd prior_scale = (y_dummy - x_dummy * prior_mean).transpose() * (y_dummy - x_dummy * prior_mean);; // prior iw scale
  int prior_shape = num_dummy - dim_design + 2;
  // posterior-------------------------------------------
  // initialize posteriors
  Eigen::MatrixXd ystar(num_augment, dim); // [Y0, Yp]
  Eigen::MatrixXd xstar(num_augment, dim_design); // [X0, Xp]
  // augment
  ystar.block(0, 0, num_design, dim) = y;
  ystar.block(num_design, 0, num_dummy, dim) = y_dummy;
  xstar.block(0, 0, num_design, dim_design) = x;
  xstar.block(num_design, 0, num_dummy, dim_design) = x_dummy;
  // point estimation
  Eigen::MatrixXd prec_mat = (xstar.transpose() * xstar); // MN precision
  Eigen::MatrixXd coef_mat = prec_mat.inverse() * xstar.transpose() * ystar; // MN mean
  Eigen::MatrixXd yhat = x * coef_mat; // X0 %*% ahat
  Eigen::MatrixXd resid = y - yhat; // Y0 - X0 %*% ahat
  Eigen::MatrixXd yhat_star = xstar * coef_mat; // xstar %*% ahat
  Eigen::MatrixXd scale_mat = (ystar - yhat_star).transpose() * (ystar - yhat_star); // IW scale
  return Rcpp::List::create(
    Rcpp::Named("prior_mean") = prior_mean,
    Rcpp::Named("prior_prec") = prior_prec,
    Rcpp::Named("prior_scale") = prior_scale,
    Rcpp::Named("prior_shape") = prior_shape,
    Rcpp::Named("mnmean") = coef_mat,
    Rcpp::Named("mnprec") = prec_mat,
    Rcpp::Named("fitted") = yhat,
    Rcpp::Named("residuals") = resid,
    Rcpp::Named("iwscale") = scale_mat
  );
}

//' BVAR(p) Point Estimates based on Nonhierarchical Matrix Normal Prior
//' 
//' Point estimates for Ghosh et al. (2018) nonhierarchical model for BVAR.
//' 
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param U Positive definite matrix, covariance matrix corresponding to the column of the model parameter B
//' 
//' @details
//' In Ghosh et al. (2018), there are many models for BVAR such as hierarchical or non-hierarchical.
//' Among these, this function chooses the most simple non-hierarchical matrix normal prior in Section 3.1.
//' 
//' @references
//' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). [https://doi:10.1080/01621459.2018.1437043](https://doi:10.1080/01621459.2018.1437043)
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_mn_flat(Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::MatrixXd U) {
  int num_design = y.rows();
  int dim = y.cols();
  int dim_design = x.cols();
  if (U.rows() != x.cols()) {
    Rcpp::stop("Wrong dimension: U");
  }
  if (U.cols() != x.cols()) {
    Rcpp::stop("Wrong dimension: U");
  }
  Eigen::MatrixXd prec_mat = (x.transpose() * x + U); // MN precision
  Eigen::MatrixXd mn_scale_mat = prec_mat.inverse(); // MN scale 1 = inverse of precision
  Eigen::MatrixXd coef_mat = mn_scale_mat * x.transpose() * y; // MN mean
  Eigen::MatrixXd yhat = x * coef_mat; // x %*% bhat
  Eigen::MatrixXd Is = Eigen::MatrixXd::Identity(num_design, num_design);
  Eigen::MatrixXd scale_mat = y.transpose() * (Is - x * mn_scale_mat * x.transpose()) * y; // IW scale
  return Rcpp::List::create(
    Rcpp::Named("mnmean") = coef_mat,
    Rcpp::Named("mnprec") = prec_mat,
    Rcpp::Named("fitted") = yhat,
    Rcpp::Named("iwscale") = scale_mat,
    Rcpp::Named("iwshape") = num_design - dim - 1
  );
}
