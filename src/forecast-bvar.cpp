#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' Forecasting BVAR of Minnesota Prior
//' 
//' @param object \code{bvarmn} object by \code{\link{bvar_minnesota}}
//' @param step Integer, Step to forecast
//' @details
//' n-step ahead forecasting using VAR(p) recursively, based on pp35 of Lütkepohl (2007).
//' 
//' @references
//' Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' 
//' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
//' 
//' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
Rcpp::List forecast_bvarmn(Rcpp::List object, int step) {
  if (!object.inherits("bvarmn")) Rcpp::stop("'object' must be bvarmn object.");
  Eigen::MatrixXd response_mat = object["y0"]; // Y0
  Eigen::MatrixXd posterior_mean_mat = object["mn_mean"]; // bhat = posterior mean of MN
  Eigen::MatrixXd posterior_prec_mat = object["mn_prec"]; // vhat = posterior precision of MN to compute SE
  int dim = object["m"]; // dimension of time series
  int var_lag = object["p"]; // VAR(p)
  int num_design = object["obs"]; // s = n - p
  int dim_design = dim * var_lag + 1; // k = mp + 1
  Eigen::MatrixXd last_pvec(1, dim_design); // vectorize the last p observation and include 1
  Eigen::MatrixXd tmp_vec(1, (var_lag - 1) * dim);
  Eigen::MatrixXd res(step, dim); // h x m matrix
  Eigen::VectorXd sig_closed(step); // se^2 for each forecast (except Sigma2 part, i.e. closed form)
  for (int i = 0; i < step; i++) {
    sig_closed(i) = 1.0;
  }
  for (int i = 0; i < var_lag; i++) {
    last_pvec.block(0, i * dim, 1, dim) = response_mat.block(num_design - 1 - i, 0, 1, dim);
  }
  last_pvec(0, dim_design - 1) = 1.0;
  sig_closed.block(0, 0, 1, 1) += last_pvec * posterior_prec_mat.inverse() * last_pvec.adjoint();
  res.block(0, 0, 1, dim) = last_pvec * posterior_mean_mat; // y(n + 1)^T = [y(n)^T, ..., y(n - p + 1)^T, 1] %*% Bhat
  if (step == 1) {
    return Rcpp::List::create(
      Rcpp::Named("posterior_mean") = res,
      Rcpp::Named("posterior_var_closed") = sig_closed
    );
  }
  // Next h - 1: recursively
  for (int i = 1; i < step; i++) {
    tmp_vec = last_pvec.block(0, 0, 1, (var_lag - 1) * dim); // remove the last m (except 1)
    last_pvec.block(0, dim, 1, (var_lag - 1) * dim) = tmp_vec;
    last_pvec.block(0, 0, 1, dim) = res.block(i - 1, 0, 1, dim);
    sig_closed.block(i, 0, 1, 1) += last_pvec * posterior_prec_mat.inverse() * last_pvec.adjoint();
    // y(n + 2)^T = [yhat(n + 1)^T, y(n)^T, ... y(n - p + 2)^T, 1] %*% Bhat
    res.block(i, 0, 1, dim) = last_pvec * posterior_mean_mat;
  }
  return Rcpp::List::create(
    Rcpp::Named("posterior_mean") = res,
    Rcpp::Named("posterior_var_closed") = sig_closed
  );
}

//' Forecasting BVAR of Non-hierarchical Matrix Normal Prior
//' 
//' @param object \code{bvarmn} object by \code{\link{bvar_minnesota}}
//' @param step Integer, Step to forecast
//' @details
//' n-step ahead forecasting using VAR(p) recursively, based on pp35 of Lütkepohl (2007).
//' 
//' @references
//' Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' 
//' Ghosh, S., Khare, K., & Michailidis, G. (2018). \emph{High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models}. Journal of the American Statistical Association, 114(526). \url{https://doi:10.1080/01621459.2018.1437043}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
Rcpp::List forecast_bvarmn_flat(Rcpp::List object, int step) {
  if (!object.inherits("bvarflat")) Rcpp::stop("'object' must be bvarflat object.");
  Eigen::MatrixXd response_mat = object["y0"]; // Y0
  Eigen::MatrixXd posterior_mean_mat = object["mn_mean"]; // bhat = posterior mean of MN
  Eigen::MatrixXd posterior_prec_mat = object["mn_prec"]; // vhat = posterior precision of MN to compute SE
  int dim = object["m"]; // dimension of time series
  int var_lag = object["p"]; // VAR(p)
  int num_design = object["obs"]; // s = n - p
  int dim_design = dim * var_lag + 1; // k = mp + 1
  Eigen::MatrixXd last_pvec(1, dim_design); // vectorize the last p observation and include 1
  Eigen::MatrixXd tmp_vec(1, (var_lag - 1) * dim);
  Eigen::MatrixXd res(step, dim); // h x m matrix
  Eigen::VectorXd sig_closed(step); // se^2 for each forecast (except Sigma2 part, i.e. closed form)
  for (int i = 0; i < step; i++) {
    sig_closed(i) = 1.0;
  }
  for (int i = 0; i < var_lag; i++) {
    last_pvec.block(0, i * dim, 1, dim) = response_mat.block(num_design - 1 - i, 0, 1, dim);
  }
  last_pvec(0, dim_design - 1) = 1.0;
  sig_closed.block(0, 0, 1, 1) += last_pvec * posterior_prec_mat.inverse() * last_pvec.adjoint();
  res.block(0, 0, 1, dim) = last_pvec * posterior_mean_mat; // y(n + 1)^T = [y(n)^T, ..., y(n - p + 1)^T, 1] %*% Bhat
  if (step == 1) {
    return Rcpp::List::create(
      Rcpp::Named("posterior_mean") = res,
      Rcpp::Named("posterior_var_closed") = sig_closed
    );
  }
  // Next h - 1: recursively
  for (int i = 1; i < step; i++) {
    tmp_vec = last_pvec.block(0, 0, 1, (var_lag - 1) * dim); // remove the last m (except 1)
    last_pvec.block(0, dim, 1, (var_lag - 1) * dim) = tmp_vec;
    last_pvec.block(0, 0, 1, dim) = res.block(i - 1, 0, 1, dim);
    sig_closed.block(i, 0, 1, 1) += last_pvec * posterior_prec_mat.inverse() * last_pvec.adjoint();
    // y(n + 2)^T = [yhat(n + 1)^T, y(n)^T, ... y(n - p + 2)^T, 1] %*% Bhat
    res.block(i, 0, 1, dim) = last_pvec * posterior_mean_mat;
  }
  return Rcpp::List::create(
    Rcpp::Named("posterior_mean") = res,
    Rcpp::Named("posterior_var_closed") = sig_closed
  );
}
