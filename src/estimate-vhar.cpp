#include "ols.h"

//' Compute Vector HAR Coefficient Matrices and Fitted Values
//' 
//' This function fits VHAR given response and design matrices of multivariate time series.
//' 
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param week Integer, order for weekly term
//' @param month Integer, order for monthly term
//' @param include_mean bool, Add constant term (Default: `true`) or not (`false`)
//' @param method Method to solve linear equation system. 1: normal equation, 2: cholesky, 3: HouseholderQR.
//' @details
//' Given Y0 and Y0, the function estimate least squares
//' \deqn{Y_0 = X_1 \Phi + Z}
//' 
//' @references
//' Baek, C. and Park, M. (2021). *Sparse vector heterogeneous autoregressive modeling for realized volatility*. J. Korean Stat. Soc. 50, 495–510. doi:[10.1007/s42952-020-00090-5](https://doi.org/10.1007/s42952-020-00090-5)
//' 
//' Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174–196. doi:[10.1093/jjfinec/nbp001](https://doi.org/10.1093/jjfinec/nbp001)
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_har(Eigen::MatrixXd y, int week, int month, bool include_mean, int method) {
	std::unique_ptr<bvhar::OlsVhar> ols_obj(new bvhar::OlsVhar(y, week, month, include_mean, method));
	return ols_obj->returnOlsRes();
}

//' Statistic for VHAR
//' 
//' Compute partial t-statistics for inference in VHAR model.
//' 
//' @param object `vharlse` object
//' @details
//' Partial t-statistic for H0: \eqn{\phi_{ij} = 0}
//' 
//' * For each variable (e.g. 1st variable)
//' * Standard error =  (1st) diagonal element of \eqn{\Sigma_e} estimator x diagonal elements of \eqn{(X_1^T X_1)^(-1)}
//' @noRd
// [[Rcpp::export]]
Rcpp::List infer_vhar(Rcpp::List object) {
  if (!object.inherits("vharlse")) {
    Rcpp::stop("'object' must be vharlse object.");
  }
  int dim = object["m"]; // dimension of time series
  Eigen::MatrixXd cov_mat = object["covmat"]; // sigma
  Eigen::MatrixXd coef_mat = object["coefficients"]; // Phihat(mp, m) = [Phi(daily), Phi(weekly), Phi(monthly), c^T]^T
  Eigen::MatrixXd design_mat = object["design"]; // X0: n x mp
  Eigen::MatrixXd HARtrans = object["HARtrans"]; // HAR transformation
  Eigen::MatrixXd vhar_design = design_mat * HARtrans.transpose(); // X1 = X0 * C0^T
  int num_design = object["obs"];
  int num_har = coef_mat.rows(); // 3m(+1)
  int df = num_design - num_har;
  Eigen::VectorXd XtX = (vhar_design.transpose() * vhar_design).inverse().diagonal(); // diagonal element of (XtX)^(-1)
  Eigen::MatrixXd res(num_har * dim, 3); // stack estimate, std, and t stat
  Eigen::ArrayXd st_err(num_har); // save standard error in for loop
  for (int i = 0; i < dim; i++) {
    res.block(i * num_har, 0, num_har, 1) = coef_mat.col(i);
    for (int j = 0; j < num_har; j++) {
      st_err[j] = sqrt(XtX[j] * cov_mat(i, i)); // variable-covariance matrix element
    }
    res.block(i * num_har, 1, num_har, 1) = st_err;
    res.block(i * num_har, 2, num_har, 1) = coef_mat.col(i).array() / st_err;
  }
  return Rcpp::List::create(
    Rcpp::Named("df") = df,
    Rcpp::Named("summary_stat") = res
  );
}
