#include "ols.h"

//' Compute VAR(p) Coefficient Matrices and Fitted Values
//' 
//' This function fits VAR(p) given response and design matrices of multivariate time series.
//' 
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param method Method to solve linear equation system. 1: normal equation, 2: cholesky, 3: HouseholderQR.
//' @details
//' Given Y0 and Y0, the function estimate least squares
//' Y0 = X0 A + Z
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_var(Eigen::MatrixXd y, int lag, bool include_mean, int method) {
	std::unique_ptr<bvhar::OlsVar> ols_obj(new bvhar::OlsVar(y, lag, include_mean, method));
	return ols_obj->returnOlsRes();
}

//' Covariance Estimate for Residual Covariance Matrix
//' 
//' Compute ubiased estimator for residual covariance.
//' 
//' @param z Matrix, residual
//' @param num_design Integer, Number of sample used (s = n - p)
//' @param dim_design Ingeger, Number of parameter for each dimension (k = mp + 1)
//' @details
//' See pp75 Lütkepohl (2007).
//' 
//' * s = n - p: sample used (`num_design`)
//' * k = mp + 1 (m: dimension, p: VAR lag): number of parameter for each dimension (`dim_design`)
//' 
//' Then an unbiased estimator for \eqn{\Sigma_e} is
//' 
//' \deqn{\hat{\Sigma}_e = \frac{1}{s - k} (Y_0 - \hat{A} X_0)^T (Y_0 - \hat{A} X_0)}
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_cov(Eigen::MatrixXd z, int num_design, int dim_design) {
  Eigen::MatrixXd cov_mat(z.cols(), z.cols());
  cov_mat = z.transpose() * z / (num_design - dim_design);
  return cov_mat;
}

//' Statistic for VAR
//' 
//' Compute partial t-statistics for inference in VAR model.
//' 
//' @param object `varlse` object
//' @details
//' Partial t-statistic for H0: aij = 0
//' 
//' * For each variable (e.g. 1st variable)
//' * Standard error =  (1st) diagonal element of \eqn{\Sigma_e} estimator x diagonal elements of \eqn{(X_0^T X_0)^(-1)}
//' 
//' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
//' @noRd
// [[Rcpp::export]]
Rcpp::List infer_var(Rcpp::List object) {
  if (!object.inherits("varlse")) {
    Rcpp::stop("'object' must be varlse object.");
  }
  int dim = object["m"]; // dimension of time series
  Eigen::MatrixXd cov_mat = object["covmat"]; // sigma
  Eigen::MatrixXd coef_mat = object["coefficients"]; // Ahat(mp, m) = [A1^T, A2^T, ..., Ap^T, c^T]^T
  Eigen::MatrixXd design_mat = object["design"]; // X0: n x mp
  int num_design = object["obs"];
  int dim_design = coef_mat.rows(); // mp(+1)
  int df = num_design - dim_design;
  Eigen::VectorXd XtX = (design_mat.transpose() * design_mat).inverse().diagonal(); // diagonal element of (XtX)^(-1)
  Eigen::MatrixXd res(dim_design * dim, 3); // stack estimate, std, and t stat
  Eigen::ArrayXd st_err(dim_design); // save standard error in for loop
  for (int i = 0; i < dim; i++) {
    res.block(i * dim_design, 0, dim_design, 1) = coef_mat.col(i);
    for (int j = 0; j < dim_design; j++) {
      st_err[j] = sqrt(XtX[j] * cov_mat(i, i)); // variable-covariance matrix element
    }
    res.block(i * dim_design, 1, dim_design, 1) = st_err;
    res.block(i * dim_design, 2, dim_design, 1) = coef_mat.col(i).array() / st_err;
  }
  return Rcpp::List::create(
    Rcpp::Named("df") = df,
    Rcpp::Named("summary_stat") = res
  );
}
