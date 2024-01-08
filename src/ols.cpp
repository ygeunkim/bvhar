#include "ols.h"

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

OlsVar::OlsVar(const Eigen::MatrixXd& y, int lag, const bool include_mean)
: lag(lag), const_term(include_mean), dim(y.cols()), data(y) {
	response = build_y0(data, lag, lag + 1);
	design = build_design(data, lag, const_term);
	num_design = response.rows();
	dim_design = design.cols();
	coef = Eigen::MatrixXd::Zero(dim_design, dim);
	yhat = Eigen::MatrixXd::Zero(num_design, dim);
	cov = Eigen::MatrixXd::Zero(dim, dim);
}

void OlsVar::estimateCoef() {
	std::cout << "Normal equation" << std::endl;
	coef = (design.transpose() * design).inverse() * design.transpose() * response;
}

void OlsVar::fitObs() {
	yhat = design * coef;
	resid = response - yhat;
}

void OlsVar::estimateCov() {
	cov = resid.transpose() * resid / (num_design - dim_design);
}

Rcpp::List OlsVar::returnOlsRes() {
	return Rcpp::List::create(
		Rcpp::Named("coefficients") = coef,
		Rcpp::Named("fitted.values") = yhat,
		Rcpp::Named("residuals") = resid,
		Rcpp::Named("covmat") = cov,
		Rcpp::Named("df") = dim_design,
		Rcpp::Named("p") = lag,
		Rcpp::Named("m") = dim,
		Rcpp::Named("obs") = num_design,
		Rcpp::Named("process") = "VAR",
		Rcpp::Named("type") = const_term ? "const" : "none",
		Rcpp::Named("y0") = response,
		Rcpp::Named("y") = data,
		Rcpp::Named("design") = design
	);
}

LltVar::LltVar(const Eigen::MatrixXd& y, int lag, const bool include_mean) : OlsVar(y, lag, include_mean) {
	llt_selfadjoint.compute(design.transpose() * design);
}

void LltVar::estimateCoef() {
	coef = llt_selfadjoint.solve(design.transpose() * response);
}

QrVar::QrVar(const Eigen::MatrixXd& y, int lag, const bool include_mean) : OlsVar(y, lag, include_mean) {
	qr_design.compute(design);
}

void QrVar::estimateCoef() {
	coef = qr_design.solve(response);
}
