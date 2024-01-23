#include <ols.h>

namespace bvhar {

MultiOls::MultiOls(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
: design(x), response(y),
	dim(response.cols()), num_design(response.rows()), dim_design(design.cols()) {
	coef = Eigen::MatrixXd::Zero(dim_design, dim);
	yhat = Eigen::MatrixXd::Zero(num_design, dim);
	resid = Eigen::MatrixXd::Zero(num_design, dim);
	cov = Eigen::MatrixXd::Zero(dim, dim);
}

void MultiOls::estimateCoef() {
	coef = (design.transpose() * design).inverse() * design.transpose() * response; // return coef -> use in OlsVar
}

void MultiOls::fitObs() {
	yhat = design * coef;
	resid = response - yhat;
}

void MultiOls::estimateCov() {
	cov = resid.transpose() * resid / (num_design - dim_design);
}

Rcpp::List MultiOls::returnOlsRes() {
	estimateCoef();
	fitObs();
	estimateCov();
	return Rcpp::List::create(
		Rcpp::Named("coefficients") = coef,
		Rcpp::Named("fitted.values") = yhat,
		Rcpp::Named("residuals") = resid,
		Rcpp::Named("covmat") = cov,
		Rcpp::Named("df") = dim_design,
		Rcpp::Named("m") = dim,
		Rcpp::Named("obs") = num_design,
		Rcpp::Named("y0") = response
		// ,
		// Rcpp::Named("design") = design
	);
}

LltOls::LltOls(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) : MultiOls(x, y) {
	llt_selfadjoint.compute(design.transpose() * design);
}

void LltOls::estimateCoef() {
	coef = llt_selfadjoint.solve(design.transpose() * response);
}

QrOls::QrOls(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) : MultiOls(x, y) {
	qr_design.compute(design);
}

void QrOls::estimateCoef() {
	coef = qr_design.solve(response);
}

OlsVar::OlsVar(const Eigen::MatrixXd& y, int lag, const bool include_mean, int method)
: lag(lag), const_term(include_mean), data(y) {
	response = build_y0(data, lag, lag + 1);
	design = build_design(data, lag, const_term);
	switch (method) {
  case 1:
		_ols = std::unique_ptr<MultiOls>(new MultiOls(design, response));
		break;
  case 2:
    _ols = std::unique_ptr<MultiOls>(new LltOls(design, response));
    break;
  case 3:
    _ols = std::unique_ptr<MultiOls>(new QrOls(design, response));
    break;
  }
}

Rcpp::List OlsVar::returnOlsRes() {
	Rcpp::List ols_res = _ols->returnOlsRes();
	ols_res["p"] = lag;
	ols_res["totobs"] = data.rows();
	ols_res["process"] = "VAR";
	ols_res["type"] = const_term ? "const" : "none";
	ols_res["design"] = design;
	ols_res["y"] = data;
	return ols_res;
}

OlsVhar::OlsVhar(const Eigen::MatrixXd& y, int week, int month, const bool include_mean, int method)
: week(week), month(month), const_term(include_mean), data(y) {
	response = build_y0(data, month, month + 1);
	har_trans = scale_har(response.cols(), week, month, const_term);
	var_design = build_design(data, month, const_term);
	design = var_design * har_trans.transpose();
	switch (method) {
  case 1:
		_ols = std::unique_ptr<MultiOls>(new MultiOls(design, response));
		break;
  case 2:
    _ols = std::unique_ptr<MultiOls>(new LltOls(design, response));
    break;
  case 3:
    _ols = std::unique_ptr<MultiOls>(new QrOls(design, response));
    break;
  }
}

Rcpp::List OlsVhar::returnOlsRes() {
	Rcpp::List ols_res = _ols->returnOlsRes();
	ols_res["p"] = 3;
	ols_res["week"] = week;
	ols_res["month"] = month;
	ols_res["totobs"] = data.rows();
	ols_res["process"] = "VHAR";
	ols_res["type"] = const_term ? "const" : "none";
	ols_res["HARtrans"] = har_trans;
	ols_res["design"] = var_design;
	ols_res["y"] = data;
	return ols_res;
}

} // namespace bvhar