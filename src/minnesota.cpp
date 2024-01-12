#include "minnesota.h"

MinnSpec::MinnSpec(Rcpp::List& bayes_spec)
: _sigma(Rcpp::as<Eigen::VectorXd>(bayes_spec["sigma"])),
	_lambda(bayes_spec["lambda"]),
	_eps(bayes_spec["eps"]) {}

BvarSpec::BvarSpec(Rcpp::List& bayes_spec)
: MinnSpec(bayes_spec),
	_delta(Rcpp::as<Eigen::VectorXd>(bayes_spec["delta"])) {}

BvharSpec::BvharSpec(Rcpp::List& bayes_spec)
: MinnSpec(bayes_spec),
	_daily(Rcpp::as<Eigen::VectorXd>(bayes_spec["daily"])),
	_weekly(Rcpp::as<Eigen::VectorXd>(bayes_spec["weekly"])),
	_monthly(Rcpp::as<Eigen::VectorXd>(bayes_spec["monthly"])) {}

Minnesota::Minnesota(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::MatrixXd& x_dummy, const Eigen::MatrixXd& y_dummy)
: design(x), response(y), dummy_design(x_dummy), dummy_response(y_dummy),
	num_design(response.rows()), dim(response.cols()), dim_design(design.cols()),
	num_dummy(dummy_design.rows()), num_augment(num_design + num_dummy) {
	prior_prec = dummy_design.transpose() * dummy_design;
	prior_mean = prior_prec.inverse() * dummy_design.transpose() * dummy_response;
	prior_scale = (dummy_response - dummy_design * prior_mean).transpose() * (dummy_response - dummy_design * prior_mean);
	prior_shape = num_dummy - dim_design + 2;
	ystar = Eigen::MatrixXd::Zero(num_augment, dim);
	ystar << y,
					 y_dummy;
	xstar = Eigen::MatrixXd::Zero(num_augment, dim_design);
	xstar << x,
					 x_dummy;
	coef = Eigen::MatrixXd::Zero(dim_design, dim);
	prec = Eigen::MatrixXd::Zero(dim, dim);
	yhat = Eigen::MatrixXd::Zero(num_design, dim);
	resid = Eigen::MatrixXd::Zero(num_design, dim);
	yhat_star = Eigen::MatrixXd::Zero(num_augment, dim);
	scale = Eigen::MatrixXd::Zero(dim, dim);
}

void Minnesota::estimateCoef() {
	prec = xstar.transpose() * xstar;
	coef = prec.inverse() * xstar.transpose() * ystar; // LLT or QR?
}

void Minnesota::fitObs() {
	yhat = design * coef;
	resid = response - yhat;
}

void Minnesota::estimateCov() {
	yhat_star = xstar * coef;
	scale = (ystar - yhat_star).transpose() * (ystar - yhat_star);
}

Rcpp::List Minnesota::returnMinnRes() {
	estimateCoef();
	fitObs();
	estimateCov();
	return Rcpp::List::create(
		Rcpp::Named("coefficients") = coef,
    Rcpp::Named("fitted.values") = yhat,
    Rcpp::Named("residuals") = resid,
		Rcpp::Named("mn_prec") = prec,
    Rcpp::Named("iw_scale") = scale,
		Rcpp::Named("iw_shape") = prior_shape + num_design,
		Rcpp::Named("df") = dim_design,
		Rcpp::Named("m") = dim,
		Rcpp::Named("obs") = num_design,
		Rcpp::Named("prior_mean") = prior_mean,
    Rcpp::Named("prior_precision") = prior_prec,
    Rcpp::Named("prior_scale") = prior_scale,
    Rcpp::Named("prior_shape") = prior_shape,
		Rcpp::Named("y0") = response,
		Rcpp::Named("design") = design
	);
}

MinnBvar::MinnBvar(const Eigen::MatrixXd& y, int lag, const BvarSpec& spec, const bool include_mean)
: lag(lag), const_term(include_mean),
	data(y), dim(data.cols()) {
	response = build_y0(data, lag, lag + 1);
	design = build_design(data, lag, const_term);
	dummy_response = build_ydummy(
		lag, spec._sigma,
		spec._lambda, spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
		const_term
	);
	dummy_design = build_xdummy(
		Eigen::VectorXd::LinSpaced(lag, 1, lag),
		spec._lambda, spec._sigma, spec._eps, const_term
	);
	_mn = std::unique_ptr<Minnesota>(new Minnesota(design, response, dummy_design, dummy_response));
}

Rcpp::List MinnBvar::returnMinnRes() {
	Rcpp::List mn_res = _mn->returnMinnRes();
	mn_res["p"] = lag;
	mn_res["totobs"] = data.rows();
	mn_res["type"] = const_term ? "const" : "none";
	mn_res["y"] = data;
	return mn_res;
}

MinnBvhar::MinnBvhar(const Eigen::MatrixXd& y, int week, int month, const MinnSpec& spec, const bool include_mean)
: week(week), month(month), const_term(include_mean),
	data(y), dim(data.cols()) {
	response = build_y0(data, month, month + 1);
	har_trans = scale_har(dim, week, month, const_term);
	var_design = build_design(data, month, const_term);
	design = var_design * har_trans.transpose();
	dummy_design = build_xdummy(
		Eigen::VectorXd::LinSpaced(3, 1, 3),
		spec._lambda, spec._sigma, spec._eps, const_term
	);
}

MinnBvharS::MinnBvharS(const Eigen::MatrixXd& y, int week, int month, const BvarSpec& spec, const bool include_mean)
: MinnBvhar(y, week, month, spec, include_mean) {
	dummy_response = build_ydummy(
		3, spec._sigma, spec._lambda,
		spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
		const_term
	);
	_mn = std::unique_ptr<Minnesota>(new Minnesota(design, response, dummy_design, dummy_response));
}

Rcpp::List MinnBvharS::returnMinnRes() {
	Rcpp::List mn_res = _mn->returnMinnRes();
	mn_res["p"] = 3;
	mn_res["week"] = week;
	mn_res["month"] = month;
	mn_res["totobs"] = data.rows();
	mn_res["type"] = const_term ? "const" : "none";
	mn_res["HARtrans"] = har_trans;
	mn_res["y"] = data;
	return mn_res;
}

MinnBvharL::MinnBvharL(const Eigen::MatrixXd& y, int week, int month, const BvharSpec& spec, const bool include_mean)
: MinnBvhar(y, week, month, spec, include_mean) {
	dummy_response = build_ydummy(
		3, spec._sigma, spec._lambda,
		spec._daily, spec._weekly, spec._monthly,
		const_term
	);
	_mn = std::unique_ptr<Minnesota>(new Minnesota(design, response, dummy_design, dummy_response));
}

Rcpp::List MinnBvharL::returnMinnRes() {
	Rcpp::List mn_res = _mn->returnMinnRes();
	mn_res["p"] = 3;
	mn_res["week"] = week;
	mn_res["month"] = month;
	mn_res["totobs"] = data.rows();
	mn_res["type"] = const_term ? "const" : "none";
	mn_res["HARtrans"] = har_trans;
	mn_res["y"] = data;
	return mn_res;
}
