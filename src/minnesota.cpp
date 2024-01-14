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

HierMinnSpec::HierMinnSpec(Rcpp::List& bayes_spec)
: acc_scale(bayes_spec["acc_scale"]),
	obs_information(Rcpp::as<Eigen::MatrixXd>(bayes_spec["obs_information"])) {
	Rcpp::List spec = bayes_spec["lambda"];
	Rcpp::NumericVector param = spec["param"];
	gamma_shp = param[0];
	gamma_rate = param[1];
	spec = bayes_spec["sigma"];
	param = spec["param"];
	invgam_shp = param[0];
	invgam_scl = param[1];
}

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
	shape = prior_shape + num_design;
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
		Rcpp::Named("iw_shape") = shape,
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

HierMinn::HierMinn(
	int num_iter,
	const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::MatrixXd& x_dummy, const Eigen::MatrixXd& y_dummy,
	const HierMinnSpec& spec, const MinnSpec& init
)
: Minnesota(x, y, x_dummy, y_dummy),
	num_iter(num_iter), is_accept(true),
	lambda(init._lambda), psi(init._sigma),
	numerator(0), denom(0),
	gamma_shp(spec.gamma_shp), gamma_rate(spec.gamma_rate),
	invgam_shp(spec.invgam_shp), invgam_scl(spec.invgam_scl),
	mcmc_step(0) {
	gaussian_variance = spec.acc_scale * spec.obs_information.inverse();
	accept_record = VectorXb::Constant(num_iter + 1, true);
	accept_record[0] = true;
	prevprior = Eigen::VectorXd::Zero(1 + dim);
	candprior = Eigen::VectorXd::Zero(1 + dim);
	coef_and_sig = Rcpp::List::create(
		Rcpp::Named("mn") = Eigen::MatrixXd::Zero(dim_design, dim),
    Rcpp::Named("iw") = Eigen::MatrixXd::Zero(dim, dim)
	);
	prevprior[0] = lambda;
	prevprior.tail(dim) = psi;
	lam_record = Eigen::VectorXd::Zero(num_iter + 1);
	psi_record = Eigen::MatrixXd::Zero(num_iter + 1, dim);
	coef_record = Eigen::MatrixXd::Zero((num_iter + 1), dim * dim_design);
	sig_record = Eigen::MatrixXd::Zero(dim * (num_iter + 1), dim);
	lam_record[0] = lambda;
	psi_record.row(0) = psi;
	estimateCoef(); // Posterior MN
	estimateCov(); // Posterior IW
}

void HierMinn::updateHyper() {
	candprior = vectorize_eigen(sim_mgaussian_chol(1, prevprior, gaussian_variance)); // Candidate ~ N(previous, scaled hessian)
	numerator = jointdens_hyperparam(
		candprior[0], candprior.tail(dim), dim, num_design,
		prior_prec, prior_scale, prior_shape,
		prec, scale, shape, gamma_shp, gamma_rate, invgam_shp, invgam_scl
	);
	denom = jointdens_hyperparam(
		prevprior[0], prevprior.tail(dim), dim, num_design,
		prior_prec, prior_scale, prior_shape,
		prec, scale, shape, gamma_shp, gamma_rate, invgam_shp, invgam_scl
	);
	is_accept = ( log(unif_rand(0, 1)) < std::min(numerator - denom, 0.0) ); // log of acceptance rate = numerator - denom
	if (is_accept) {
		lambda = candprior[0];
    psi = candprior.tail(dim);
		prevprior[0] = lambda;
		prevprior.tail(dim) = psi;
	}
	accept_record[mcmc_step] = is_accept;
	lam_record[mcmc_step] = lambda;
	psi_record.row(mcmc_step) = psi;
}

void HierMinn::updateMniw() {
	coef_and_sig = sim_mniw(1, coef, prec.inverse(), scale, shape);
	coef_record.row(mcmc_step) = vectorize_eigen(coef_and_sig["mn"]);
	sig_record.block(mcmc_step * dim, 0, dim, dim) = Rcpp::as<Eigen::MatrixXd>(coef_and_sig["iw"]);
}

void HierMinn::addStep() {
	mcmc_step++;
}

void HierMinn::doPosteriorDraws() {
	updateHyper();
	updateMniw();
}

void HierMinn::estimatePosterior() {
	coef = coef_record.colwise().mean();
}

Rcpp::List HierMinn::returnRecords(int num_burn) const {
	return Rcpp::List::create(
		Rcpp::Named("lambda_record") = lam_record.tail(num_iter - num_burn),
    Rcpp::Named("psi_record") = psi_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("sigma_record") = sig_record.bottomRows(dim * (num_iter - num_burn)),
    Rcpp::Named("acceptance") = accept_record.tail(num_iter - num_burn)
	);
}

Rcpp::List HierMinn::returnMinnRes(int num_burn) {
	estimatePosterior();
	Rcpp::List record_res = returnRecords(num_burn);
	record_res["coefficients"] = coef;
	record_res["df"] = dim_design;
	record_res["m"] = dim;
	record_res["obs"] = num_design;
	record_res["y0"] = response;
	record_res["design"] = design;
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

HierBvar::HierBvar(int num_iter, const Eigen::MatrixXd& y, int lag, const HierMinnSpec& spec, const BvarSpec& init, const bool include_mean)
: MinnBvar(y, lag, init, include_mean) {
	_mn = std::unique_ptr<HierMinn>(new HierMinn(num_iter, design, response, dummy_design, dummy_response, spec, init));
}

Rcpp::List HierBvar::returnMinnRes(int num_burn) {
	_mn->doPosteriorDraws();
	Rcpp::List mn_res = _mn->returnMinnRes(num_burn);
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
