#ifndef MINNESOTA_H
#define MINNESOTA_H

// #include <RcppEigen.h>
#include "bvhardesign.h"
#include <memory> // std::unique_ptr

namespace bvhar {

struct MinnSpec {
	Eigen::VectorXd _sigma;
	double _lambda;
	double _eps;

	MinnSpec(Rcpp::List& bayes_spec)
	: _sigma(Rcpp::as<Eigen::VectorXd>(bayes_spec["sigma"])),
		_lambda(bayes_spec["lambda"]),
		_eps(bayes_spec["eps"]) {}
};

struct BvarSpec : public MinnSpec {
	Eigen::VectorXd _delta;

	BvarSpec(Rcpp::List& bayes_spec)
	: MinnSpec(bayes_spec),
		_delta(Rcpp::as<Eigen::VectorXd>(bayes_spec["delta"])) {}
};

struct BvharSpec : public MinnSpec {
	Eigen::VectorXd _daily;
	Eigen::VectorXd _weekly;
	Eigen::VectorXd _monthly;

	BvharSpec(Rcpp::List& bayes_spec)
	: MinnSpec(bayes_spec),
		_daily(Rcpp::as<Eigen::VectorXd>(bayes_spec["daily"])),
		_weekly(Rcpp::as<Eigen::VectorXd>(bayes_spec["weekly"])),
		_monthly(Rcpp::as<Eigen::VectorXd>(bayes_spec["monthly"])) {}
};

class Minnesota {
public:
	Minnesota(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::MatrixXd& x_dummy, const Eigen::MatrixXd& y_dummy)
	: design(x), response(y), dummy_design(x_dummy), dummy_response(y_dummy),
		dim(response.cols()), num_design(response.rows()), dim_design(design.cols()),
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
	virtual ~Minnesota() = default;
	void estimateCoef() {
		prec = xstar.transpose() * xstar;
		coef = prec.inverse() * xstar.transpose() * ystar; // LLT or QR?
	}
	virtual void fitObs() {
		yhat = design * coef;
		resid = response - yhat;
	}
	void estimateCov() { // Posterior IW scale
		yhat_star = xstar * coef;
		scale = (ystar - yhat_star).transpose() * (ystar - yhat_star);
	}
	Rcpp::List returnMinnRes() {
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
private:
	Eigen::MatrixXd design;
	Eigen::MatrixXd response;
	Eigen::MatrixXd dummy_design;
	Eigen::MatrixXd dummy_response;
	int dim;
	int num_design;
	int dim_design;
	int num_dummy; // kp + k(+ 1)
	int num_augment; // n + num_dummy
	Eigen::MatrixXd prior_prec;
	Eigen::MatrixXd prior_mean;
	Eigen::MatrixXd prior_scale;
	int prior_shape;
	Eigen::MatrixXd ystar; // [Y0, Yp]
	Eigen::MatrixXd xstar; // [X0, Xp]
	Eigen::MatrixXd coef; // MN mean
	Eigen::MatrixXd prec; // MN precision
	Eigen::MatrixXd yhat;
	Eigen::MatrixXd resid;
	Eigen::MatrixXd yhat_star;
	Eigen::MatrixXd scale; // IW scale
};

class MinnBvar {
public:
	MinnBvar(const Eigen::MatrixXd& y, int lag, const BvarSpec& spec, const bool include_mean)
	: lag(lag), const_term(include_mean),
		data(y), dim(data.cols()) {
		response = build_y0(data, lag, lag + 1);
		design = build_x0(data, lag, const_term);
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
	virtual ~MinnBvar() = default;
	Rcpp::List returnMinnRes() {
		Rcpp::List mn_res = _mn->returnMinnRes();
		mn_res["p"] = lag;
		mn_res["totobs"] = data.rows();
		mn_res["type"] = const_term ? "const" : "none";
		mn_res["y"] = data;
		return mn_res;
	}
private:
	int lag;
	bool const_term;
	Eigen::MatrixXd data;
	int dim;
	std::unique_ptr<Minnesota> _mn;
	Eigen::MatrixXd design;
	Eigen::MatrixXd response;
	Eigen::MatrixXd dummy_design;
	Eigen::MatrixXd dummy_response;
};

class MinnBvhar {
public:
	MinnBvhar(const Eigen::MatrixXd& y, int week, int month, const MinnSpec& spec, const bool include_mean)
	: week(week), month(month), const_term(include_mean),
		data(y), dim(data.cols()) {
		response = build_y0(data, month, month + 1);
		har_trans = bvhar::build_vhar(dim, week, month, const_term);
		var_design = build_x0(data, month, const_term);
		design = var_design * har_trans.transpose();
		dummy_design = build_xdummy(
			Eigen::VectorXd::LinSpaced(3, 1, 3),
			spec._lambda, spec._sigma, spec._eps, const_term
		);
	}
	virtual ~MinnBvhar() = default;
	virtual Rcpp::List returnMinnRes() = 0;
protected:
	int week;
	int month;
	bool const_term;
	Eigen::MatrixXd data;
	int dim;
	Eigen::MatrixXd var_design;
	Eigen::MatrixXd response;
	Eigen::MatrixXd har_trans;
	Eigen::MatrixXd design;
	Eigen::MatrixXd dummy_design;
};

class MinnBvharS : public MinnBvhar {
public:
	MinnBvharS(const Eigen::MatrixXd& y, int week, int month, const BvarSpec& spec, const bool include_mean)
	: MinnBvhar(y, week, month, spec, include_mean) {
		dummy_response = build_ydummy(
			3, spec._sigma, spec._lambda,
			spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
			const_term
		);
		_mn = std::unique_ptr<Minnesota>(new Minnesota(design, response, dummy_design, dummy_response));
	}
	virtual ~MinnBvharS() noexcept = default;
	Rcpp::List returnMinnRes() override {
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
private:
	std::unique_ptr<Minnesota> _mn;
	Eigen::MatrixXd dummy_response;
};

class MinnBvharL : public MinnBvhar {
public:
	MinnBvharL(const Eigen::MatrixXd& y, int week, int month, const BvharSpec& spec, const bool include_mean)
	: MinnBvhar(y, week, month, spec, include_mean) {
		dummy_response = build_ydummy(
			3, spec._sigma, spec._lambda,
			spec._daily, spec._weekly, spec._monthly,
			const_term
		);
		_mn = std::unique_ptr<Minnesota>(new Minnesota(design, response, dummy_design, dummy_response));
	}
	virtual ~MinnBvharL() noexcept = default;
	Rcpp::List returnMinnRes() override {
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
private:
	std::unique_ptr<Minnesota> _mn;
	Eigen::MatrixXd dummy_response;
};

} // namespace bvhar

#endif // MINNESOTA_H