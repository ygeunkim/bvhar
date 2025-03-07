#ifndef BVHAR_BAYES_MNIW_MINNESOTA_H
#define BVHAR_BAYES_MNIW_MINNESOTA_H

// #include <RcppEigen.h>
// #include "../misc/draw.h"
// #include "../../math/design.h"
#include "./config.h"
// #include <memory> // std::unique_ptr
#include "../../core/progress.h"

namespace bvhar {

class Minnesota;
class McmcMniw;
class MinnBvar;
class MinnBvhar;
class MinnBvharS;
class MinnBvharL;
class MhMinnesota;
class MinnFlat;

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
	LIST returnMinnRes() {
		estimateCoef();
		fitObs();
		estimateCov();
		return CREATE_LIST(
			// NAMED("mn_mean") = coef,
			NAMED("coefficients") = coef,
			NAMED("fitted.values") = yhat,
			NAMED("residuals") = resid,
			NAMED("mn_prec") = prec,
			NAMED("covmat") = scale,
			NAMED("iw_shape") = prior_shape + num_design,
			NAMED("df") = dim_design,
			NAMED("m") = dim,
			NAMED("obs") = num_design,
			NAMED("prior_mean") = prior_mean,
			NAMED("prior_precision") = prior_prec,
			NAMED("prior_scale") = prior_scale,
			NAMED("prior_shape") = prior_shape,
			NAMED("y0") = response,
			NAMED("design") = design
		);
	}
	MinnFit returnMinnFit() {
		estimateCoef();
		fitObs();
		estimateCov();
		MinnFit res(coef, prec, scale, prior_shape + num_design);
		return res;
	}
protected:
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

class McmcMniw {
public:
	McmcMniw(int num_iter, const MinnFit& mn_fit, unsigned int seed)
	: mn_fit(mn_fit),
		num_iter(num_iter), dim(mn_fit._coef.cols()), dim_design(mn_fit._coef.rows()),
		mn_record(num_iter, dim, dim_design),
		mniw(2), mcmc_step(0), rng(seed) {}
	virtual ~McmcMniw() = default;
	void addStep() { mcmc_step++; }
	void updateRecords() { mn_record.assignRecords(mcmc_step, mniw); }
	void updateMniw() { mniw = sim_mn_iw(mn_fit._coef, mn_fit._prec, mn_fit._iw_scale, mn_fit._iw_shape, true, rng); }
	void doPosteriorDraws() {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateMniw();
		updateRecords();
	}
	LIST returnRecords(int num_burn, int thin) const {
		LIST res = CREATE_LIST(
			NAMED("alpha_record") = mn_record.coef_record,
			NAMED("sigma_record") = mn_record.sig_record
		);
		for (auto& record : res) {
			ACCESS_LIST(record, res) = thin_record(CAST<Eigen::MatrixXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
		}
		return res;
	}
private:
	MinnFit mn_fit;
	int num_iter;
	int dim;
	int dim_design;
	MinnRecords mn_record;
	std::vector<Eigen::MatrixXd> mniw;
	std::atomic<int> mcmc_step; // MCMC step
	BHRNG rng; // RNG instance for multi-chain
	std::mutex mtx;
};

class MinnBvar {
public:
	MinnBvar(const Eigen::MatrixXd& y, int lag, const BvarSpec& spec, const bool include_mean)
	: lag(lag), const_term(include_mean), data(y), dim(data.cols()) {
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
		_mn.reset(new Minnesota(design, response, dummy_design, dummy_response));
	}
	MinnBvar(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, int lag, const BvarSpec& spec, const bool include_mean)
	: lag(lag), const_term(include_mean), dim(y.cols()),
		design(x), response(y) {
		dummy_response = build_ydummy(
			lag, spec._sigma,
			spec._lambda, spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
			const_term
		);
		dummy_design = build_xdummy(
			Eigen::VectorXd::LinSpaced(lag, 1, lag),
			spec._lambda, spec._sigma, spec._eps, const_term
		);
		_mn.reset(new Minnesota(design, response, dummy_design, dummy_response));
	}
	virtual ~MinnBvar() = default;
	LIST returnMinnRes() {
		LIST mn_res = _mn->returnMinnRes();
		mn_res["p"] = lag;
		mn_res["totobs"] = data.rows();
		mn_res["type"] = const_term ? "const" : "none";
		// mn_res["y"] = data;
		return mn_res;
	}
	MinnFit returnMinnFit() {
		return _mn->returnMinnFit();
	}
private:
	int lag;
	bool const_term;
	Eigen::MatrixXd data;
	int dim;
	Eigen::MatrixXd design;
	Eigen::MatrixXd response;
	Eigen::MatrixXd dummy_design;
	Eigen::MatrixXd dummy_response;
	std::unique_ptr<Minnesota> _mn;
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
	MinnBvhar(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, int week, int month, const MinnSpec& spec, const bool include_mean)
	: week(week), month(month), const_term(include_mean),
		dim(y.cols()), var_design(x), response(y) {
		// response = build_y0(data, month, month + 1);
		har_trans = bvhar::build_vhar(dim, week, month, const_term);
		// var_design = build_x0(data, month, const_term);
		design = var_design * har_trans.transpose();
		dummy_design = build_xdummy(
			Eigen::VectorXd::LinSpaced(3, 1, 3),
			spec._lambda, spec._sigma, spec._eps, const_term
		);
	}
	virtual ~MinnBvhar() = default;
	virtual LIST returnMinnRes() = 0;
	virtual MinnFit returnMinnFit() = 0;
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
		_mn.reset(new Minnesota(design, response, dummy_design, dummy_response));
	}
	MinnBvharS(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, int week, int month, const BvarSpec& spec, const bool include_mean)
	: MinnBvhar(x, y, week, month, spec, include_mean) {
		dummy_response = build_ydummy(
			3, spec._sigma, spec._lambda,
			spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
			const_term
		);
		_mn.reset(new Minnesota(design, response, dummy_design, dummy_response));
	}
	virtual ~MinnBvharS() noexcept = default;
	LIST returnMinnRes() override {
		LIST mn_res = _mn->returnMinnRes();
		mn_res["p"] = 3;
		mn_res["week"] = week;
		mn_res["month"] = month;
		mn_res["totobs"] = data.rows();
		mn_res["type"] = const_term ? "const" : "none";
		mn_res["HARtrans"] = har_trans;
		// mn_res["y"] = data;
		return mn_res;
	}
	MinnFit returnMinnFit() override {
		return _mn->returnMinnFit();
	}
private:
	Eigen::MatrixXd dummy_response;
	std::unique_ptr<Minnesota> _mn;
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
		_mn.reset(new Minnesota(design, response, dummy_design, dummy_response));
	}
	MinnBvharL(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, int week, int month, const BvharSpec& spec, const bool include_mean)
	: MinnBvhar(x, y, week, month, spec, include_mean) {
		dummy_response = build_ydummy(
			3, spec._sigma, spec._lambda,
			spec._daily, spec._weekly, spec._monthly,
			const_term
		);
		_mn.reset(new Minnesota(design, response, dummy_design, dummy_response));
	}
	virtual ~MinnBvharL() noexcept = default;
	LIST returnMinnRes() override {
		LIST mn_res = _mn->returnMinnRes();
		mn_res["p"] = 3;
		mn_res["week"] = week;
		mn_res["month"] = month;
		mn_res["totobs"] = data.rows();
		mn_res["type"] = const_term ? "const" : "none";
		mn_res["HARtrans"] = har_trans;
		// mn_res["y"] = data;
		return mn_res;
	}
	MinnFit returnMinnFit() override {
		return _mn->returnMinnFit();
	}
private:
	Eigen::MatrixXd dummy_response;
	std::unique_ptr<Minnesota> _mn;
};

class MhMinnesota : Minnesota {
public:
	MhMinnesota(
		int num_iter, const MhMinnSpec& spec, const MhMinnInits& inits, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		const Eigen::MatrixXd& x_dummy, const Eigen::MatrixXd& y_dummy, unsigned int seed
	)
	: Minnesota(x, y, x_dummy, y_dummy),
		num_iter(num_iter),
		mn_record(num_iter, dim, dim_design),
		mniw(2), mcmc_step(0), rng(seed),
		// Minnesota(num_iter, x, y, x_dummy, y_dummy, seed),
		mh_record(num_iter, dim),
		gamma_shp(spec._gam_rate), gamma_rate(spec._gam_shape),
		invgam_shp(spec._invgam_shape), invgam_scl(spec._invgam_scl), gaussian_variance(inits._acc_scale * inits._hess.inverse()),
		prevprior(Eigen::VectorXd::Zero(1 + dim)),
		candprior(Eigen::VectorXd::Zero(1 + dim)), numerator(0), denom(0),
		is_accept(true), lambda(inits._lambda), psi(inits._psi) {
		prevprior[0] = inits._lambda;
		prevprior.tail(dim) = inits._psi;
		mh_record.lam_record[0] = inits._lambda;
		mh_record.psi_record.row(0) = inits._psi;
		mh_record.accept_record[0] = is_accept;
	}
	virtual ~MhMinnesota() = default;
	void computePosterior() {
		estimateCoef();
		estimateCov();
	}
	void addStep() { mcmc_step++; }
	void updateRecords() {
		mn_record.assignRecords(mcmc_step, mniw); 
		mh_record.assignRecords(mcmc_step, lambda, psi, is_accept); 
	}
	void updateHyper() {
		candprior = Eigen::Map<Eigen::VectorXd>(sim_mgaussian_chol(1, prevprior, gaussian_variance, rng).data(), 1 + dim);
		numerator = jointdens_hyperparam(
			candprior[0], candprior.segment(1, dim), dim, num_design,
			prior_prec, prior_scale, prior_shape, prec, scale, prior_shape + num_design, gamma_shp, gamma_rate, invgam_shp, invgam_scl
		);
    denom = jointdens_hyperparam(
			prevprior[0], prevprior.segment(1, dim), dim, num_design,
			prior_prec, prior_scale, prior_shape, prec, scale, prior_shape + num_design, gamma_shp, gamma_rate, invgam_shp, invgam_scl
		);
		is_accept = ( log(unif_rand(rng)) < std::min(numerator - denom, 0.0) );
		if (is_accept) {
			lambda = candprior[0];
			psi = candprior.tail(dim);
		}
	}
	void updateMniw() { mniw = sim_mn_iw(coef, prec, scale, prior_shape + num_design, true, rng); }
	void doPosteriorDraws() {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateHyper();
		updateMniw();
		updateRecords();
	}
	LIST returnRecords(int num_burn, int thin) const {
		LIST res = CREATE_LIST(
			NAMED("lambda_record") = mh_record.lam_record,
			NAMED("psi_record") = mh_record.psi_record,
			NAMED("alpha_record") = mn_record.coef_record,
			NAMED("sigma_record") = mn_record.sig_record,
			// NAMED("accept_record") = thin_record(mn_record.accept_record, num_iter, num_burn, thin)
			NAMED("accept_record") = mh_record.accept_record
		);
		for (auto& record : res) {
			if (IS_MATRIX(ACCESS_LIST(record, res))) {
				ACCESS_LIST(record, res) = thin_record(CAST<Eigen::MatrixXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			} else if (IS_VECTOR(ACCESS_LIST(record, res))) {
				ACCESS_LIST(record, res) = thin_record(CAST<Eigen::VectorXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			} else if (IS_LOGICAL(ACCESS_LIST(record, res))) {
				ACCESS_LIST(record, res) = thin_record(CAST<VectorXb>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			}
		}
		return res;
	}
private:
	int num_iter;
	MinnRecords mn_record;
	std::vector<Eigen::MatrixXd> mniw;
	std::atomic<int> mcmc_step; // MCMC step
	BHRNG rng; // RNG instance for multi-chain
	std::mutex mtx;
	MhMinnRecords mh_record;
	double gamma_shp;
	double gamma_rate;
	double invgam_shp;
	double invgam_scl;
	Eigen::MatrixXd gaussian_variance;
	Eigen::VectorXd prevprior;
	Eigen::VectorXd candprior;
	double numerator;
  double denom;
	bool is_accept;
	double lambda;
	Eigen::VectorXd psi;
};

class MinnFlat {
public:
	// MinnFlat(int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::MatrixXd& prec, unsigned int seed)
	MinnFlat(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::MatrixXd& prec)
	: design(x), response(y), prior_prec(prec),
		// num_iter(num_iter), dim(response.cols()), num_design(response.rows()), dim_design(design.cols()),
		dim(response.cols()), num_design(response.rows()), dim_design(design.cols()),
		// mn_record(num_iter, dim, dim_design),
		// mniw(2), mcmc_step(0), rng(seed),
		coef(Eigen::MatrixXd::Zero(dim_design, dim)),
		prec(Eigen::MatrixXd::Zero(dim, dim)),
		shape(0.0),
		yhat(Eigen::MatrixXd::Zero(num_design, dim)),
		resid(Eigen::MatrixXd::Zero(num_design, dim)) {
		// prior_prec = dummy_design.transpose() * dummy_design;
		// prior_mean = prior_prec.inverse() * dummy_design.transpose() * dummy_response;
		// prior_scale = (dummy_response - dummy_design * prior_mean).transpose() * (dummy_response - dummy_design * prior_mean);
		// prior_shape = num_dummy - dim_design + 2;
		// ystar = Eigen::MatrixXd::Zero(num_augment, dim);
		// ystar << y,
		// 				y_dummy;
		// xstar = Eigen::MatrixXd::Zero(num_augment, dim_design);
		// xstar << x,
		// 				x_dummy;
		// coef = Eigen::MatrixXd::Zero(dim_design, dim);
		// prec = Eigen::MatrixXd::Zero(dim, dim);
		// yhat = Eigen::MatrixXd::Zero(num_design, dim);
		// resid = Eigen::MatrixXd::Zero(num_design, dim);
		// yhat_star = Eigen::MatrixXd::Zero(num_augment, dim);
		// scale = Eigen::MatrixXd::Zero(dim, dim);
	}
	virtual ~MinnFlat() = default;
	void estimateCoef() {
		prec = (design.transpose() * design + prior_prec);
		coef = prec.inverse() * design.transpose() * response;
	}
	virtual void fitObs() {
		yhat = design * coef;
		resid = response - yhat;
	}
	void estimateCov() { // Posterior IW scale
		scale = response.transpose() * (Eigen::MatrixXd::Identity(num_design, num_design) - design * prec.inverse() * design.transpose()) * response;
		shape = num_design - dim - 1;
	}
	// virtual void computePosterior() {
	// 	estimateCoef();
	// 	fitObs();
	// 	estimateCov();
	// }
	// void addStep() { mcmc_step++; }
	// virtual void updateRecords() { mn_record.assignRecords(mcmc_step, mniw); }
	// void updateMniw() { mniw = sim_mn_iw(coef, prec, scale, shape, true, rng); }
	// virtual void doPosteriorDraws() {
	// 	std::lock_guard<std::mutex> lock(mtx);
	// 	addStep();
	// 	updateMniw();
	// 	updateRecords();
	// }
	// Rcpp::List returnRecords(int num_burn, int thin) const {
	// 	Rcpp::List res = CREATE_LIST(
	// 		NAMED("alpha_record") = mn_record.coef_record,
	// 		NAMED("sigma_record") = mn_record.sig_record
	// 	);
	// 	for (auto& record : res) {
	// 		record = thin_record(CAST<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
	// 	}
	// 	return res;
	// }
	LIST returnMinnRes() {
		estimateCoef();
		fitObs();
		estimateCov();
		return CREATE_LIST(
			// NAMED("mn_mean") = coef,
			NAMED("coefficients") = coef,
			NAMED("fitted.values") = yhat,
			NAMED("residuals") = resid,
			NAMED("mn_prec") = prec,
			// NAMED("iw_scale") = scale,
			NAMED("covmat") = scale,
			NAMED("iw_shape") = shape,
			NAMED("df") = dim_design,
			NAMED("m") = dim,
			NAMED("obs") = num_design,
			NAMED("prior_mean") = Eigen::MatrixXd::Zero(dim_design, dim),
			NAMED("prior_precision") = prior_prec,
			// NAMED("prior_scale") = prior_scale,
			// NAMED("prior_shape") = prior_shape,
			NAMED("y0") = response,
			NAMED("design") = design
		);
	}
	// MinnRecords returnMinnRecords(int num_burn, int thin) const {
	// 	MinnRecords res_record(
	// 		thin_record(mn_record.coef_record, num_iter, num_burn, thin).derived(),
	// 		thin_record(mn_record.sig_record, num_iter, num_burn, thin).derived()
	// 	);
	// 	return res_record;
	// }
	MinnFit returnMinnFit() {
		estimateCoef();
		fitObs();
		estimateCov();
		MinnFit res(coef, prec, scale, shape);
		return res;
	}
protected:
	Eigen::MatrixXd design;
	Eigen::MatrixXd response;
	Eigen::MatrixXd prior_prec;
	// int num_iter;
	int dim;
	int num_design;
	int dim_design;
	// int num_dummy; // kp + k(+ 1)
	// int num_augment; // n + num_dummy
	// MinnRecords mn_record;
	// std::vector<Eigen::MatrixXd> mniw;
	// std::atomic<int> mcmc_step; // MCMC step
	// boost::random::mt19937 rng; // RNG instance for multi-chain
	// std::mutex mtx;
	// Eigen::MatrixXd prior_prec;
	Eigen::MatrixXd prior_mean;
	// Eigen::MatrixXd prior_scale;
	// int prior_shape;
	// Eigen::MatrixXd ystar; // [Y0, Yp]
	// Eigen::MatrixXd xstar; // [X0, Xp]
	Eigen::MatrixXd coef; // MN mean
	Eigen::MatrixXd prec; // MN precision
	double shape;
	Eigen::MatrixXd yhat;
	Eigen::MatrixXd resid;
	// Eigen::MatrixXd yhat_star;
	Eigen::MatrixXd scale; // IW scale
};

} // namespace bvhar

#endif // BVHAR_BAYES_MNIW_MINNESOTA_H