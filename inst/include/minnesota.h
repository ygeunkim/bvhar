#ifndef MINNESOTA_H
#define MINNESOTA_H

// #include <RcppEigen.h>
#include "bvhardesign.h"
// #include <memory> // std::unique_ptr
#include "bvhardraw.h"
#include "bvharprogress.h"

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

struct MinnFit {
	Eigen::MatrixXd _coef;
	Eigen::MatrixXd _prec;
	Eigen::MatrixXd _iw_scale;
	double _iw_shape;
	
	MinnFit(const Eigen::MatrixXd& coef_mat, const Eigen::MatrixXd& prec_mat, const Eigen::MatrixXd& iw_scale, double iw_shape)
	: _coef(coef_mat), _prec(prec_mat), _iw_scale(iw_scale), _iw_shape(iw_shape) {}
};

struct MhMinnInits {
	double _lambda;
	Eigen::VectorXd _psi;
	Eigen::MatrixXd _hess;
	double _acc_scale;
	// Eigen::VectorXd _delta;

	MhMinnInits(Rcpp::List& init) {
		Eigen::VectorXd par = Rcpp::as<Eigen::VectorXd>(init["par"]);
		_lambda = par[0];
		_psi = par.tail(par.size() - 1);
		_hess = Rcpp::as<Eigen::MatrixXd>(init["hessian"]);
		_acc_scale = init["scale_variance"];
		// _delta = Rcpp::as<Eigen::VectorXd>(spec["delta"]);
	}
};

struct MhMinnSpec {
	double _gam_shape;
	double _gam_rate;
	double _invgam_shape;
	double _invgam_scl;

	MhMinnSpec(Rcpp::List& lambda, Rcpp::List& psi) {
		// Rcpp::List lambda = spec["lambda"];
		// Rcpp::List psi = spec["sigma"];
		// Rcpp::List param = lambda["param"];
		Eigen::VectorXd lam_param = Rcpp::as<Eigen::VectorXd>(lambda["param"]);
		_gam_shape = lam_param[0];
		_gam_rate = lam_param[1];
		Eigen::VectorXd psi_param = Rcpp::as<Eigen::VectorXd>(psi["param"]);
		_invgam_shape = psi_param[0];
		_invgam_scl = psi_param[1];
	}
};

struct MinnRecords {
	Eigen::MatrixXd coef_record; // alpha in VAR
	Eigen::MatrixXd sig_record;

	MinnRecords(int num_iter, int dim, int dim_design)
	: coef_record(Eigen::MatrixXd::Zero(num_iter + 1, dim * dim_design)),
		sig_record(Eigen::MatrixXd::Zero(num_iter + 1, dim * dim)) {}
	
	MinnRecords(const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& sig_record)
	: coef_record(alpha_record), sig_record(sig_record) {}
	
	void assignRecords(int id, std::vector<Eigen::MatrixXd>& mniw_draw) {
		coef_record.row(id) = vectorize_eigen(mniw_draw[0]);
		sig_record.row(id) = vectorize_eigen(mniw_draw[1]);
	}
};

struct MhMinnRecords {
	// Eigen::MatrixXd coef_record; // alpha in VAR
	// Eigen::MatrixXd sig_record;
	Eigen::VectorXd lam_record;
	Eigen::MatrixXd psi_record;
	VectorXb accept_record;
	// int _dim;
	
	// MhMinnRecords(int num_iter, int dim, int dim_design)
	MhMinnRecords(int num_iter, int dim)
	// : coef_record(Eigen::MatrixXd::Zero(num_iter + 1, dim * dim_design)),
	// 	sig_record(Eigen::MatrixXd::Zero(num_iter + 1, dim * dim)),
	: lam_record(Eigen::VectorXd::Zero(num_iter + 1)),
		psi_record(Eigen::MatrixXd::Zero(num_iter + 1, dim)),
		accept_record(VectorXb(num_iter + 1)) {}
	void assignRecords(
		int id,
		// std::vector<Eigen::MatrixXd>& mniw_draw,
		double lambda, Eigen::Ref<Eigen::VectorXd> psi, bool is_accept
	) {
		// coef_record.row(id) = vectorize_eigen(mniw_draw[0]);
		// sig_record.row(id) = vectorize_eigen(mniw_draw[1]);
		lam_record[id] = lambda;
		psi_record.row(id) = psi;
		accept_record[id] = is_accept;
	}
};

class Minnesota {
public:
	Minnesota(
		int num_iter,
		const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::MatrixXd& x_dummy, const Eigen::MatrixXd& y_dummy,
		unsigned int seed
	)
	: design(x), response(y), dummy_design(x_dummy), dummy_response(y_dummy),
		num_iter(num_iter),
		dim(response.cols()), num_design(response.rows()), dim_design(design.cols()),
		num_dummy(dummy_design.rows()), num_augment(num_design + num_dummy),
		mn_record(num_iter, dim, dim_design),
		mniw(2), mcmc_step(0), rng(seed) {
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
	virtual void computePosterior() {
		estimateCoef();
		fitObs();
		estimateCov();
	}
	void addStep() { mcmc_step++; }
	virtual void updateRecords() { mn_record.assignRecords(mcmc_step, mniw); }
	void updateMniw() { mniw = sim_mn_iw(coef, prec, scale, prior_shape + num_design, true, rng); }
	virtual void doPosteriorDraws() {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateMniw();
		updateRecords();
	}
	virtual Rcpp::List returnRecords(int num_burn, int thin) const {
		Rcpp::List res = Rcpp::List::create(
			Rcpp::Named("alpha_record") = mn_record.coef_record,
			Rcpp::Named("sigma_record") = mn_record.sig_record
		);
		for (auto& record : res) {
			record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
		}
		return res;
	}
	Rcpp::List returnMinnRes() {
		// estimateCoef();
		// fitObs();
		// estimateCov();
		return Rcpp::List::create(
			Rcpp::Named("mn_mean") = coef,
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
	MinnFit returnMinnFit() {
		estimateCoef();
		fitObs();
		estimateCov();
		MinnFit res(coef, prec, scale, prior_shape + num_design);
		return res;
	}
	MinnRecords returnMinnRecords(int num_burn, int thin) const {
		MinnRecords res_record(
			thin_record(mn_record.coef_record, num_iter, num_burn, thin).derived(),
			thin_record(mn_record.sig_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
protected:
	Eigen::MatrixXd design;
	Eigen::MatrixXd response;
	Eigen::MatrixXd dummy_design;
	Eigen::MatrixXd dummy_response;
	int num_iter;
	int dim;
	int num_design;
	int dim_design;
	int num_dummy; // kp + k(+ 1)
	int num_augment; // n + num_dummy
	MinnRecords mn_record;
	std::vector<Eigen::MatrixXd> mniw;
	std::atomic<int> mcmc_step; // MCMC step
	boost::random::mt19937 rng; // RNG instance for multi-chain
	std::mutex mtx;
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

// class MinnBvar {
// public:
// 	MinnBvar(
// 		const Eigen::MatrixXd& y, int lag,
// 		int num_chains, int num_iter, const BvarSpec& spec, const bool include_mean,
// 		bool display_progress, Eigen::VectorXi seed_chain
// 	)
// 	: lag(lag), const_term(include_mean),
// 		data(y), dim(data.cols()),
// 		num_chains(num_chains), num_iter(num_iter), progress(display_progress),
// 		_records(num_chains) {
// 		response = build_y0(data, lag, lag + 1);
// 		design = build_x0(data, lag, const_term);
// 		dummy_response = build_ydummy(
// 			lag, spec._sigma,
// 			spec._lambda, spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
// 			const_term
// 		);
// 		dummy_design = build_xdummy(
// 			Eigen::VectorXd::LinSpaced(lag, 1, lag),
// 			spec._lambda, spec._sigma, spec._eps, const_term
// 		);
// 		// _mn.reset(new Minnesota(num_iter, design, response, dummy_design, dummy_response, seed));
// 		for (int i = 0; i < num_chains; ++i) {
// 			_mn[i].reset(new Minnesota(
// 				num_iter, design, response, dummy_design, dummy_response,
// 				static_cast<unsigned int>(seed_chain[i])
// 			));
// 			_mn[i]->computePosterior();
// 		}
// 	}
// 	virtual ~MinnBvar() = default;
// 	void doPosteriorDraws(int nthreads, int num_burn, int thin) {
// 		auto run_conj = [&](int chain) {
// 			bvhar::bvharprogress bar(num_iter, progress);
// 			// bvhar::bvharinterrupt();
// 			for (int i = 0; i < num_iter; i++) {
// 				// if (bvhar::bvharinterrupt::is_interrupted()) {
// 				// #ifdef _OPENMP
// 				// 	#pragma omp critical
// 				// #endif
// 				// 	{
// 				// 		res[chain] = mn_objs[chain]->returnRecords(0, 1);
// 				// 	}
// 				// 	break;
// 				// }
// 				bar.increment();
// 				bar.update();
// 				_mn[chain]->doPosteriorDraws();
// 			}
// 		#ifdef _OPENMP
// 			#pragma omp critical
// 		#endif
// 			{
// 				_records[chain] = _mn[chain]->returnRecords(num_burn, thin);
// 			}
// 		};
// 		if (num_chains == 1) {
// 			run_conj(0);
// 		} else {
// 		#ifdef _OPENMP
// 			#pragma omp parallel for num_threads(nthreads)
// 		#endif
// 			for (int chain = 0; chain < num_chains; chain++) {
// 				run_conj(chain);
// 			}
// 		}
// 	}
// 	Rcpp::List returnMinnRes() {
// 		// Rcpp::List mn_res = _mn->returnMinnRes();
// 		Rcpp::List mn_res = _mn[0]->returnMinnRes();
// 		mn_res["p"] = lag;
// 		mn_res["totobs"] = data.rows();
// 		mn_res["type"] = const_term ? "const" : "none";
// 		mn_res["y"] = data;
// 		mn_res.push_back(_records);
// 		return mn_res;
// 	}
// 	MinnFit returnMinnFit() {
// 		// return _mn->returnMinnFit();
// 		return _mn[0]->returnMinnFit();
// 	}
// private:
// 	int lag;
// 	bool const_term;
// 	Eigen::MatrixXd data;
// 	int dim;
// 	int num_chains;
// 	int num_iter;
// 	bool progress;
// 	std::vector<Rcpp::List> _records;
// 	// std::unique_ptr<Minnesota> _mn;
// 	std::vector<std::unique_ptr<Minnesota>> _mn;
// 	Eigen::MatrixXd design;
// 	Eigen::MatrixXd response;
// 	Eigen::MatrixXd dummy_design;
// 	Eigen::MatrixXd dummy_response;
// };

// class MinnBvhar {
// public:
// 	MinnBvhar(const Eigen::MatrixXd& y, int week, int month, const MinnSpec& spec, const bool include_mean)
// 	: week(week), month(month), const_term(include_mean),
// 		data(y), dim(data.cols()) {
// 		response = build_y0(data, month, month + 1);
// 		har_trans = bvhar::build_vhar(dim, week, month, const_term);
// 		var_design = build_x0(data, month, const_term);
// 		design = var_design * har_trans.transpose();
// 		dummy_design = build_xdummy(
// 			Eigen::VectorXd::LinSpaced(3, 1, 3),
// 			spec._lambda, spec._sigma, spec._eps, const_term
// 		);
// 	}
// 	virtual ~MinnBvhar() = default;
// 	virtual void doPosteriorDraws(int nthreads, int num_burn, int thin) = 0;
// 	virtual Rcpp::List returnMinnRes() = 0;
// 	virtual MinnFit returnMinnFit() = 0;
// protected:
// 	int week;
// 	int month;
// 	bool const_term;
// 	Eigen::MatrixXd data;
// 	int dim;
// 	Eigen::MatrixXd var_design;
// 	Eigen::MatrixXd response;
// 	Eigen::MatrixXd har_trans;
// 	Eigen::MatrixXd design;
// 	Eigen::MatrixXd dummy_design;
// };

// class MinnBvharS : public MinnBvhar {
// public:
// 	MinnBvharS(
// 		const Eigen::MatrixXd& y,
// 		int week, int month,
// 		int num_chains, int num_iter, const BvarSpec& spec, const bool include_mean,
// 		bool display_progress, Eigen::VectorXi seed_chain
// 	)
// 	: MinnBvhar(y, week, month, spec, include_mean),
// 		num_chains(num_chains), num_iter(num_iter), progress(display_progress),
// 		_records(num_chains) {
// 		dummy_response = build_ydummy(
// 			3, spec._sigma, spec._lambda,
// 			spec._delta, Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim),
// 			const_term
// 		);
// 		// _mn.reset(new Minnesota(num_iter, design, response, dummy_design, dummy_response));
// 		for (int i = 0; i < num_chains; ++i) {
// 			_mn[i].reset(new Minnesota(
// 				num_iter, design, response, dummy_design, dummy_response,
// 				static_cast<unsigned int>(seed_chain[i])
// 			));
// 			_mn[i]->computePosterior();
// 		}
// 	}
// 	virtual ~MinnBvharS() noexcept = default;
// 	void doPosteriorDraws(int nthreads, int num_burn, int thin) override {
// 		auto run_conj = [&](int chain) {
// 			bvhar::bvharprogress bar(num_iter, progress);
// 			for (int i = 0; i < num_iter; i++) {
// 				bar.increment();
// 				bar.update();
// 				_mn[chain]->doPosteriorDraws();
// 			}
// 		#ifdef _OPENMP
// 			#pragma omp critical
// 		#endif
// 			{
// 				_records[chain] = _mn[chain]->returnRecords(num_burn, thin);
// 			}
// 		};
// 		if (num_chains == 1) {
// 			run_conj(0);
// 		} else {
// 		#ifdef _OPENMP
// 			#pragma omp parallel for num_threads(nthreads)
// 		#endif
// 			for (int chain = 0; chain < num_chains; chain++) {
// 				run_conj(chain);
// 			}
// 		}
// 	}
// 	Rcpp::List returnMinnRes() override {
// 		// Rcpp::List mn_res = _mn->returnMinnRes();
// 		Rcpp::List mn_res = _mn[0]->returnMinnRes();
// 		mn_res["p"] = 3;
// 		mn_res["week"] = week;
// 		mn_res["month"] = month;
// 		mn_res["totobs"] = data.rows();
// 		mn_res["type"] = const_term ? "const" : "none";
// 		mn_res["HARtrans"] = har_trans;
// 		mn_res["y"] = data;
// 		mn_res.push_back(_records);
// 		return mn_res;
// 	}
// 	MinnFit returnMinnFit() override {
// 		// return _mn->returnMinnFit();
// 		return _mn[0]->returnMinnFit();
// 	}
// private:
// 	int num_chains;
// 	int num_iter;
// 	bool progress;
// 	std::vector<Rcpp::List> _records;
// 	// std::unique_ptr<Minnesota> _mn;
// 	Eigen::MatrixXd dummy_response;
// 	std::vector<std::unique_ptr<Minnesota>> _mn;
// };

// class MinnBvharL : public MinnBvhar {
// public:
// 	MinnBvharL(
// 		const Eigen::MatrixXd& y,
// 		int week, int month,
// 		int num_chains, int num_iter, const BvharSpec& spec, const bool include_mean,
// 		bool display_progress, Eigen::VectorXi seed_chain
// 	)
// 	: MinnBvhar(y, week, month, spec, include_mean),
// 		num_chains(num_chains), num_iter(num_iter), progress(display_progress),
// 		_records(num_chains) {
// 		dummy_response = build_ydummy(
// 			3, spec._sigma, spec._lambda,
// 			spec._daily, spec._weekly, spec._monthly,
// 			const_term
// 		);
// 		// _mn.reset(new Minnesota(num_iter, design, response, dummy_design, dummy_response));
// 		for (int i = 0; i < num_chains; ++i) {
// 			_mn[i].reset(new Minnesota(
// 				num_iter, design, response, dummy_design, dummy_response,
// 				static_cast<unsigned int>(seed_chain[i])
// 			));
// 			_mn[i]->computePosterior();
// 		}
// 	}
// 	virtual ~MinnBvharL() noexcept = default;
// 	void doPosteriorDraws(int nthreads, int num_burn, int thin) override {
// 		auto run_conj = [&](int chain) {
// 			bvhar::bvharprogress bar(num_iter, progress);
// 			for (int i = 0; i < num_iter; i++) {
// 				bar.increment();
// 				bar.update();
// 				_mn[chain]->doPosteriorDraws();
// 			}
// 		#ifdef _OPENMP
// 			#pragma omp critical
// 		#endif
// 			{
// 				_records[chain] = _mn[chain]->returnRecords(num_burn, thin);
// 			}
// 		};
// 		if (num_chains == 1) {
// 			run_conj(0);
// 		} else {
// 		#ifdef _OPENMP
// 			#pragma omp parallel for num_threads(nthreads)
// 		#endif
// 			for (int chain = 0; chain < num_chains; chain++) {
// 				run_conj(chain);
// 			}
// 		}
// 	}
// 	Rcpp::List returnMinnRes() override {
// 		// Rcpp::List mn_res = _mn->returnMinnRes();
// 		Rcpp::List mn_res = _mn[0]->returnMinnRes();
// 		mn_res["p"] = 3;
// 		mn_res["week"] = week;
// 		mn_res["month"] = month;
// 		mn_res["totobs"] = data.rows();
// 		mn_res["type"] = const_term ? "const" : "none";
// 		mn_res["HARtrans"] = har_trans;
// 		mn_res["y"] = data;
// 		mn_res.push_back(_records);
// 		return mn_res;
// 	}
// 	MinnFit returnMinnFit() override {
// 		// return _mn->returnMinnFit();
// 		return _mn[0]->returnMinnFit();
// 	}
// private:
// 	int num_chains;
// 	int num_iter;
// 	bool progress;
// 	std::vector<Rcpp::List> _records;
// 	// std::unique_ptr<Minnesota> _mn;
// 	Eigen::MatrixXd dummy_response;
// 	std::vector<std::unique_ptr<Minnesota>> _mn;
// };

class MhMinnesota : public Minnesota {
public:
	MhMinnesota(
		int num_iter, const MhMinnSpec& spec, const MhMinnInits& inits, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		const Eigen::MatrixXd& x_dummy, const Eigen::MatrixXd& y_dummy, unsigned int seed
	)
	: Minnesota(num_iter, x, y, x_dummy, y_dummy, seed),
		// num_iter(num_iter),
		// mn_record(num_iter, dim, dim_design),
		mh_record(num_iter, dim),
		gamma_shp(spec._gam_rate), gamma_rate(spec._gam_shape),
		invgam_shp(spec._invgam_shape), invgam_scl(spec._invgam_scl), gaussian_variance(inits._acc_scale * inits._hess.inverse()),
		prevprior(Eigen::VectorXd::Zero(1 + dim)),
		candprior(Eigen::VectorXd::Zero(1 + dim)), numerator(0), denom(0),
		is_accept(true), lambda(inits._lambda), psi(inits._psi) {
		// mniw(2), mcmc_step(0), rng(seed) {
		prevprior[0] = inits._lambda;
		prevprior.tail(dim) = inits._psi;
		mh_record.lam_record[0] = inits._lambda;
		mh_record.psi_record.row(0) = inits._psi;
		mh_record.accept_record[0] = is_accept;
	}
	virtual ~MhMinnesota() = default;
	void computePosterior() override {
		estimateCoef();
		estimateCov();
	}
	// void addStep() { mcmc_step++; }
	void updateRecords() override {
		mn_record.assignRecords(mcmc_step, mniw); 
		mh_record.assignRecords(mcmc_step, lambda, psi, is_accept); 
	}
	void updateHyper() {
		candprior = Eigen::Map<Eigen::VectorXd>(sim_mgaussian_chol(1, prevprior, gaussian_variance).data(), 1 + dim);
		numerator = jointdens_hyperparam(
			candprior[0], candprior.segment(1, dim), dim, num_design,
			prior_prec, prior_scale, prior_shape, prec, scale, prior_shape + num_design, gamma_shp, gamma_rate, invgam_shp, invgam_scl
		);
    denom = jointdens_hyperparam(
			prevprior[0], prevprior.segment(1, dim), dim, num_design,
			prior_prec, prior_scale, prior_shape, prec, scale, prior_shape + num_design, gamma_shp, gamma_rate, invgam_shp, invgam_scl
		);
		is_accept = ( log(bvhar::unif_rand(0, 1)) < std::min(numerator - denom, 0.0) );
		if (is_accept) {
			lambda = candprior[0];
			psi = candprior.tail(dim);
		}
	}
	// void updateMniw() { mniw = sim_mn_iw(coef, prec, scale, prior_shape + num_design, true, rng); }
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateHyper();
		updateMniw();
		updateRecords();
	}
	Rcpp::List returnRecords(int num_burn, int thin) const override {
		Rcpp::List res = Rcpp::List::create(
			Rcpp::Named("lambda_record") = mh_record.lam_record,
			Rcpp::Named("psi_record") = mh_record.psi_record,
			Rcpp::Named("alpha_record") = mn_record.coef_record,
			Rcpp::Named("sigma_record") = mn_record.sig_record,
			// Rcpp::Named("accept_record") = thin_record(mn_record.accept_record, num_iter, num_burn, thin)
			Rcpp::Named("accept_record") = mh_record.accept_record
		);
		for (auto& record : res) {
			if (Rcpp::is<Rcpp::NumericMatrix>(record)) {
				record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
			} else if (Rcpp::is<Rcpp::NumericVector>(record)) {
				record = thin_record(Rcpp::as<Eigen::VectorXd>(record), num_iter, num_burn, thin);
			} else if (Rcpp::is<Rcpp::LogicalVector>(record)) {
				record = thin_record(Rcpp::as<VectorXb>(record), num_iter, num_burn, thin);
			}
		}
		return res;
	}
private:
	// int num_iter;
	// MinnRecords mn_record;
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
	// std::vector<Eigen::MatrixXd> mniw;
	// std::atomic<int> mcmc_step; // MCMC step
	// boost::random::mt19937 rng; // RNG instance for multi-chain
	// std::mutex mtx;
};

class MinnFlat {
public:
	MinnFlat(int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::MatrixXd& prec, unsigned int seed)
	: design(x), response(y), prior_prec(prec),
		num_iter(num_iter), dim(response.cols()), num_design(response.rows()), dim_design(design.cols()),
		mn_record(num_iter, dim, dim_design),
		mniw(2), mcmc_step(0), rng(seed),
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
	virtual void computePosterior() {
		estimateCoef();
		fitObs();
		estimateCov();
	}
	void addStep() { mcmc_step++; }
	virtual void updateRecords() { mn_record.assignRecords(mcmc_step, mniw); }
	void updateMniw() { mniw = sim_mn_iw(coef, prec, scale, shape, true, rng); }
	virtual void doPosteriorDraws() {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateMniw();
		updateRecords();
	}
	Rcpp::List returnRecords(int num_burn, int thin) const {
		Rcpp::List res = Rcpp::List::create(
			Rcpp::Named("alpha_record") = mn_record.coef_record,
			Rcpp::Named("sigma_record") = mn_record.sig_record
		);
		for (auto& record : res) {
			record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
		}
		return res;
	}
	Rcpp::List returnMinnRes() {
		// estimateCoef();
		// fitObs();
		// estimateCov();
		return Rcpp::List::create(
			Rcpp::Named("mn_mean") = coef,
			Rcpp::Named("fitted.values") = yhat,
			Rcpp::Named("residuals") = resid,
			Rcpp::Named("mn_prec") = prec,
			Rcpp::Named("iw_scale") = scale,
			Rcpp::Named("iw_shape") = shape,
			Rcpp::Named("df") = dim_design,
			Rcpp::Named("m") = dim,
			Rcpp::Named("obs") = num_design,
			Rcpp::Named("prior_mean") = Eigen::MatrixXd::Zero(dim_design, dim),
			Rcpp::Named("prior_precision") = prior_prec,
			// Rcpp::Named("prior_scale") = prior_scale,
			// Rcpp::Named("prior_shape") = prior_shape,
			Rcpp::Named("y0") = response,
			Rcpp::Named("design") = design
		);
	}
	MinnRecords returnMinnRecords(int num_burn, int thin) const {
		MinnRecords res_record(
			thin_record(mn_record.coef_record, num_iter, num_burn, thin).derived(),
			thin_record(mn_record.sig_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
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
	int num_iter;
	int dim;
	int num_design;
	int dim_design;
	// int num_dummy; // kp + k(+ 1)
	// int num_augment; // n + num_dummy
	MinnRecords mn_record;
	std::vector<Eigen::MatrixXd> mniw;
	std::atomic<int> mcmc_step; // MCMC step
	boost::random::mt19937 rng; // RNG instance for multi-chain
	std::mutex mtx;
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

#endif // MINNESOTA_H