#ifndef MCMCSV_H
#define MCMCSV_H

#include "bvhardesign.h"
#include "bvhardraw.h"
#include "bvharprogress.h"

namespace bvhar {

struct SvParams {
	int _iter;
	Eigen::MatrixXd _x;
	Eigen::MatrixXd _y;
	Eigen::VectorXd _sig_shp;
	Eigen::VectorXd _sig_scl;
	Eigen::VectorXd _init_mean;
	Eigen::MatrixXd _init_prec;
	Eigen::VectorXd _mean_non;
	double _sd_non;
	bool _mean;

	SvParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& spec, Rcpp::List& intercept,
		bool include_mean
	)
	: _iter(num_iter), _x(x), _y(y),
		_sig_shp(Rcpp::as<Eigen::VectorXd>(spec["shape"])),
		_sig_scl(Rcpp::as<Eigen::VectorXd>(spec["scale"])),
		_init_mean(Rcpp::as<Eigen::VectorXd>(spec["initial_mean"])),
		_init_prec(Rcpp::as<Eigen::MatrixXd>(spec["initial_prec"])),
		_mean_non(Rcpp::as<Eigen::VectorXd>(intercept["mean_non"])),
		_sd_non(intercept["sd_non"]), _mean(include_mean) {}
};

struct MinnParams : public SvParams {
	Eigen::MatrixXd _prec_diag;
	Eigen::MatrixXd _prior_mean;
	Eigen::MatrixXd _prior_prec;

	MinnParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& sv_spec, Rcpp::List& priors, Rcpp::List& intercept,
		bool include_mean
	)
	: SvParams(num_iter, x, y, sv_spec, intercept, include_mean),
		_prec_diag(Eigen::MatrixXd::Zero(y.cols(), y.cols())) {
		int lag = priors["p"]; // append to bayes_spec, p = 3 in VHAR
		Eigen::VectorXd _sigma = Rcpp::as<Eigen::VectorXd>(priors["sigma"]);
		double _lambda = priors["lambda"];
		double _eps = priors["eps"];
		int dim = _sigma.size();
		Eigen::VectorXd _daily(dim);
		Eigen::VectorXd _weekly(dim);
		Eigen::VectorXd _monthly(dim);
		if (priors.containsElementNamed("delta")) {
			_daily = Rcpp::as<Eigen::VectorXd>(priors["delta"]);
			_weekly.setZero();
			_monthly.setZero();
		} else {
			_daily = Rcpp::as<Eigen::VectorXd>(priors["daily"]);
			_weekly = Rcpp::as<Eigen::VectorXd>(priors["weekly"]);
			_monthly = Rcpp::as<Eigen::VectorXd>(priors["monthly"]);
		}
		Eigen::MatrixXd dummy_response = build_ydummy(lag, _sigma, _lambda, _daily, _weekly, _monthly, false);
		Eigen::MatrixXd dummy_design = build_xdummy(
			Eigen::VectorXd::LinSpaced(lag, 1, lag),
			_lambda, _sigma, _eps, false
		);
		_prior_prec = dummy_design.transpose() * dummy_design;
		_prior_mean = _prior_prec.inverse() * dummy_design.transpose() * dummy_response;
		_prec_diag = Eigen::MatrixXd::Zero(dim, dim);
		_prec_diag.diagonal() = 1 / _sigma.array();
	}
};

struct SsvsParams : public SvParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;
	Eigen::VectorXd _coef_spike;
	Eigen::VectorXd _coef_slab;
	Eigen::VectorXd _coef_weight;
	Eigen::VectorXd _contem_spike;
	Eigen::VectorXd _contem_slab;
	Eigen::VectorXd _contem_weight;
	double _coef_s1;
	double _coef_s2;
	double _contem_s1;
	double _contem_s2;

	SsvsParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& sv_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		Rcpp::List& ssvs_spec, Rcpp::List& intercept,
		bool include_mean
	)
	: SvParams(num_iter, x, y, sv_spec, intercept, include_mean),
		_grp_id(grp_id), _grp_mat(grp_mat),
		_coef_spike(Rcpp::as<Eigen::VectorXd>(ssvs_spec["coef_spike"])),
		_coef_slab(Rcpp::as<Eigen::VectorXd>(ssvs_spec["coef_slab"])),
		_coef_weight(Rcpp::as<Eigen::VectorXd>(ssvs_spec["coef_mixture"])),
		_contem_spike(Rcpp::as<Eigen::VectorXd>(ssvs_spec["chol_spike"])),
		_contem_slab(Rcpp::as<Eigen::VectorXd>(ssvs_spec["chol_slab"])),
		_contem_weight(Rcpp::as<Eigen::VectorXd>(ssvs_spec["chol_mixture"])),
		_coef_s1(ssvs_spec["coef_s1"]), _coef_s2(ssvs_spec["coef_s2"]),
		_contem_s1(ssvs_spec["chol_s1"]), _contem_s2(ssvs_spec["chol_s2"]) {}
};

struct HorseshoeParams : public SvParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;

	HorseshoeParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& sv_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		Rcpp::List& intercept, bool include_mean
	)
	: SvParams(num_iter, x, y, sv_spec, intercept, include_mean), _grp_id(grp_id), _grp_mat(grp_mat) {}
};

struct SvInits {
	Eigen::MatrixXd _coef;
	Eigen::VectorXd _contem;
	Eigen::VectorXd _lvol_init;
	Eigen::MatrixXd _lvol;
	Eigen::VectorXd _lvol_sig;

	SvInits(const SvParams& params) {
		_coef = (params._x.transpose() * params._x).llt().solve(params._x.transpose() * params._y); // OLS
		int dim = params._y.cols();
		int num_lowerchol = dim * (dim - 1) / 2;
		int num_design = params._y.rows();
		_contem = .001 * Eigen::VectorXd::Zero(num_lowerchol);
		_lvol_init = (params._y - params._x * _coef).transpose().array().square().rowwise().mean().log();
		_lvol = _lvol_init.transpose().replicate(num_design, 1);
		_lvol_sig = .1 * Eigen::VectorXd::Ones(dim);
	}
	SvInits(Rcpp::List& init)
	: _coef(Rcpp::as<Eigen::MatrixXd>(init["init_coef"])),
		_contem(Rcpp::as<Eigen::VectorXd>(init["init_contem"])),
		_lvol_init(Rcpp::as<Eigen::VectorXd>(init["lvol_init"])),
		_lvol(Rcpp::as<Eigen::MatrixXd>(init["lvol"])),
		_lvol_sig(Rcpp::as<Eigen::VectorXd>(init["lvol_sig"])) {}
	SvInits(Rcpp::List& init, int num_design)
	: _coef(Rcpp::as<Eigen::MatrixXd>(init["init_coef"])),
		_contem(Rcpp::as<Eigen::VectorXd>(init["init_contem"])),
		_lvol_init(Rcpp::as<Eigen::VectorXd>(init["lvol_init"])),
		_lvol(_lvol_init.transpose().replicate(num_design, 1)),
		_lvol_sig(Rcpp::as<Eigen::VectorXd>(init["lvol_sig"])) {}
};

struct SsvsInits : public SvInits {
	Eigen::VectorXd _coef_dummy;
	Eigen::VectorXd _coef_weight; // in SsvsParams: move coef_mixture and chol_mixture in set_ssvs()?
	Eigen::VectorXd _contem_weight; // in SsvsParams
	
	SsvsInits(Rcpp::List& init)
	: SvInits(init),
		_coef_dummy(Rcpp::as<Eigen::VectorXd>(init["init_coef_dummy"])),
		_coef_weight(Rcpp::as<Eigen::VectorXd>(init["coef_mixture"])),
		_contem_weight(Rcpp::as<Eigen::VectorXd>(init["chol_mixture"])) {}
	SsvsInits(Rcpp::List& init, int num_design)
	: SvInits(init, num_design),
		_coef_dummy(Rcpp::as<Eigen::VectorXd>(init["init_coef_dummy"])),
		_coef_weight(Rcpp::as<Eigen::VectorXd>(init["coef_mixture"])),
		_contem_weight(Rcpp::as<Eigen::VectorXd>(init["chol_mixture"])) {}
};

struct HorseshoeInits : public SvInits {
	Eigen::VectorXd _init_local;
	Eigen::VectorXd _init_global;
	Eigen::VectorXd _init_contem_local;
	Eigen::VectorXd _init_conetm_global;
	
	HorseshoeInits(Rcpp::List& init)
	: SvInits(init),
		_init_local(Rcpp::as<Eigen::VectorXd>(init["local_sparsity"])),
		_init_global(Rcpp::as<Eigen::VectorXd>(init["global_sparsity"])),
		_init_contem_local(Rcpp::as<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_conetm_global(Rcpp::as<Eigen::VectorXd>(init["contem_global_sparsity"])) {}
	HorseshoeInits(Rcpp::List& init, int num_design)
	: SvInits(init, num_design),
		_init_local(Rcpp::as<Eigen::VectorXd>(init["local_sparsity"])),
		_init_global(Rcpp::as<Eigen::VectorXd>(init["global_sparsity"])),
		_init_contem_local(Rcpp::as<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_conetm_global(Rcpp::as<Eigen::VectorXd>(init["contem_global_sparsity"])) {}
};

struct SvRecords {
	Eigen::MatrixXd coef_record; // alpha in VAR
	Eigen::MatrixXd contem_coef_record; // a = a21, a31, a32, ..., ak1, ..., ak(k-1)
	Eigen::MatrixXd lvol_sig_record; // sigma_h^2 = (sigma_(h1i)^2, ..., sigma_(hki)^2)
	Eigen::MatrixXd lvol_init_record; // h0 = h10, ..., hk0
	Eigen::MatrixXd lvol_record; // time-varying h = (h_1, ..., h_k) with h_j = (h_j1, ..., h_jn), row-binded
	
	SvRecords(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_coef)),
		contem_coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol)),
		lvol_sig_record(Eigen::MatrixXd::Ones(num_iter + 1, dim)),
		lvol_init_record(Eigen::MatrixXd::Zero(num_iter + 1, dim)),
		lvol_record(Eigen::MatrixXd::Zero(num_iter + 1, num_design * dim)) {}
	SvRecords(
		const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& h_record,
		const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& sigh_record
	)
	: coef_record(alpha_record), contem_coef_record(a_record),
		lvol_sig_record(sigh_record), lvol_init_record(Eigen::MatrixXd::Zero(coef_record.rows(), lvol_sig_record.cols())),
		lvol_record(h_record) {}
	SvRecords(
		const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& c_record, const Eigen::MatrixXd& h_record,
		const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& sigh_record
	)
	: coef_record(Eigen::MatrixXd::Zero(alpha_record.rows(), alpha_record.cols() + c_record.cols())), contem_coef_record(a_record),
		lvol_sig_record(sigh_record), lvol_init_record(Eigen::MatrixXd::Zero(coef_record.rows(), lvol_sig_record.cols())),
		lvol_record(h_record) {
		coef_record << alpha_record, c_record;
	}
	void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef,
		const Eigen::MatrixXd& lvol_draw, const Eigen::VectorXd& lvol_sig, const Eigen::VectorXd& lvol_init
	) {
		coef_record.row(id) = coef_vec;
		contem_coef_record.row(id) = contem_coef;
		// lvol_record.row(id) = vectorize_eigen(lvol_draw.transpose().eval());
		lvol_record.row(id) = lvol_draw.transpose().reshaped();
		lvol_sig_record.row(id) = lvol_sig;
		lvol_init_record.row(id) = lvol_init;
	}
};

struct SsvsRecords {
	Eigen::MatrixXd coef_dummy_record;
	Eigen::MatrixXd coef_weight_record;
	Eigen::MatrixXd contem_dummy_record;
	Eigen::MatrixXd contem_weight_record;

	SsvsRecords(int num_iter, int num_alpha, int num_grp, int num_lowerchol)
	: coef_dummy_record(Eigen::MatrixXd::Ones(num_iter + 1, num_alpha)),
		coef_weight_record(Eigen::MatrixXd::Zero(num_iter + 1, num_grp)),
		contem_dummy_record(Eigen::MatrixXd::Ones(num_iter + 1, num_lowerchol)),
		contem_weight_record(Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol)) {}
	void assignRecords(int id, const Eigen::VectorXd& coef_dummy, const Eigen::VectorXd& coef_weight, const Eigen::VectorXd& contem_dummy, const Eigen::VectorXd& contem_weight) {
		coef_dummy_record.row(id) = coef_dummy;
		coef_weight_record.row(id) = coef_weight;
		contem_dummy_record.row(id) = contem_dummy;
		contem_weight_record.row(id) = contem_weight;
	}
};

struct HorseshoeRecords {
	Eigen::MatrixXd local_record;
	Eigen::MatrixXd global_record;
	Eigen::MatrixXd shrink_record;

	HorseshoeRecords(int num_iter, int num_alpha, int num_grp, int num_lowerchol)
	: local_record(Eigen::MatrixXd::Zero(num_iter + 1, num_alpha)),
		global_record(Eigen::MatrixXd::Zero(num_iter + 1, num_grp)),
		shrink_record(Eigen::MatrixXd::Zero(num_iter + 1, num_alpha)) {}
	void assignRecords(int id, const Eigen::VectorXd& shrink_fac, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& global_lev) {
		shrink_record.row(id) = shrink_fac;
		local_record.row(id) = local_lev;
		global_record.row(id) = global_lev;
	}
};

class McmcSv {
public:
	McmcSv(const SvParams& params, const SvInits& inits, unsigned int seed)
	: include_mean(params._mean),
		x(params._x), y(params._y),
		num_iter(params._iter), dim(y.cols()), dim_design(x.cols()), num_design(y.rows()),
		num_lowerchol(dim * (dim - 1) / 2), num_coef(dim * dim_design),
		num_alpha(include_mean ? num_coef - dim : num_coef),
		sv_record(num_iter, dim, num_design, num_coef, num_lowerchol),
		mcmc_step(0), rng(seed),
		prior_mean_non(params._mean_non),
		prior_sd_non(params._sd_non * Eigen::VectorXd::Ones(dim)),
		coef_vec(Eigen::VectorXd::Zero(num_coef)),
		contem_coef(inits._contem),
		lvol_draw(inits._lvol), lvol_init(inits._lvol_init), lvol_sig(inits._lvol_sig),
		prior_alpha_mean(Eigen::VectorXd::Zero(num_coef)),
		prior_alpha_prec(Eigen::MatrixXd::Zero(num_coef, num_coef)),
		prior_chol_mean(Eigen::VectorXd::Zero(num_lowerchol)),
		prior_chol_prec(Eigen::MatrixXd::Identity(num_lowerchol, num_lowerchol)),
		coef_mat(inits._coef),
		contem_id(0),
		chol_lower(build_inv_lower(dim, contem_coef)),
		latent_innov(y - x * coef_mat),
		ortho_latent(Eigen::MatrixXd::Zero(num_design, dim)),
		prior_mean_j(Eigen::VectorXd::Zero(dim_design)),
		prior_prec_j(Eigen::MatrixXd::Identity(dim_design, dim_design)),
		coef_j(coef_mat),
		response_contem(Eigen::VectorXd::Zero(num_design)),
		sqrt_sv(Eigen::MatrixXd::Zero(num_design, dim)),
		prior_sig_shp(params._sig_shp), prior_sig_scl(params._sig_scl),
		prior_init_mean(params._init_mean), prior_init_prec(params._init_prec) {
		if (include_mean) {
			prior_alpha_mean.tail(dim) = prior_mean_non;
			prior_alpha_prec.bottomRightCorner(dim, dim).diagonal() = prior_sd_non.array().square();
		}
		// coef_vec.head(num_alpha) = vectorize_eigen(coef_mat.topRows(num_alpha / dim).eval());
		coef_vec.head(num_alpha) = coef_mat.topRows(num_alpha / dim).reshaped();
		if (include_mean) {
			coef_vec.tail(dim) = coef_mat.bottomRows(1).transpose();
		}
		sv_record.assignRecords(0, coef_vec, contem_coef, lvol_draw, lvol_sig, lvol_init);
	}
	virtual ~McmcSv() = default;
	void updateCoef() {
		for (int j = 0; j < dim; j++) {
			prior_mean_j = prior_alpha_mean.segment(dim_design * j, dim_design);
			prior_prec_j = prior_alpha_prec.block(dim_design * j, dim_design * j, dim_design, dim_design);
			coef_j = coef_mat;
			coef_j.col(j).setZero();
			Eigen::MatrixXd chol_lower_j = chol_lower.bottomRows(dim - j); // L_(j:k) = a_jt to a_kt for t = 1, ..., j - 1
			Eigen::MatrixXd sqrt_sv_j = sqrt_sv.rightCols(dim - j); // use h_jt to h_kt for t = 1, .. n => (k - j + 1) x k
			// Eigen::MatrixXd design_coef = kronecker_eigen(chol_lower_j.col(j), x).array().colwise() * vectorize_eigen(sqrt_sv_j).array(); // L_(j:k, j) otimes X0 scaled by D_(1:n, j:k): n(k - j + 1) x kp
			Eigen::MatrixXd design_coef = kronecker_eigen(chol_lower_j.col(j), x).array().colwise() * sqrt_sv_j.reshaped().array(); // L_(j:k, j) otimes X0 scaled by D_(1:n, j:k): n(k - j + 1) x kp
			// Eigen::VectorXd response_j = vectorize_eigen(
			// 	(((y - x * coef_j) * chol_lower_j.transpose()).array() * sqrt_sv_j.array()).matrix().eval() // Hadamard product between: (Y - X0 A(-j))L_(j:k)^T and D_(1:n, j:k)
			// ); // Response vector of j-th column coef equation: n(k - j + 1)-dim
			Eigen::VectorXd response_j = (((y - x * coef_j) * chol_lower_j.transpose()).array() * sqrt_sv_j.array()).reshaped(); // Hadamard product between: (Y - X0 A(-j))L_(j:k)^T and D_(1:n, j:k)
			varsv_regression(
				coef_mat.col(j),
				design_coef, response_j,
				prior_mean_j, prior_prec_j,
				rng
			);
		}
		// coef_vec.head(num_alpha) = vectorize_eigen(coef_mat.topRows(num_alpha / dim).eval());
		coef_vec.head(num_alpha) = coef_mat.topRows(num_alpha / dim).reshaped();
		if (include_mean) {
			coef_vec.tail(dim) = coef_mat.bottomRows(1).transpose();
		}
	}
	void updateState() {
		ortho_latent = latent_innov * chol_lower.transpose(); // L eps_t <=> Z0 U
		ortho_latent = (ortho_latent.array().square() + .0001).array().log(); // adjustment log(e^2 + c) for some c = 10^(-4) against numerical problems
		for (int t = 0; t < dim; t++) {
			varsv_ht(lvol_draw.col(t), lvol_init[t], lvol_sig[t], ortho_latent.col(t), rng);
		}
	}
	void updateImpact() {
		for (int j = 2; j < dim + 1; j++) {
			response_contem = latent_innov.col(j - 2).array() * sqrt_sv.col(j - 2).array(); // n-dim
			// Eigen::MatrixXd design_contem = latent_innov.leftCols(j - 1).array().colwise() * vectorize_eigen(sqrt_sv.col(j - 2).eval()).array(); // n x (j - 1)
			Eigen::MatrixXd design_contem = latent_innov.leftCols(j - 1).array().colwise() * sqrt_sv.col(j - 2).reshaped().array(); // n x (j - 1)
			contem_id = (j - 1) * (j - 2) / 2;
			varsv_regression(
				contem_coef.segment(contem_id, j - 1),
				design_contem, response_contem,
				prior_chol_mean.segment(contem_id, j - 1),
				prior_chol_prec.block(contem_id, contem_id, j - 1, j - 1),
				rng
			);
		}
	}
	void updateStateVar() { varsv_sigh(lvol_sig, prior_sig_shp, prior_sig_scl, lvol_init, lvol_draw, rng); }
	void updateInitState() { varsv_h0(lvol_init, prior_init_mean, prior_init_prec, lvol_draw.row(0), lvol_sig, rng); }
	void addStep() { mcmc_step++; }
	virtual void updateCoefPrec() = 0;
	virtual void updateCoefShrink() = 0;
	virtual void updateImpactPrec() = 0;
	virtual void updateRecords() = 0;
	virtual void doPosteriorDraws() = 0;
	virtual Rcpp::List returnRecords(int num_burn, int thin) const = 0;
	SvRecords returnSvRecords(int num_burn, int thin) const {
		// For out-of-forecasting: Not return lvol_init_record
		SvRecords res_record(
			thin_record(sv_record.coef_record, num_iter, num_burn, thin).derived(),
			thin_record(sv_record.lvol_record, num_iter, num_burn, thin).derived(),
			thin_record(sv_record.contem_coef_record, num_iter, num_burn, thin).derived(),
			thin_record(sv_record.lvol_sig_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}

protected:
	bool include_mean;
	Eigen::MatrixXd x;
	Eigen::MatrixXd y;
	std::mutex mtx;
	int num_iter;
	int dim; // k
  int dim_design; // kp(+1)
  int num_design; // n = T - p
  int num_lowerchol;
  int num_coef;
	int num_alpha;
	SvRecords sv_record;
	std::atomic<int> mcmc_step; // MCMC step
	boost::random::mt19937 rng; // RNG instance for multi-chain
	Eigen::VectorXd prior_mean_non; // prior mean of intercept term
	Eigen::VectorXd prior_sd_non; // prior sd of intercept term: c^2 I
	Eigen::VectorXd coef_vec;
	Eigen::VectorXd contem_coef;
	Eigen::MatrixXd lvol_draw; // h_j = (h_j1, ..., h_jn)
	Eigen::VectorXd lvol_init;
	Eigen::VectorXd lvol_sig;
	Eigen::VectorXd prior_alpha_mean; // prior mean vector of alpha
	Eigen::MatrixXd prior_alpha_prec; // prior precision of alpha
	Eigen::VectorXd prior_chol_mean; // prior mean vector of a = 0
	Eigen::MatrixXd prior_chol_prec; // prior precision of a = I
	Eigen::MatrixXd coef_mat;
	int contem_id;
	Eigen::MatrixXd chol_lower; // L in Sig_t^(-1) = L D_t^(-1) LT
	Eigen::MatrixXd latent_innov; // Z0 = Y0 - X0 A = (eps_p+1, eps_p+2, ..., eps_n+p)^T
  Eigen::MatrixXd ortho_latent; // orthogonalized Z0
	Eigen::VectorXd prior_mean_j; // Prior mean vector of j-th column of A
  Eigen::MatrixXd prior_prec_j; // Prior precision of j-th column of A
  Eigen::MatrixXd coef_j; // j-th column of A = 0: A(-j) = (alpha_1, ..., alpha_(j-1), 0, alpha_(j), ..., alpha_k)
	Eigen::VectorXd response_contem; // j-th column of Z0 = Y0 - X0 * A: n-dim
	Eigen::MatrixXd sqrt_sv; // stack sqrt of exp(h_t) = (exp(-h_1t / 2), ..., exp(-h_kt / 2)), t = 1, ..., n => n x k

private:
	Eigen::VectorXd prior_sig_shp;
	Eigen::VectorXd prior_sig_scl;
	Eigen::VectorXd prior_init_mean;
	Eigen::MatrixXd prior_init_prec;
};

class MinnSv : public McmcSv {
public:
	MinnSv(const MinnParams& params, const SvInits& inits, unsigned int seed)
		: McmcSv(params, inits, seed) {
		// prior_alpha_mean.head(num_alpha) = vectorize_eigen(params._prior_mean);
		prior_alpha_mean.head(num_alpha) = params._prior_mean.reshaped();
		prior_alpha_prec.topLeftCorner(num_alpha, num_alpha) = kronecker_eigen(params._prec_diag, params._prior_prec);
		if (include_mean) {
			prior_alpha_mean.tail(dim) = params._mean_non;
		}
	}
	virtual ~MinnSv() = default;
	void updateCoefPrec() override {};
	void updateCoefShrink() override {};
	void updateImpactPrec() override {};
	void updateRecords() override { sv_record.assignRecords(mcmc_step, coef_vec, contem_coef, lvol_draw, lvol_sig, lvol_init); }
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		sqrt_sv = (-lvol_draw / 2).array().exp(); // D_t before coef
		updateCoef();
		latent_innov = y - x * coef_mat; // E_t before a
		updateImpact();
		chol_lower = build_inv_lower(dim, contem_coef); // L before h_t
		updateState();
		updateStateVar();
		updateInitState();
		updateRecords();
	}
	Rcpp::List returnRecords(int num_burn, int thin) const override {
		Rcpp::List res = Rcpp::List::create(
			Rcpp::Named("alpha_record") = sv_record.coef_record.leftCols(num_alpha),
			Rcpp::Named("h_record") = sv_record.lvol_record,
			Rcpp::Named("a_record") = sv_record.contem_coef_record,
			Rcpp::Named("h0_record") = sv_record.lvol_init_record,
			Rcpp::Named("sigh_record") = sv_record.lvol_sig_record
		);
		if (include_mean) {
			res["c_record"] = sv_record.coef_record.rightCols(dim);
		}
		for (auto& record : res) {
			record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
		}
		return res;
	}
};

class SsvsSv : public McmcSv {
public:
	SsvsSv(const SsvsParams& params, const SsvsInits& inits, unsigned int seed)
	: McmcSv(params, inits, seed),
		grp_id(params._grp_id), grp_mat(params._grp_mat), grp_vec(grp_mat.reshaped()), num_grp(grp_id.size()),
		ssvs_record(num_iter, num_alpha, num_grp, num_lowerchol),
		coef_dummy(inits._coef_dummy), coef_weight(inits._coef_weight),
		contem_dummy(Eigen::VectorXd::Ones(num_lowerchol)), contem_weight(inits._contem_weight),
		coef_spike(params._coef_spike), coef_slab(params._coef_slab),
		contem_spike(params._contem_spike), contem_slab(params._contem_slab),
		coef_s1(params._coef_s1), coef_s2(params._coef_s2),
		contem_s1(params._contem_s1), contem_s2(params._contem_s2),
		prior_sd(Eigen::VectorXd::Zero(num_coef)),
		slab_weight(Eigen::VectorXd::Ones(num_alpha)), slab_weight_mat(Eigen::MatrixXd::Ones(num_alpha / dim, dim)),
		coef_mixture_mat(Eigen::VectorXd::Zero(num_alpha)) {
		if (include_mean) {
			prior_sd.tail(dim) = prior_sd_non;
		}
		ssvs_record.assignRecords(0, coef_dummy, coef_weight, contem_dummy, contem_weight);
	}
	virtual ~SsvsSv() = default;
	void updateCoefPrec() override {
		coef_mixture_mat = build_ssvs_sd(coef_spike, coef_slab, coef_dummy);
		prior_sd.head(num_alpha) = coef_mixture_mat;
		prior_alpha_prec.setZero();
		prior_alpha_prec.diagonal() = 1 / prior_sd.array().square();
	}
	void updateCoefShrink() override {
		for (int j = 0; j < num_grp; j++) {
			slab_weight_mat = (grp_mat.array() == grp_id[j]).select(
				coef_weight[j],
				slab_weight_mat
			);
		}
		// slab_weight = vectorize_eigen(slab_weight_mat);
		slab_weight = slab_weight_mat.reshaped();
		ssvs_dummy(
			coef_dummy,
			coef_vec.head(num_alpha),
			coef_slab, coef_spike, slab_weight,
			rng
		);
		ssvs_mn_weight(coef_weight, grp_vec, grp_id, coef_dummy, coef_s1, coef_s2, rng);
	}
	void updateImpactPrec() override {
		ssvs_dummy(contem_dummy, contem_coef, contem_slab, contem_spike, contem_weight, rng);
		ssvs_weight(contem_weight, contem_dummy, contem_s1, contem_s2, rng);
		prior_chol_prec.diagonal() = 1 / build_ssvs_sd(contem_spike, contem_slab, contem_dummy).array().square();
	}
	void updateRecords() override {
		sv_record.assignRecords(mcmc_step, coef_vec, contem_coef, lvol_draw, lvol_sig, lvol_init);
		ssvs_record.assignRecords(mcmc_step, coef_dummy, coef_weight, contem_dummy, contem_weight);
	}
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefPrec();
		sqrt_sv = (-lvol_draw / 2).array().exp(); // D_t before coef
		updateCoef();
		updateCoefShrink();
		updateImpactPrec();
		latent_innov = y - x * coef_mat; // E_t before a
		updateImpact();
		chol_lower = build_inv_lower(dim, contem_coef); // L before h_t
		updateState();
		updateStateVar();
		updateInitState();
		updateRecords();
	}
	Rcpp::List returnRecords(int num_burn, int thin) const override {
		Rcpp::List res = Rcpp::List::create(
			Rcpp::Named("alpha_record") = sv_record.coef_record.leftCols(num_alpha),
			Rcpp::Named("h_record") = sv_record.lvol_record,
			Rcpp::Named("a_record") = sv_record.contem_coef_record,
			Rcpp::Named("h0_record") = sv_record.lvol_init_record,
			Rcpp::Named("sigh_record") = sv_record.lvol_sig_record,
			Rcpp::Named("gamma_record") = ssvs_record.coef_dummy_record
		);
		if (include_mean) {
			res["c_record"] = sv_record.coef_record.rightCols(dim);
		}
		for (auto& record : res) {
			record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
		}
		return res;
	}
private:
	Eigen::VectorXi grp_id;
	Eigen::MatrixXi grp_mat;
	Eigen::VectorXi grp_vec;
	int num_grp;
	SsvsRecords ssvs_record;
	Eigen::VectorXd coef_dummy;
	Eigen::VectorXd coef_weight;
	Eigen::VectorXd contem_dummy;
	Eigen::VectorXd contem_weight;
	Eigen::VectorXd coef_spike;
	Eigen::VectorXd coef_slab;
	Eigen::VectorXd contem_spike;
	Eigen::VectorXd contem_slab;
	double coef_s1, coef_s2;
	double contem_s1, contem_s2;
	Eigen::VectorXd prior_sd;
	Eigen::VectorXd slab_weight; // pij vector
	Eigen::MatrixXd slab_weight_mat; // pij matrix: (dim*p) x dim
	Eigen::VectorXd coef_mixture_mat;
};

class HorseshoeSv : public McmcSv {
public:
	HorseshoeSv(const HorseshoeParams& params, const HorseshoeInits& inits, unsigned int seed)
	: McmcSv(params, inits, seed),
		grp_id(params._grp_id), grp_mat(params._grp_mat), grp_vec(grp_mat.reshaped()), num_grp(grp_id.size()),
		hs_record(num_iter, num_alpha, num_grp, num_lowerchol),
		local_lev(inits._init_local), global_lev(inits._init_global),
		shrink_fac(Eigen::VectorXd::Zero(num_alpha)),
		latent_local(Eigen::VectorXd::Zero(num_alpha)), latent_global(Eigen::VectorXd::Zero(num_grp)),
		lambda_mat(Eigen::MatrixXd::Zero(num_alpha, num_alpha)),
		coef_var(Eigen::VectorXd::Zero(num_alpha)),
		coef_var_loc(Eigen::MatrixXd::Zero(num_alpha / dim, dim)),
		contem_local_lev(inits._init_contem_local), contem_global_lev(inits._init_conetm_global),
		contem_var(Eigen::VectorXd::Zero(num_lowerchol)),
		latent_contem_local(Eigen::VectorXd::Zero(num_lowerchol)), latent_contem_global(Eigen::VectorXd::Zero(1)) {
		hs_record.assignRecords(0, shrink_fac, local_lev, global_lev);
	}
	virtual ~HorseshoeSv() = default;
	void updateCoefPrec() override {
		for (int j = 0; j < num_grp; j++) {
			coef_var_loc = (grp_mat.array() == grp_id[j]).select(
				global_lev[j],
				coef_var_loc
			);
		}
		// coef_var = vectorize_eigen(coef_var_loc);
		coef_var = coef_var_loc.reshaped();
		build_shrink_mat(lambda_mat, coef_var, local_lev);
		prior_alpha_prec.topLeftCorner(num_alpha, num_alpha) = lambda_mat;
		shrink_fac = 1 / (1 + lambda_mat.diagonal().array());
	}
	void updateCoefShrink() override {
		horseshoe_latent(latent_local, local_lev, rng);
		horseshoe_latent(latent_global, global_lev, rng);
		horseshoe_local_sparsity(local_lev, latent_local, coef_var, coef_vec.head(num_alpha), 1, rng);
		horseshoe_mn_global_sparsity(global_lev, grp_vec, grp_id, latent_global, local_lev, coef_vec.head(num_alpha), 1, rng);
	}
	void updateImpactPrec() override {
		horseshoe_latent(latent_contem_local, contem_local_lev, rng);
		horseshoe_latent(latent_contem_global, contem_global_lev, rng);
		// contem_var = vectorize_eigen(contem_global_lev.replicate(1, num_lowerchol).eval());
		contem_var = contem_global_lev.replicate(1, num_lowerchol).reshaped();
		horseshoe_local_sparsity(contem_local_lev, latent_contem_local, contem_var, contem_coef, 1, rng);
		contem_global_lev[0] = horseshoe_global_sparsity(latent_contem_global[0], latent_contem_local, contem_coef, 1, rng);
		build_shrink_mat(prior_chol_prec, contem_var, contem_local_lev);
	}
	void updateRecords() override {
		sv_record.assignRecords(mcmc_step, coef_vec, contem_coef, lvol_draw, lvol_sig, lvol_init);
		hs_record.assignRecords(mcmc_step, shrink_fac, local_lev, global_lev);
	}
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefPrec();
		sqrt_sv = (-lvol_draw / 2).array().exp(); // D_t before coef
		updateCoef();
		updateCoefShrink();
		updateImpactPrec();
		latent_innov = y - x * coef_mat; // E_t before a
		updateImpact();
		chol_lower = build_inv_lower(dim, contem_coef); // L before h_t
		updateState();
		updateStateVar();
		updateInitState();
		updateRecords();
	}
	Rcpp::List returnRecords(int num_burn, int thin) const override {
		Rcpp::List res = Rcpp::List::create(
			Rcpp::Named("alpha_record") = sv_record.coef_record.leftCols(num_alpha),
			Rcpp::Named("h_record") = sv_record.lvol_record,
			Rcpp::Named("a_record") = sv_record.contem_coef_record,
			Rcpp::Named("h0_record") = sv_record.lvol_init_record,
			Rcpp::Named("sigh_record") = sv_record.lvol_sig_record,
			Rcpp::Named("lambda_record") = hs_record.local_record,
			Rcpp::Named("tau_record") = hs_record.global_record,
			Rcpp::Named("kappa_record") = hs_record.shrink_record
		);
		if (include_mean) {
			res["c_record"] = sv_record.coef_record.rightCols(dim);
		}
		for (auto& record : res) {
			record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
		}
		return res;
	}

private:
	Eigen::VectorXi grp_id;
	Eigen::MatrixXi grp_mat;
	Eigen::VectorXi grp_vec;
	int num_grp;
	HorseshoeRecords hs_record;
	Eigen::VectorXd local_lev;
	Eigen::VectorXd global_lev;
	Eigen::VectorXd shrink_fac;
	Eigen::VectorXd latent_local;
	Eigen::VectorXd latent_global;
	Eigen::MatrixXd lambda_mat;
	Eigen::VectorXd coef_var;
	Eigen::MatrixXd coef_var_loc;
	Eigen::VectorXd contem_local_lev;
	Eigen::VectorXd contem_global_lev;
	Eigen::VectorXd contem_var;
	Eigen::VectorXd latent_contem_local;
	Eigen::VectorXd latent_contem_global;
};

} // namespace bvhar

#endif // MCMCSV_H