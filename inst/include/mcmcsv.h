#ifndef MCMCSV_H
#define MCMCSV_H

#include "bvhardesign.h"
#include "bvhardraw.h"
#include "bvharprogress.h"

namespace bvhar {

// Parameters
struct SvParams;
struct MinnSvParams;
struct HierMinnSvParams;
struct SsvsSvParams;
struct HsSvParams;
struct NgSvParams;
struct DlSvParams;
// Initialization
struct SvInits;
struct HierminnSvInits;
struct SsvsSvInits;
struct HsSvInits;
struct NgSvInits;
// MCMC records
struct SvRecords;
// MCMC algorithms
class McmcSv;
class MinnSv;
class HierminnSv;
class SsvsSv;
class HorseshoeSv;
class NormalgammaSv;
class DirLaplaceSv;

struct SvParams : public RegParams {
	Eigen::VectorXd _init_mean;
	Eigen::MatrixXd _init_prec;

	SvParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& spec, Rcpp::List& intercept,
		bool include_mean
	)
	: RegParams(num_iter, x, y, spec, intercept, include_mean),
		_init_mean(Rcpp::as<Eigen::VectorXd>(spec["initial_mean"])),
		_init_prec(Rcpp::as<Eigen::MatrixXd>(spec["initial_prec"])) {}
};

struct MinnSvParams : public SvParams {
	Eigen::MatrixXd _prec_diag;
	Eigen::MatrixXd _prior_mean;
	Eigen::MatrixXd _prior_prec;

	MinnSvParams(
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
		_prior_prec = (dummy_design.transpose() * dummy_design).inverse();
		_prior_mean = _prior_prec * dummy_design.transpose() * dummy_response;
		_prec_diag.diagonal() = _sigma;
	}
};

struct HierminnSvParams : public SvParams {
	double shape;
	double rate;
	Eigen::MatrixXd _prec_diag;
	Eigen::MatrixXd _prior_mean;
	Eigen::MatrixXd _prior_prec;
	Eigen::MatrixXi _grp_mat;
	bool _minnesota;
	std::set<int> _own_id;
	std::set<int> _cross_id;

	HierminnSvParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& sv_spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		Rcpp::List& priors, Rcpp::List& intercept,
		bool include_mean
	)
	: SvParams(num_iter, x, y, sv_spec, intercept, include_mean),
		shape(priors["shape"]), rate(priors["rate"]),
		_prec_diag(Eigen::MatrixXd::Zero(y.cols(), y.cols())) {
		int lag = priors["p"]; // append to bayes_spec, p = 3 in VHAR
		Eigen::VectorXd _sigma = Rcpp::as<Eigen::VectorXd>(priors["sigma"]);
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
		Eigen::MatrixXd dummy_response = build_ydummy(lag, _sigma, 1, _daily, _weekly, _monthly, false);
		Eigen::MatrixXd dummy_design = build_xdummy(
			Eigen::VectorXd::LinSpaced(lag, 1, lag),
			1, _sigma, _eps, false
		);
		_prior_prec = (dummy_design.transpose() * dummy_design).inverse();
		_prior_mean = _prior_prec * dummy_design.transpose() * dummy_response;
		_prec_diag.diagonal() = _sigma;
		_grp_mat = grp_mat;
		_minnesota = true;
		if (own_id.size() == 1 && cross_id.size() == 1) {
			_minnesota = false;
		}
		for (int i = 0; i < own_id.size(); ++i) {
			_own_id.insert(own_id[i]);
		}
		for (int i = 0; i < cross_id.size(); ++i) {
			_cross_id.insert(cross_id[i]);
		}
	}
};

struct SsvsSvParams : public SvParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;
	Eigen::VectorXd _coef_spike;
	Eigen::VectorXd _coef_slab;
	Eigen::VectorXd _coef_weight;
	Eigen::VectorXd _contem_spike;
	Eigen::VectorXd _contem_slab;
	Eigen::VectorXd _contem_weight;
	Eigen::VectorXd _coef_s1;
	Eigen::VectorXd _coef_s2;
	double _contem_s1;
	double _contem_s2;

	SsvsSvParams(
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
		_coef_s1(Rcpp::as<Eigen::VectorXd>(ssvs_spec["coef_s1"])), _coef_s2(Rcpp::as<Eigen::VectorXd>(ssvs_spec["coef_s2"])),
		_contem_s1(ssvs_spec["chol_s1"]), _contem_s2(ssvs_spec["chol_s2"]) {}
};

struct HsSvParams : public SvParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;

	HsSvParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& sv_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		Rcpp::List& intercept, bool include_mean
	)
	: SvParams(num_iter, x, y, sv_spec, intercept, include_mean), _grp_id(grp_id), _grp_mat(grp_mat) {}
};

struct NgSvParams : public SvParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;
	double _mh_sd;
	// double _local_shape;
	// double _contem_shape;
	double _group_shape;
	double _group_scl;
	double _global_shape;
	double _global_scl;
	double _contem_global_shape;
	double _contem_global_scl;

	NgSvParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& sv_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		Rcpp::List& ng_spec, Rcpp::List& intercept,
		bool include_mean
	)
	: SvParams(num_iter, x, y, sv_spec, intercept, include_mean), _grp_id(grp_id), _grp_mat(grp_mat),
		// _local_shape(ng_spec["local_shape"]),
		// _contem_shape(ng_spec["contem_shape"]),
		_mh_sd(ng_spec["shape_sd"]),
		_group_shape(ng_spec["group_shape"]), _group_scl(ng_spec["group_scale"]),
		_global_shape(ng_spec["global_shape"]), _global_scl(ng_spec["global_scale"]),
		_contem_global_shape(ng_spec["contem_global_shape"]), _contem_global_scl(ng_spec["contem_global_scale"]) {}
};

struct DlSvParams : public SvParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;
	double _dl_concen;
	double _contem_dl_concen;

	DlSvParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& sv_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		Rcpp::List& dl_spec, Rcpp::List& intercept,
		bool include_mean
	)
	: SvParams(num_iter, x, y, sv_spec, intercept, include_mean),
		_grp_id(grp_id), _grp_mat(grp_mat),
		_dl_concen(dl_spec["dirichlet"]), _contem_dl_concen(dl_spec["contem_dirichlet"]) {}
};

struct SvInits : public RegInits {
	Eigen::VectorXd _lvol_init;
	Eigen::MatrixXd _lvol;
	Eigen::VectorXd _lvol_sig;

	SvInits(const SvParams& params)
	: RegInits(params) {
		int dim = params._y.cols();
		int num_design = params._y.rows();
		_lvol_init = (params._y - params._x * _coef).transpose().array().square().rowwise().mean().log();
		_lvol = _lvol_init.transpose().replicate(num_design, 1);
		_lvol_sig = .1 * Eigen::VectorXd::Ones(dim);
	}
	SvInits(Rcpp::List& init)
	: RegInits(init),
		_lvol_init(Rcpp::as<Eigen::VectorXd>(init["lvol_init"])),
		_lvol(Rcpp::as<Eigen::MatrixXd>(init["lvol"])),
		_lvol_sig(Rcpp::as<Eigen::VectorXd>(init["lvol_sig"])) {}
	SvInits(Rcpp::List& init, int num_design)
	: RegInits(init),
		_lvol_init(Rcpp::as<Eigen::VectorXd>(init["lvol_init"])),
		_lvol(_lvol_init.transpose().replicate(num_design, 1)),
		_lvol_sig(Rcpp::as<Eigen::VectorXd>(init["lvol_sig"])) {}
};

struct HierminnSvInits : public SvInits {
	double _own_lambda;
	double _cross_lambda;
	double _contem_lambda;

	HierminnSvInits(Rcpp::List& init)
	: SvInits(init),
		_own_lambda(init["own_lambda"]), _cross_lambda(init["cross_lambda"]), _contem_lambda(init["contem_lambda"]) {}
};

struct SsvsSvInits : public SvInits {
	Eigen::VectorXd _coef_dummy;
	Eigen::VectorXd _coef_weight; // in SsvsSvParams: move coef_mixture and chol_mixture in set_ssvs()?
	Eigen::VectorXd _contem_weight; // in SsvsSvParams
	
	SsvsSvInits(Rcpp::List& init)
	: SvInits(init),
		_coef_dummy(Rcpp::as<Eigen::VectorXd>(init["init_coef_dummy"])),
		_coef_weight(Rcpp::as<Eigen::VectorXd>(init["coef_mixture"])),
		_contem_weight(Rcpp::as<Eigen::VectorXd>(init["chol_mixture"])) {}
	SsvsSvInits(Rcpp::List& init, int num_design)
	: SvInits(init, num_design),
		_coef_dummy(Rcpp::as<Eigen::VectorXd>(init["init_coef_dummy"])),
		_coef_weight(Rcpp::as<Eigen::VectorXd>(init["coef_mixture"])),
		_contem_weight(Rcpp::as<Eigen::VectorXd>(init["chol_mixture"])) {}
};

struct HsSvInits : public SvInits {
	Eigen::VectorXd _init_local;
	Eigen::VectorXd _init_group;
	double _init_global;
	Eigen::VectorXd _init_contem_local;
	Eigen::VectorXd _init_conetm_global;
	
	HsSvInits(Rcpp::List& init)
	: SvInits(init),
		_init_local(Rcpp::as<Eigen::VectorXd>(init["local_sparsity"])),
		_init_group(Rcpp::as<Eigen::VectorXd>(init["group_sparsity"])),
		_init_global(init["global_sparsity"]),
		_init_contem_local(Rcpp::as<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_conetm_global(Rcpp::as<Eigen::VectorXd>(init["contem_global_sparsity"])) {}
	HsSvInits(Rcpp::List& init, int num_design)
	: SvInits(init, num_design),
		_init_local(Rcpp::as<Eigen::VectorXd>(init["local_sparsity"])),
		_init_group(Rcpp::as<Eigen::VectorXd>(init["group_sparsity"])),
		_init_global(init["global_sparsity"]),
		_init_contem_local(Rcpp::as<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_conetm_global(Rcpp::as<Eigen::VectorXd>(init["contem_global_sparsity"])) {}
};

struct NgSvInits : public HsSvInits {
	Eigen::VectorXd _init_local_shape;
	double _init_contem_shape;

	NgSvInits(Rcpp::List& init)
	: HsSvInits(init),
		_init_local_shape(Rcpp::as<Eigen::VectorXd>(init["local_shape"])),
		_init_contem_shape(init["contem_shape"]) {}
	
	NgSvInits(Rcpp::List& init, int num_design)
	: HsSvInits(init, num_design),
		_init_local_shape(Rcpp::as<Eigen::VectorXd>(init["local_shape"])),
		_init_contem_shape(init["contem_shape"]) {}
};

struct SvRecords : public RegRecords {
	Eigen::MatrixXd lvol_sig_record; // sigma_h^2 = (sigma_(h1i)^2, ..., sigma_(hki)^2)
	Eigen::MatrixXd lvol_init_record; // h0 = h10, ..., hk0
	Eigen::MatrixXd lvol_record; // time-varying h = (h_1, ..., h_k) with h_j = (h_j1, ..., h_jn), row-binded
	
	SvRecords(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: RegRecords(num_iter, dim, num_design, num_coef, num_lowerchol),
		lvol_sig_record(Eigen::MatrixXd::Ones(num_iter + 1, dim)),
		lvol_init_record(Eigen::MatrixXd::Zero(num_iter + 1, dim)),
		lvol_record(Eigen::MatrixXd::Zero(num_iter + 1, num_design * dim)) {}
	SvRecords(
		const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& h_record,
		const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& sigh_record
	)
	: RegRecords(alpha_record, a_record),
		lvol_sig_record(sigh_record), lvol_init_record(Eigen::MatrixXd::Zero(coef_record.rows(), lvol_sig_record.cols())),
		lvol_record(h_record) {}
	SvRecords(
		const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& c_record, const Eigen::MatrixXd& h_record,
		const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& sigh_record
	)
	: RegRecords(Eigen::MatrixXd::Zero(alpha_record.rows(), alpha_record.cols() + c_record.cols()), a_record),
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
		lvol_record.row(id) = lvol_draw.transpose().reshaped();
		lvol_sig_record.row(id) = lvol_sig;
		lvol_init_record.row(id) = lvol_init;
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
		sparse_record(num_iter, dim, num_design, num_alpha, num_lowerchol),
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
		sparse_coef(Eigen::MatrixXd::Zero(num_alpha / dim, dim)), sparse_contem(Eigen::VectorXd::Zero(num_lowerchol)),
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
		sparse_record.assignRecords(0, sparse_coef, sparse_contem);
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
			draw_savs(sparse_coef.col(j), coef_mat.col(j).head(num_alpha / dim), design_coef);
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
			draw_savs(sparse_contem.segment(contem_id, j - 1), contem_coef.segment(contem_id, j - 1), design_contem);
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
	void updateCoefRecords() {
		sv_record.assignRecords(mcmc_step, coef_vec, contem_coef, lvol_draw, lvol_sig, lvol_init);
		sparse_record.assignRecords(mcmc_step, sparse_coef, sparse_contem);
	}
	virtual Rcpp::List returnRecords(int num_burn, int thin) const = 0;
	SvRecords returnSvRecords(int num_burn, int thin, bool sparse = false) const {
		if (sparse) {
			Eigen::MatrixXd coef_record(num_iter + 1, num_coef);
			if (include_mean) {
				coef_record << sparse_record.coef_record, sv_record.coef_record.rightCols(dim);
			} else {
				coef_record = sparse_record.coef_record;
			}
			return SvRecords(
				thin_record(coef_record, num_iter, num_burn, thin).derived(),
				thin_record(sv_record.lvol_record, num_iter, num_burn, thin).derived(),
				thin_record(sparse_record.contem_coef_record, num_iter, num_burn, thin).derived(),
				thin_record(sv_record.lvol_sig_record, num_iter, num_burn, thin).derived()
			);
		}
		SvRecords res_record(
			thin_record(sv_record.coef_record, num_iter, num_burn, thin).derived(),
			thin_record(sv_record.lvol_record, num_iter, num_burn, thin).derived(),
			thin_record(sv_record.contem_coef_record, num_iter, num_burn, thin).derived(),
			thin_record(sv_record.lvol_sig_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
	virtual SsvsRecords returnSsvsRecords(int num_burn, int thin) const = 0;
	virtual HorseshoeRecords returnHsRecords(int num_burn, int thin) const = 0;
	virtual NgRecords returnNgRecords(int num_burn, int thin) const = 0;

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
	SparseRecords sparse_record;
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
	Eigen::MatrixXd sparse_coef;
	Eigen::VectorXd sparse_contem;

private:
	Eigen::VectorXd prior_sig_shp;
	Eigen::VectorXd prior_sig_scl;
	Eigen::VectorXd prior_init_mean;
	Eigen::MatrixXd prior_init_prec;
};

class MinnSv : public McmcSv {
public:
	MinnSv(const MinnSvParams& params, const SvInits& inits, unsigned int seed) : McmcSv(params, inits, seed) {
		// prior_alpha_mean.head(num_alpha) = vectorize_eigen(params._prior_mean);
		prior_alpha_mean.head(num_alpha) = params._prior_mean.reshaped();
		prior_alpha_prec.topLeftCorner(num_alpha, num_alpha).diagonal() = 1 / kronecker_eigen(params._prec_diag, params._prior_prec).diagonal().array();
		if (include_mean) {
			prior_alpha_mean.tail(dim) = params._mean_non;
		}
	}
	virtual ~MinnSv() = default;
	void updateCoefPrec() override {}
	void updateCoefShrink() override {};
	void updateImpactPrec() override {};
	void updateRecords() override { updateCoefRecords(); }
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
			Rcpp::Named("sigh_record") = sv_record.lvol_sig_record,
			Rcpp::Named("alpha_sparse_record") = sparse_record.coef_record,
			Rcpp::Named("a_sparse_record") = sparse_record.contem_coef_record
		);
		if (include_mean) {
			res["c_record"] = sv_record.coef_record.rightCols(dim);
		}
		for (auto& record : res) {
			record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
		}
		return res;
	}
	SsvsRecords returnSsvsRecords(int num_burn, int thin) const override {
		return SsvsRecords();
	}
	HorseshoeRecords returnHsRecords(int num_burn, int thin) const override {
		return HorseshoeRecords();
	}
	NgRecords returnNgRecords(int num_burn, int thin) const override {
		return NgRecords();
	}
};

class HierminnSv : public McmcSv {
public:
	HierminnSv(const HierminnSvParams& params, const HierminnSvInits& inits, unsigned int seed)
		: McmcSv(params, inits, seed),
			own_id(params._own_id), cross_id(params._cross_id), coef_minnesota(params._minnesota), grp_mat(params._grp_mat), grp_vec(grp_mat.reshaped()),
			own_lambda(inits._own_lambda), cross_lambda(inits._cross_lambda), contem_lambda(inits._contem_lambda),
			own_shape(params.shape), own_rate(params.rate),
			cross_shape(params.shape), cross_rate(params.rate),
			contem_shape(params.shape), contem_rate(params.rate) {
		prior_alpha_mean.head(num_alpha) = params._prior_mean.reshaped();
		prior_alpha_prec.topLeftCorner(num_alpha, num_alpha).diagonal() = 1 / kronecker_eigen(params._prec_diag, params._prior_prec).diagonal().array();
		for (int i = 0; i < num_alpha; ++i) {
			if (own_id.find(grp_vec[i]) != own_id.end()) {
				prior_alpha_prec(i, i) /= own_lambda; // divide because it is precision
			}
			if (cross_id.find(grp_vec[i]) != cross_id.end()) {
				prior_alpha_prec(i, i) /= cross_lambda; // divide because it is precision
			}
		}
		if (include_mean) {
			prior_alpha_mean.tail(dim) = params._mean_non;
		}
		prior_chol_prec.diagonal() /= contem_lambda; // divide because it is precision
	}
	virtual ~HierminnSv() = default;
	void updateCoefPrec() override {
		minnesota_lambda(
			own_lambda, own_shape, own_rate,
			coef_vec.head(num_alpha), prior_alpha_mean.head(num_alpha), prior_alpha_prec,
			grp_vec, own_id, rng
		);
		if (coef_minnesota) {
			minnesota_lambda(
				cross_lambda, cross_shape, cross_rate,
				coef_vec.head(num_alpha), prior_alpha_mean.head(num_alpha), prior_alpha_prec,
				grp_vec, cross_id, rng
			);
		}
	}
	void updateCoefShrink() override {
		for (int i = 0; i < num_alpha; ++i) {
			if (own_id.find(grp_vec[i]) != own_id.end()) {
				prior_alpha_prec(i, i) /= own_lambda; // divide because it is precision
			}
			if (cross_id.find(grp_vec[i]) != cross_id.end()) {
				prior_alpha_prec(i, i) /= cross_lambda; // divide because it is precision
			}
		}
	};
	void updateImpactPrec() override {
		minnesota_contem_lambda(
			contem_lambda, contem_shape, contem_rate,
			contem_coef, prior_chol_mean, prior_chol_prec,
			rng
		);
	};
	void updateRecords() override { updateCoefRecords(); }
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
			Rcpp::Named("alpha_sparse_record") = sparse_record.coef_record,
			Rcpp::Named("a_sparse_record") = sparse_record.contem_coef_record
		);
		if (include_mean) {
			res["c_record"] = sv_record.coef_record.rightCols(dim);
		}
		for (auto& record : res) {
			record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
		}
		return res;
	}
	SsvsRecords returnSsvsRecords(int num_burn, int thin) const override {
		return SsvsRecords();
	}
	HorseshoeRecords returnHsRecords(int num_burn, int thin) const override {
		return HorseshoeRecords();
	}
	NgRecords returnNgRecords(int num_burn, int thin) const override {
		return NgRecords();
	}
private:
	std::set<int> own_id;
	std::set<int> cross_id;
	bool coef_minnesota;
	Eigen::MatrixXi grp_mat;
	Eigen::VectorXi grp_vec;
	// int num_grp;
	double own_lambda;
	double cross_lambda;
	double contem_lambda;
	double own_shape;
	double own_rate;
	double cross_shape;
	double cross_rate;
	double contem_shape;
	double contem_rate;
};

class SsvsSv : public McmcSv {
public:
	SsvsSv(const SsvsSvParams& params, const SsvsSvInits& inits, unsigned int seed)
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
		updateCoefRecords();
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
			Rcpp::Named("gamma_record") = ssvs_record.coef_dummy_record,
			Rcpp::Named("alpha_sparse_record") = sparse_record.coef_record,
			Rcpp::Named("a_sparse_record") = sparse_record.contem_coef_record
		);
		if (include_mean) {
			res["c_record"] = sv_record.coef_record.rightCols(dim);
		}
		for (auto& record : res) {
			record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
		}
		return res;
	}
	SsvsRecords returnSsvsRecords(int num_burn, int thin) const override {
		SsvsRecords res_record(
			thin_record(ssvs_record.coef_dummy_record, num_iter, num_burn, thin).derived(),
			thin_record(ssvs_record.coef_weight_record, num_iter, num_burn, thin).derived(),
			thin_record(ssvs_record.contem_dummy_record, num_iter, num_burn, thin).derived(),
			thin_record(ssvs_record.contem_weight_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
	HorseshoeRecords returnHsRecords(int num_burn, int thin) const override {
		return HorseshoeRecords();
	}
	NgRecords returnNgRecords(int num_burn, int thin) const override {
		return NgRecords();
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
	Eigen::VectorXd coef_s1, coef_s2;
	double contem_s1, contem_s2;
	Eigen::VectorXd prior_sd;
	Eigen::VectorXd slab_weight; // pij vector
	Eigen::MatrixXd slab_weight_mat; // pij matrix: (dim*p) x dim
	Eigen::VectorXd coef_mixture_mat;
};

class HorseshoeSv : public McmcSv {
public:
	HorseshoeSv(const HsSvParams& params, const HsSvInits& inits, unsigned int seed)
	: McmcSv(params, inits, seed),
		grp_id(params._grp_id), grp_mat(params._grp_mat), grp_vec(grp_mat.reshaped()), num_grp(grp_id.size()),
		hs_record(num_iter, num_alpha, num_grp),
		local_lev(inits._init_local), group_lev(inits._init_group), global_lev(inits._init_global),
		local_fac(Eigen::VectorXd::Zero(num_alpha)),
		shrink_fac(Eigen::VectorXd::Zero(num_alpha)),
		latent_local(Eigen::VectorXd::Zero(num_alpha)), latent_group(Eigen::VectorXd::Zero(num_grp)), latent_global(0.0),
		lambda_mat(Eigen::MatrixXd::Zero(num_alpha, num_alpha)),
		coef_var(Eigen::VectorXd::Zero(num_alpha)),
		coef_var_loc(Eigen::MatrixXd::Zero(num_alpha / dim, dim)),
		contem_local_lev(inits._init_contem_local), contem_global_lev(inits._init_conetm_global),
		contem_var(Eigen::VectorXd::Zero(num_lowerchol)),
		latent_contem_local(Eigen::VectorXd::Zero(num_lowerchol)), latent_contem_global(Eigen::VectorXd::Zero(1)) {
		// hs_record.assignRecords(0, shrink_fac, local_lev, global_lev);
		hs_record.assignRecords(0, shrink_fac, local_lev, group_lev, global_lev);
	}
	virtual ~HorseshoeSv() = default;
	void updateCoefPrec() override {
		for (int j = 0; j < num_grp; j++) {
			coef_var_loc = (grp_mat.array() == grp_id[j]).select(
				// global_lev[j],
				// global_lev * group_lev[j],
				group_lev[j],
				coef_var_loc
			);
		}
		coef_var = coef_var_loc.reshaped();
		// build_shrink_mat(lambda_mat, coef_var, global_lev * local_lev);
		local_fac.array() = coef_var.array() * local_lev.array();
		lambda_mat.setZero();
		lambda_mat.diagonal() = 1 / (global_lev * local_fac.array()).square();
		prior_alpha_prec.topLeftCorner(num_alpha, num_alpha) = lambda_mat;
		shrink_fac = 1 / (1 + lambda_mat.diagonal().array());
	}
	void updateCoefShrink() override {
		horseshoe_latent(latent_local, local_lev, rng);
		// horseshoe_latent(latent_global, global_lev, rng);
		horseshoe_latent(latent_group, group_lev, rng);
		horseshoe_latent(latent_global, global_lev, rng);
		global_lev = horseshoe_global_sparsity(latent_global, local_fac, coef_vec.head(num_alpha), 1, rng);
		horseshoe_mn_sparsity(group_lev, grp_vec, grp_id, latent_group, global_lev, local_lev, coef_vec.head(num_alpha), 1, rng);
		horseshoe_local_sparsity(local_lev, latent_local, coef_var, coef_vec.head(num_alpha), global_lev, rng);
		// horseshoe_mn_global_sparsity(global_lev, grp_vec, grp_id, latent_global, local_lev, coef_vec.head(num_alpha), 1, rng);
	}
	void updateImpactPrec() override {
		horseshoe_latent(latent_contem_local, contem_local_lev, rng);
		horseshoe_latent(latent_contem_global, contem_global_lev, rng);
		contem_var = contem_global_lev.replicate(1, num_lowerchol).reshaped();
		horseshoe_local_sparsity(contem_local_lev, latent_contem_local, contem_var, contem_coef, 1, rng);
		contem_global_lev[0] = horseshoe_global_sparsity(latent_contem_global[0], latent_contem_local, contem_coef, 1, rng);
		build_shrink_mat(prior_chol_prec, contem_var, contem_local_lev);
	}
	void updateRecords() override {
		updateCoefRecords();
		hs_record.assignRecords(mcmc_step, shrink_fac, local_lev, group_lev, global_lev);
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
			Rcpp::Named("eta_record") = hs_record.group_record,
			Rcpp::Named("tau_record") = hs_record.global_record,
			Rcpp::Named("kappa_record") = hs_record.shrink_record,
			Rcpp::Named("alpha_sparse_record") = sparse_record.coef_record,
			Rcpp::Named("a_sparse_record") = sparse_record.contem_coef_record
		);
		if (include_mean) {
			res["c_record"] = sv_record.coef_record.rightCols(dim);
		}
		for (auto& record : res) {
			if (Rcpp::is<Rcpp::NumericMatrix>(record)) {
				record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
			} else {
				record = thin_record(Rcpp::as<Eigen::VectorXd>(record), num_iter, num_burn, thin);
			}
			// record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
		}
		return res;
	}
	SsvsRecords returnSsvsRecords(int num_burn, int thin) const override {
		return SsvsRecords();
	}
	HorseshoeRecords returnHsRecords(int num_burn, int thin) const override {
		HorseshoeRecords res_record(
			thin_record(hs_record.local_record, num_iter, num_burn, thin).derived(),
			thin_record(hs_record.group_record, num_iter, num_burn, thin).derived(),
			thin_record(hs_record.global_record, num_iter, num_burn, thin).derived(),
			thin_record(hs_record.shrink_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
	NgRecords returnNgRecords(int num_burn, int thin) const override {
		return NgRecords();
	}

private:
	Eigen::VectorXi grp_id;
	Eigen::MatrixXi grp_mat;
	Eigen::VectorXi grp_vec;
	int num_grp;
	HorseshoeRecords hs_record;
	Eigen::VectorXd local_lev;
	// Eigen::VectorXd global_lev;
	Eigen::VectorXd group_lev;
	double global_lev;
	Eigen::VectorXd local_fac;
	Eigen::VectorXd shrink_fac;
	Eigen::VectorXd latent_local;
	// Eigen::VectorXd latent_global;
	Eigen::VectorXd latent_group;
	double latent_global;
	Eigen::MatrixXd lambda_mat;
	Eigen::VectorXd coef_var;
	Eigen::MatrixXd coef_var_loc;
	Eigen::VectorXd contem_local_lev;
	Eigen::VectorXd contem_global_lev;
	Eigen::VectorXd contem_var;
	Eigen::VectorXd latent_contem_local;
	Eigen::VectorXd latent_contem_global;
};

class NormalgammaSv : public McmcSv {
public:
	NormalgammaSv(const NgSvParams& params, const NgSvInits& inits, unsigned int seed)
	: McmcSv(params, inits, seed),
		grp_id(params._grp_id), grp_mat(params._grp_mat), grp_vec(grp_mat.reshaped()), num_grp(grp_id.size()),
		ng_record(num_iter, num_alpha, num_grp),
		mh_sd(params._mh_sd),
		// local_shape(params._local_shape), contem_shape(params._contem_shape),
		local_shape(inits._init_local_shape), local_shape_fac(Eigen::VectorXd::Ones(num_alpha)),
		local_shape_loc(Eigen::MatrixXd::Ones(num_alpha / dim, dim)),
		contem_shape(inits._init_contem_shape),
		group_shape(params._group_shape), group_scl(params._global_scl),
		global_shape(params._global_shape), global_scl(params._global_scl),
		contem_global_shape(params._contem_global_shape), contem_global_scl(params._contem_global_scl),
		local_lev(inits._init_local), group_lev(inits._init_group), global_lev(inits._init_global),
		local_fac(Eigen::VectorXd::Zero(num_alpha)),
		coef_var(Eigen::VectorXd::Zero(num_alpha)),
		coef_var_loc(Eigen::MatrixXd::Zero(num_alpha / dim, dim)),
		contem_local_lev(inits._init_contem_local), contem_global_lev(inits._init_conetm_global),
		contem_var(Eigen::VectorXd::Zero(num_lowerchol)),
		contem_fac(Eigen::VectorXd::Zero(num_lowerchol)) {
		ng_record.assignRecords(0, local_lev, group_lev, global_lev);
	}
	virtual ~NormalgammaSv() = default;
	void updateCoefPrec() override {
		ng_mn_shape_jump(local_shape, local_lev, group_lev, grp_vec, grp_id, global_lev, mh_sd, rng);
		for (int j = 0; j < num_grp; j++) {
			coef_var_loc = (grp_mat.array() == grp_id[j]).select(
				group_lev[j],
				coef_var_loc
			);
			local_shape_loc = (grp_mat.array() == grp_id[j]).select(
				local_shape[j],
				local_shape_loc
			);
		}
		coef_var = coef_var_loc.reshaped();
		local_shape_fac = local_shape_loc.reshaped();
		// local_fac.array() = coef_var.array() * local_lev.array();
		// prior_alpha_prec.topLeftCorner(num_alpha, num_alpha).diagonal() = 1 / (global_lev * local_fac.array()).square();
		local_fac.array() = global_lev * coef_var.array() * local_lev.array();
		prior_alpha_prec.topLeftCorner(num_alpha, num_alpha).diagonal() = 1 / local_fac.array().square();
	}
	void updateCoefShrink() override {
		local_fac.array() = local_lev.array() / coef_var.array();
		// global_lev = ng_global_sparsity(local_fac, local_shape, global_shape, global_scl, rng);
		global_lev = ng_global_sparsity(local_fac, local_shape_fac, global_shape, global_scl, rng);
		local_fac.array() = global_lev * coef_var.array() * local_lev.array();
		// ng_local_sparsity(local_fac, local_shape, coef_vec.head(num_alpha), global_lev * coef_var, rng);
		ng_local_sparsity(local_fac, local_shape_fac, coef_vec.head(num_alpha), global_lev * coef_var, rng);
		local_lev.array() = local_fac.array() / (global_lev * coef_var.array());
		ng_mn_sparsity(group_lev, grp_vec, grp_id, local_shape, global_lev, local_fac, group_shape, group_scl, rng);
	}
	void updateImpactPrec() override {
		contem_var = contem_global_lev.replicate(1, num_lowerchol).reshaped();
		contem_fac = contem_global_lev[0] * contem_local_lev;
		contem_shape = ng_shape_jump(contem_shape, contem_fac, contem_global_lev[0], mh_sd, rng);
		ng_local_sparsity(contem_fac, contem_shape, contem_coef, contem_var, rng);
		contem_local_lev.array() *= contem_global_lev[0] / contem_fac.array();
		double old_contem_global = contem_global_lev[0];
		contem_global_lev[0] = ng_global_sparsity(contem_fac, contem_shape, contem_global_shape, contem_global_scl, rng);
		contem_fac.array() *= contem_global_lev[0] / old_contem_global;
		prior_chol_prec.diagonal() = 1 / (contem_global_lev[0] * contem_local_lev.array()).square();
	}
	void updateRecords() override {
		updateCoefRecords();
		ng_record.assignRecords(mcmc_step, local_lev, group_lev, global_lev);
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
			Rcpp::Named("lambda_record") = ng_record.local_record,
			Rcpp::Named("eta_record") = ng_record.group_record,
			Rcpp::Named("tau_record") = ng_record.global_record,
			Rcpp::Named("alpha_sparse_record") = sparse_record.coef_record,
			Rcpp::Named("a_sparse_record") = sparse_record.contem_coef_record
		);
		if (include_mean) {
			res["c_record"] = sv_record.coef_record.rightCols(dim);
		}
		for (auto& record : res) {
			if (Rcpp::is<Rcpp::NumericMatrix>(record)) {
				record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
			} else {
				record = thin_record(Rcpp::as<Eigen::VectorXd>(record), num_iter, num_burn, thin);
			}
			// record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
		}
		return res;
	}
	SsvsRecords returnSsvsRecords(int num_burn, int thin) const override {
		return SsvsRecords();
	}
	HorseshoeRecords returnHsRecords(int num_burn, int thin) const override {
		return HorseshoeRecords();
	}
	NgRecords returnNgRecords(int num_burn, int thin) const override {
		NgRecords res_record(
			thin_record(ng_record.local_record, num_iter, num_burn, thin).derived(),
			thin_record(ng_record.group_record, num_iter, num_burn, thin).derived(),
			thin_record(ng_record.global_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}

private:
	Eigen::VectorXi grp_id;
	Eigen::MatrixXi grp_mat;
	Eigen::VectorXi grp_vec;
	int num_grp;
	NgRecords ng_record;
	double mh_sd;
	// double local_shape, contem_shape;
	Eigen::VectorXd local_shape, local_shape_fac;
	Eigen::MatrixXd local_shape_loc;
	double contem_shape;
	double group_shape, group_scl, global_shape, global_scl, contem_global_shape, contem_global_scl;
	Eigen::VectorXd local_lev;
	Eigen::VectorXd group_lev;
	double global_lev;
	Eigen::VectorXd local_fac;
	Eigen::VectorXd coef_var;
	Eigen::MatrixXd coef_var_loc;
	Eigen::VectorXd contem_local_lev;
	Eigen::VectorXd contem_global_lev;
	Eigen::VectorXd contem_var;
	Eigen::VectorXd contem_fac;
};

class DirLaplaceSv : public McmcSv {
public:
	DirLaplaceSv(const DlSvParams& params, const HsSvInits& inits, unsigned int seed)
	: McmcSv(params, inits, seed),
		grp_id(params._grp_id), grp_mat(params._grp_mat), grp_vec(grp_mat.reshaped()), num_grp(grp_id.size()),
		dl_record(num_iter, num_alpha, num_grp),
		dir_concen(params._dl_concen), contem_dir_concen(params._contem_dl_concen),
		local_lev(inits._init_local), group_lev(inits._init_group), global_lev(inits._init_global),
		local_fac(Eigen::VectorXd::Zero(num_alpha)),
		latent_local(Eigen::VectorXd::Zero(num_alpha)),
		// shrink_fac(Eigen::VectorXd::Zero(num_alpha)),
		// lambda_mat(Eigen::MatrixXd::Zero(num_alpha, num_alpha)),
		coef_var(Eigen::VectorXd::Zero(num_alpha)),
		coef_var_loc(Eigen::MatrixXd::Zero(num_alpha / dim, dim)),
		contem_local_lev(inits._init_contem_local), contem_global_lev(inits._init_conetm_global),
		latent_contem_local(Eigen::VectorXd::Zero(num_lowerchol)) {
		dl_record.assignRecords(0, local_lev, group_lev, global_lev);
	}
	virtual ~DirLaplaceSv() = default;
	void updateCoefPrec() override {
		for (int j = 0; j < num_grp; j++) {
			coef_var_loc = (grp_mat.array() == grp_id[j]).select(
				group_lev[j],
				coef_var_loc
			);
		}
		coef_var = coef_var_loc.reshaped();
		// build_shrink_mat(lambda_mat, coef_var, global_lev * local_lev);
		local_fac.array() = coef_var.array() * local_lev.array();
		dl_latent(latent_local, global_lev * local_fac, coef_vec.head(num_alpha), rng);
		// lambda_mat.setZero();
		// lambda_mat.diagonal() = 1 / (global_lev * local_fac.array()).square();
		// prior_alpha_prec.topLeftCorner(num_alpha, num_alpha) = lambda_mat;
		prior_alpha_prec.topLeftCorner(num_alpha, num_alpha).diagonal() = 1 / (global_lev * local_fac.array() * latent_local.array()).square();
		// shrink_fac = 1 / (1 + lambda_mat.diagonal().array());
	}
	void updateCoefShrink() override {
		global_lev = dl_global_sparsity(local_fac, dir_concen, coef_vec.head(num_alpha), rng);
		dl_mn_sparsity(group_lev, grp_vec, grp_id, global_lev, local_lev, dir_concen, coef_vec.head(num_alpha), rng);
		dl_local_sparsity(local_lev, dir_concen, coef_vec.head(num_alpha), rng);
	}
	void updateImpactPrec() override {
		dl_latent(latent_contem_local, contem_local_lev, contem_coef, rng);
		dl_local_sparsity(contem_local_lev, contem_dir_concen, contem_coef, rng);
		contem_global_lev[0] = dl_global_sparsity(contem_local_lev, contem_dir_concen, contem_coef, rng);
		prior_chol_prec.diagonal() = 1 / (contem_global_lev[0] * contem_global_lev[0] * contem_local_lev.array().square() * latent_contem_local.array());
	}
	void updateRecords() override {
		updateCoefRecords();
		dl_record.assignRecords(mcmc_step, local_lev, group_lev, global_lev);
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
			Rcpp::Named("lambda_record") = dl_record.local_record,
			Rcpp::Named("eta_record") = dl_record.group_record,
			Rcpp::Named("tau_record") = dl_record.global_record,
			Rcpp::Named("alpha_sparse_record") = sparse_record.coef_record,
			Rcpp::Named("a_sparse_record") = sparse_record.contem_coef_record
		);
		if (include_mean) {
			res["c_record"] = sv_record.coef_record.rightCols(dim);
		}
		for (auto& record : res) {
			if (Rcpp::is<Rcpp::NumericMatrix>(record)) {
				record = thin_record(Rcpp::as<Eigen::MatrixXd>(record), num_iter, num_burn, thin);
			} else {
				record = thin_record(Rcpp::as<Eigen::VectorXd>(record), num_iter, num_burn, thin);
			}
		}
		return res;
	}
	SsvsRecords returnSsvsRecords(int num_burn, int thin) const override {
		return SsvsRecords();
	}
	HorseshoeRecords returnHsRecords(int num_burn, int thin) const override {
		return HorseshoeRecords();
	}
	NgRecords returnNgRecords(int num_burn, int thin) const override {
		NgRecords res_record(
			thin_record(dl_record.local_record, num_iter, num_burn, thin).derived(),
			thin_record(dl_record.group_record, num_iter, num_burn, thin).derived(),
			thin_record(dl_record.global_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}

private:
	Eigen::VectorXi grp_id;
	Eigen::MatrixXi grp_mat;
	Eigen::VectorXi grp_vec;
	int num_grp;
	NgRecords dl_record;
	double dir_concen, contem_dir_concen;
	Eigen::VectorXd local_lev;
	Eigen::VectorXd group_lev;
	double global_lev;
	Eigen::VectorXd local_fac;
	Eigen::VectorXd latent_local;
	// Eigen::VectorXd shrink_fac;
	// Eigen::MatrixXd lambda_mat;
	Eigen::VectorXd coef_var;
	Eigen::MatrixXd coef_var_loc;
	Eigen::VectorXd contem_local_lev;
	Eigen::VectorXd contem_global_lev;
	Eigen::VectorXd latent_contem_local;
};

} // namespace bvhar

#endif // MCMCSV_H