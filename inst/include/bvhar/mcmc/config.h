/**
 * @file bvharconfig.h
 * @author your name (you@domain.com)
 * @brief Headers including MCMC configuration structs
 */

#ifndef BVHARCONFIG_H
#define BVHARCONFIG_H

#include "bvhardraw.h"
#include "bvhardesign.h"
#include <utility>

namespace bvhar {

// Parameters
struct RegParams;
struct SvParams;
template <typename BaseRegParams> struct MinnParams;
template <typename BaseRegParams> struct HierminnParams;
template <typename BaseRegParams> struct SsvsParams;
template <typename BaseRegParams> struct HorseshoeParams;
template <typename BaseRegParams> struct NgParams;
template <typename BaseRegParams> struct DlParams;
template <typename BaseRegParams> struct GdpParams;
// Initialization
struct RegInits;
struct LdltInits;
struct SvInits;
template <typename BaseRegInits> struct HierminnInits;
template <typename BaseRegInits> struct SsvsInits;
template <typename BaseRegInits> struct GlInits;
template <typename BaseRegInits> struct HsInits;
template <typename BaseRegInits> struct NgInits;
template <typename BaseRegInits> struct GdpInits;
// MCMC records
struct RegRecords;
struct SparseRecords;
struct LdltRecords;
struct SvRecords;
struct SsvsRecords;
struct GlobalLocalRecords;
struct HorseshoeRecords;
struct NgRecords;

/**
 * @brief Hyperparameters for `McmcReg`
 * 
 */
struct RegParams {
	int _iter;
	Eigen::MatrixXd _x, _y;
	Eigen::VectorXd _sig_shp, _sig_scl, _mean_non;
	double _sd_non;
	bool _mean;
	int _dim, _dim_design, _num_design, _num_lowerchol, _num_coef, _num_alpha, _nrow;
	std::set<int> _own_id;
	std::set<int> _cross_id;
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;

	RegParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& intercept,
		bool include_mean
	)
	: _iter(num_iter), _x(x), _y(y),
		_sig_shp(CAST<Eigen::VectorXd>(spec["shape"])),
		_sig_scl(CAST<Eigen::VectorXd>(spec["scale"])),
		_mean_non(CAST<Eigen::VectorXd>(intercept["mean_non"])),
		_sd_non(CAST_DOUBLE(intercept["sd_non"])), _mean(include_mean),
		_dim(y.cols()), _dim_design(x.cols()), _num_design(y.rows()),
		_num_lowerchol(_dim * (_dim - 1) / 2), _num_coef(_dim * _dim_design),
		_num_alpha(_mean ? _num_coef - _dim : _num_coef), _nrow(_num_alpha / _dim),
		_grp_id(grp_id), _grp_mat(grp_mat) {
		set_grp_id(_own_id, _cross_id, own_id, cross_id);
	}
};

/**
 * @brief Hyperparameters for `McmcSv`
 * 
 */
struct SvParams : public RegParams {
	Eigen::VectorXd _init_mean;
	Eigen::MatrixXd _init_prec;

	SvParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& intercept,
		bool include_mean
	)
	: RegParams(num_iter, x, y, spec, own_id, cross_id, grp_id, grp_mat, intercept, include_mean),
		_init_mean(CAST<Eigen::VectorXd>(spec["initial_mean"])),
		_init_prec(CAST<Eigen::MatrixXd>(spec["initial_prec"])) {}
};

/**
 * @brief Hyperparameters for Minnesota prior `McmcMinn`
 * 
 * @tparam BaseRegParams `RegParams` or `SvParams`
 */
template <typename BaseRegParams = RegParams>
struct MinnParams : public BaseRegParams {
	Eigen::MatrixXd _prec_diag;
	Eigen::MatrixXd _prior_mean;
	Eigen::MatrixXd _prior_prec;
	MinnParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& reg_spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& priors, LIST& intercept,
		bool include_mean
	)
	: BaseRegParams(num_iter, x, y, reg_spec, own_id, cross_id, grp_id, grp_mat, intercept, include_mean),
		_prec_diag(Eigen::MatrixXd::Zero(y.cols(), y.cols())) {
		int lag = CAST_INT(priors["p"]); // append to bayes_spec, p = 3 in VHAR
		// Eigen::MatrixXd coef_ols = (x.transpose() * x).selfadjointView<Eigen::Lower>().llt().solve(x.transpose() * y);
		// Eigen::MatrixXd resid = y - x * coef_ols;
		// Eigen::VectorXd _sigma = (y.rows() >= x.cols()) ? (resid.transpose() * resid).diagonal() / (y.rows() - x.cols()) : (resid.transpose() * resid).diagonal() / y.rows();
		Eigen::VectorXd _sigma = CAST<Eigen::VectorXd>(priors["sigma"]);
		double _lambda = CAST_DOUBLE(priors["lambda"]);
		double _eps = CAST_DOUBLE(priors["eps"]);
		int dim = _sigma.size();
		Eigen::VectorXd _daily(dim);
		Eigen::VectorXd _weekly(dim);
		Eigen::VectorXd _monthly(dim);
		if (CONTAINS(priors, "delta")) {
			_daily = CAST<Eigen::VectorXd>(priors["delta"]);
			_weekly.setZero();
			_monthly.setZero();
		} else {
			_daily = CAST<Eigen::VectorXd>(priors["daily"]);
			_weekly = CAST<Eigen::VectorXd>(priors["weekly"]);
			_monthly = CAST<Eigen::VectorXd>(priors["monthly"]);
		}
		Eigen::MatrixXd dummy_response = build_ydummy(lag, _sigma, _lambda, _daily, _weekly, _monthly, false);
		Eigen::MatrixXd dummy_design = build_xdummy(
			Eigen::VectorXd::LinSpaced(lag, 1, lag),
			_lambda, _sigma, _eps, false
		);
		_prior_prec = dummy_design.transpose() * dummy_design;
		_prior_mean = _prior_prec.llt().solve(dummy_design.transpose() * dummy_response);
		_prec_diag.diagonal() = 1 / _sigma.array();
	}
};

/**
 * @brief Hyperparameters for Hierarchical Minnesota prior `McmcHierminn`
 * 
 * @tparam BaseRegParams `RegParams` or `SvParams`
 */
template <typename BaseRegParams = RegParams>
struct HierminnParams : public BaseRegParams {
	double shape;
	double rate;
	int _grid_size;
	Eigen::MatrixXd _prec_diag;
	Eigen::MatrixXd _prior_mean;
	Eigen::MatrixXd _prior_prec;
	// Eigen::MatrixXi _grp_mat;
	bool _minnesota;

	HierminnParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& reg_spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& priors, LIST& intercept,
		bool include_mean
	)
	: BaseRegParams(num_iter, x, y, reg_spec, own_id, cross_id, grp_id, grp_mat, intercept, include_mean),
		shape(CAST_DOUBLE(priors["shape"])), rate(CAST_DOUBLE(priors["rate"])), _grid_size(CAST_INT(priors["grid_size"])),
		_prec_diag(Eigen::MatrixXd::Zero(y.cols(), y.cols())) {
		int lag = CAST_INT(priors["p"]); // append to bayes_spec, p = 3 in VHAR
		// Eigen::MatrixXd coef_ols = (x.transpose() * x).selfadjointView<Eigen::Lower>().llt().solve(x.transpose() * y);
		// Eigen::MatrixXd resid = y - x * coef_ols;
		// Eigen::VectorXd _sigma = (y.rows() >= x.cols()) ? (resid.transpose() * resid).diagonal() / (y.rows() - x.cols()) : (resid.transpose() * resid).diagonal() / y.rows();
		Eigen::VectorXd _sigma = CAST<Eigen::VectorXd>(priors["sigma"]);
		double _eps = CAST_DOUBLE(priors["eps"]);
		int dim = _sigma.size();
		Eigen::VectorXd _daily(dim);
		Eigen::VectorXd _weekly(dim);
		Eigen::VectorXd _monthly(dim);
		if (CONTAINS(priors, "delta")) {
			_daily = CAST<Eigen::VectorXd>(priors["delta"]);
			_weekly.setZero();
			_monthly.setZero();
		} else {
			_daily = CAST<Eigen::VectorXd>(priors["daily"]);
			_weekly = CAST<Eigen::VectorXd>(priors["weekly"]);
			_monthly = CAST<Eigen::VectorXd>(priors["monthly"]);
		}
		Eigen::MatrixXd dummy_response = build_ydummy(lag, _sigma, 1, _daily, _weekly, _monthly, false);
		Eigen::MatrixXd dummy_design = build_xdummy(
			Eigen::VectorXd::LinSpaced(lag, 1, lag),
			1, _sigma, _eps, false
		);
		_prior_prec = dummy_design.transpose() * dummy_design;
		_prior_mean = _prior_prec.llt().solve(dummy_design.transpose() * dummy_response);
		_prec_diag.diagonal() = 1 / _sigma.array();
		// _grp_mat = grp_mat;
		_minnesota = true;
		std::set<int> unique_grp(grp_mat.data(), grp_mat.data() + grp_mat.size());
		if (unique_grp.size() == 1) {
			_minnesota = false;
		}
	}
};

/**
 * @brief Hyperparameters for SSVS prior `McmcSsvs`
 * 
 * @tparam BaseRegParams `RegParams` or `SvParams`
 */
template <typename BaseRegParams = RegParams>
struct SsvsParams : public BaseRegParams {
	// Eigen::VectorXi _grp_id;
	// Eigen::MatrixXi _grp_mat;
	Eigen::VectorXd _coef_s1, _coef_s2;
	double _contem_s1, _contem_s2;
	// double _coef_spike_scl, _contem_spike_scl;
	double _coef_slab_shape, _coef_slab_scl, _contem_slab_shape, _contem_slab_scl;
	int _coef_grid, _contem_grid;

	SsvsParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& reg_spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& ssvs_spec, LIST& intercept,
		bool include_mean
	)
	: BaseRegParams(num_iter, x, y, reg_spec, own_id, cross_id, grp_id, grp_mat, intercept, include_mean),
		// _grp_id(grp_id), _grp_mat(grp_mat),
		_coef_s1(CAST<Eigen::VectorXd>(ssvs_spec["coef_s1"])), _coef_s2(CAST<Eigen::VectorXd>(ssvs_spec["coef_s2"])),
		_contem_s1(CAST_DOUBLE(ssvs_spec["chol_s1"])), _contem_s2(CAST_DOUBLE(ssvs_spec["chol_s2"])),
		// _coef_spike_scl(CAST_DOUBLE(ssvs_spec["coef_spike_scl"])), _contem_spike_scl(CAST_DOUBLE(ssvs_spec["chol_spike_scl"])),
		_coef_slab_shape(CAST_DOUBLE(ssvs_spec["coef_slab_shape"])), _coef_slab_scl(CAST_DOUBLE(ssvs_spec["coef_slab_scl"])),
		_contem_slab_shape(CAST_DOUBLE(ssvs_spec["chol_slab_shape"])), _contem_slab_scl(CAST_DOUBLE(ssvs_spec["chol_slab_scl"])),
		_coef_grid(CAST_INT(ssvs_spec["coef_grid"])), _contem_grid(CAST_INT(ssvs_spec["chol_grid"])) {}
};

/**
 * @brief Hyperparameters for Horseshoe prior `McmcHorseshoe`
 * 
 * @tparam BaseRegParams `RegParams` or `SvParams`
 */
template <typename BaseRegParams = RegParams>
struct HorseshoeParams : public BaseRegParams {
	// Eigen::VectorXi _grp_id;
	// Eigen::MatrixXi _grp_mat;

	HorseshoeParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& reg_spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& intercept, bool include_mean
	)
	: BaseRegParams(num_iter, x, y, reg_spec, own_id, cross_id, grp_id, grp_mat, intercept, include_mean) {}
};

/**
 * @brief Hyperparameters for Normal-gamma prior `McmcNg`
 * 
 * @tparam BaseRegParams `RegParams` or `SvParams`
 */
template <typename BaseRegParams = RegParams>
struct NgParams : public BaseRegParams {
	// Eigen::VectorXi _grp_id;
	// Eigen::MatrixXi _grp_mat;
	double _mh_sd;
	double _group_shape;
	double _group_scl;
	double _global_shape;
	double _global_scl;
	double _contem_global_shape;
	double _contem_global_scl;

	NgParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& reg_spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& ng_spec, LIST& intercept,
		bool include_mean
	)
	: BaseRegParams(num_iter, x, y, reg_spec, own_id, cross_id, grp_id, grp_mat, intercept, include_mean),
		// _grp_id(grp_id), _grp_mat(grp_mat),
		_mh_sd(CAST_DOUBLE(ng_spec["shape_sd"])),
		_group_shape(CAST_DOUBLE(ng_spec["group_shape"])), _group_scl(CAST_DOUBLE(ng_spec["group_scale"])),
		_global_shape(CAST_DOUBLE(ng_spec["global_shape"])), _global_scl(CAST_DOUBLE(ng_spec["global_scale"])),
		_contem_global_shape(CAST_DOUBLE(ng_spec["contem_global_shape"])), _contem_global_scl(CAST_DOUBLE(ng_spec["contem_global_scale"])) {}
};

/**
 * @brief Hyperparameters for Dirichlet-Laplace prior `McmcDl`
 * 
 * @tparam BaseRegParams `RegParams` or `SvParams`
 */
template <typename BaseRegParams = RegParams>
struct DlParams : public BaseRegParams {
	// Eigen::VectorXi _grp_id;
	// Eigen::MatrixXi _grp_mat;
	int _grid_size;
	double _shape;
	double _scl;

	DlParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& reg_spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& dl_spec, LIST& intercept,
		bool include_mean
	)
	: BaseRegParams(num_iter, x, y, reg_spec, own_id, cross_id, grp_id, grp_mat, intercept, include_mean),
		// _grp_id(grp_id), _grp_mat(grp_mat),
		_grid_size(CAST_INT(dl_spec["grid_size"])), _shape(CAST_DOUBLE(dl_spec["shape"])), _scl(CAST_DOUBLE(dl_spec["scale"])) {}
};

/**
 * @brief Hyperparameters for GDP prior `McmcGdp`
 * 
 * @tparam BaseRegParams `RegParams` or `SvParams`
 */
template <typename BaseRegParams = RegParams>
struct GdpParams : public BaseRegParams {
	// Eigen::VectorXi _grp_id;
	// Eigen::MatrixXi _grp_mat;
	int _grid_shape;
	int _grid_rate;

	GdpParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& reg_spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& gdp_spec, LIST& intercept,
		bool include_mean
	)
	: BaseRegParams(num_iter, x, y, reg_spec, own_id, cross_id, grp_id, grp_mat, intercept, include_mean),
		// _grp_id(grp_id), _grp_mat(grp_mat),
		_grid_shape(CAST_INT(gdp_spec["grid_shape"])), _grid_rate(CAST_INT(gdp_spec["grid_rate"])) {}
};


/**
 * @brief MCMC initial values for `McmcTriangular`
 * 
 */
struct RegInits {
	Eigen::MatrixXd _coef;
	Eigen::VectorXd _contem;

	RegInits(const RegParams& params) {
		_coef = (params._x.transpose() * params._x).llt().solve(params._x.transpose() * params._y); // OLS
		int dim = params._y.cols();
		int num_lowerchol = dim * (dim - 1) / 2;
		_contem = .001 * Eigen::VectorXd::Zero(num_lowerchol);
	}

	RegInits(LIST& init)
	: _coef(CAST<Eigen::MatrixXd>(init["init_coef"])),
		_contem(CAST<Eigen::VectorXd>(init["init_contem"])) {}
};

/**
 * @brief MCMC initial values for `McmcReg`
 * 
 */
struct LdltInits : public RegInits {
	Eigen::VectorXd _diag;

	LdltInits(LIST& init)
	: RegInits(init),
		_diag(CAST<Eigen::VectorXd>(init["init_diag"])) {}
	
	LdltInits(LIST& init, int num_design)
	: RegInits(init),
		_diag(CAST<Eigen::VectorXd>(init["init_diag"])) {}
};

/**
 * @brief MCMC initial values for `McmcSv`
 * 
 */
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
	
	SvInits(LIST& init)
	: RegInits(init),
		_lvol_init(CAST<Eigen::VectorXd>(init["lvol_init"])),
		_lvol(CAST<Eigen::MatrixXd>(init["lvol"])),
		_lvol_sig(CAST<Eigen::VectorXd>(init["lvol_sig"])) {}
	
	SvInits(LIST& init, int num_design)
	: RegInits(init),
		_lvol_init(CAST<Eigen::VectorXd>(init["lvol_init"])),
		_lvol(_lvol_init.transpose().replicate(num_design, 1)),
		_lvol_sig(CAST<Eigen::VectorXd>(init["lvol_sig"])) {}
};

/**
 * @brief MCMC initial values for `McmcHierminn`
 * 
 * @tparam BaseRegInits 
 */
template <typename BaseRegInits = LdltInits>
struct HierminnInits : public BaseRegInits {
	double _own_lambda;
	double _cross_lambda;
	double _contem_lambda;

	HierminnInits(LIST& init)
	: BaseRegInits(init),
		_own_lambda(CAST_DOUBLE(init["own_lambda"])), _cross_lambda(CAST_DOUBLE(init["cross_lambda"])), _contem_lambda(CAST_DOUBLE(init["contem_lambda"])) {}
	
	HierminnInits(LIST& init, int num_design)
	: BaseRegInits(init, num_design),
		_own_lambda(CAST_DOUBLE(init["own_lambda"])), _cross_lambda(CAST_DOUBLE(init["cross_lambda"])), _contem_lambda(CAST_DOUBLE(init["contem_lambda"])) {}
};

/**
 * @brief MCMC initial values for `McmcSsvs`
 * 
 * @tparam BaseRegInits `LdltInits` or `SvInits`
 */
template <typename BaseRegInits = LdltInits>
struct SsvsInits : public BaseRegInits {
	Eigen::VectorXd _coef_dummy;
	Eigen::VectorXd _coef_weight; // in SsvsSvParams: move coef_mixture and chol_mixture in set_ssvs()?
	Eigen::VectorXd _contem_weight; // in SsvsSvParams
	Eigen::VectorXd _coef_slab;
	Eigen::VectorXd _contem_slab;
	double _coef_spike_scl, _contem_spike_scl;
	
	SsvsInits(LIST& init)
	: BaseRegInits(init),
		_coef_dummy(CAST<Eigen::VectorXd>(init["init_coef_dummy"])),
		_coef_weight(CAST<Eigen::VectorXd>(init["coef_mixture"])),
		_contem_weight(CAST<Eigen::VectorXd>(init["chol_mixture"])),
		_coef_slab(CAST<Eigen::VectorXd>(init["coef_slab"])),
		_contem_slab(CAST<Eigen::VectorXd>(init["contem_slab"])),
		_coef_spike_scl(CAST_DOUBLE(init["coef_spike_scl"])),
		_contem_spike_scl(CAST_DOUBLE(init["chol_spike_scl"])) {}
	
	SsvsInits(LIST& init, int num_design)
	: BaseRegInits(init, num_design),
		_coef_dummy(CAST<Eigen::VectorXd>(init["init_coef_dummy"])),
		_coef_weight(CAST<Eigen::VectorXd>(init["coef_mixture"])),
		_contem_weight(CAST<Eigen::VectorXd>(init["chol_mixture"])),
		_coef_slab(CAST<Eigen::VectorXd>(init["coef_slab"])),
		_contem_slab(CAST<Eigen::VectorXd>(init["contem_slab"])),
		_coef_spike_scl(CAST_DOUBLE(init["coef_spike_scl"])),
		_contem_spike_scl(CAST_DOUBLE(init["chol_spike_scl"])) {}
};

/**
 * @brief MCMC initial values for global-local shrinkage prior.
 * `McmcDl` takes this.
 * 
 * @tparam BaseRegInits `LdldInits` or `SvInits` 
 */
template <typename BaseRegInits = LdltInits>
struct GlInits : public BaseRegInits {
	Eigen::VectorXd _init_local;
	double _init_global;
	Eigen::VectorXd _init_contem_local;
	Eigen::VectorXd _init_conetm_global;
	
	GlInits(LIST& init)
	: BaseRegInits(init),
		_init_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_init_global(CAST_DOUBLE(init["global_sparsity"])),
		_init_contem_local(CAST<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_conetm_global(CAST<Eigen::VectorXd>(init["contem_global_sparsity"])) {}
	
	GlInits(LIST& init, int num_design)
	: BaseRegInits(init, num_design),
		_init_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_init_global(CAST_DOUBLE(init["global_sparsity"])),
		_init_contem_local(CAST<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_conetm_global(CAST<Eigen::VectorXd>(init["contem_global_sparsity"])) {}
};

/**
 * @brief MCMC initial values for `McmcHorseshoe`
 * 
 * @tparam BaseRegInits `LdltInits` or `SvInits`
 */
template <typename BaseRegInits = LdltInits>
struct HsInits : public GlInits<BaseRegInits> {
	Eigen::VectorXd _init_group;
	
	HsInits(LIST& init)
	: GlInits<BaseRegInits>(init),
		_init_group(CAST<Eigen::VectorXd>(init["group_sparsity"])) {}
	
	HsInits(LIST& init, int num_design)
	: GlInits<BaseRegInits>(init, num_design),
		_init_group(CAST<Eigen::VectorXd>(init["group_sparsity"])) {}
};

/**
 * @brief MCMC initial values for `McmcNg`
 * 
 * @tparam BaseRegInits `LdltInits` or `SvInits`
 */
template <typename BaseRegInits = LdltInits>
struct NgInits : public HsInits<BaseRegInits> {
	Eigen::VectorXd _init_local_shape;
	double _init_contem_shape;

	NgInits(LIST& init)
	: HsInits<BaseRegInits>(init),
		_init_local_shape(CAST<Eigen::VectorXd>(init["local_shape"])),
		_init_contem_shape(CAST_DOUBLE(init["contem_shape"])) {}
	
	NgInits(LIST& init, int num_design)
	: HsInits<BaseRegInits>(init, num_design),
		_init_local_shape(CAST<Eigen::VectorXd>(init["local_shape"])),
		_init_contem_shape(CAST_DOUBLE(init["contem_shape"])) {}
};

/**
 * @brief MCMC initial values for `McmcGdp`
 * 
 * @tparam BaseRegInits `LdltInits` or `SvInits`
 */
template <typename BaseRegInits = LdltInits>
struct GdpInits : public BaseRegInits {
	Eigen::VectorXd _init_local;
	Eigen::VectorXd _init_group_rate;
	Eigen::VectorXd _init_contem_local;
	Eigen::VectorXd _init_contem_rate;
	double _init_gamma_shape, _init_gamma_rate, _init_contem_gamma_shape, _init_contem_gamma_rate;

	GdpInits(LIST& init)
	: BaseRegInits(init),
		_init_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_init_group_rate(CAST<Eigen::VectorXd>(init["group_rate"])),
		_init_contem_local(CAST<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_contem_rate(CAST<Eigen::VectorXd>(init["contem_rate"])),
		_init_gamma_shape(CAST_DOUBLE(init["gamma_shape"])), _init_gamma_rate(CAST_DOUBLE(init["gamma_rate"])),
		_init_contem_gamma_shape(CAST_DOUBLE(init["contem_gamma_shape"])), _init_contem_gamma_rate(CAST_DOUBLE(init["contem_gamma_rate"])) {}
	
	GdpInits(LIST& init, int num_design)
	: BaseRegInits(init, num_design),
		_init_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_init_group_rate(CAST<Eigen::VectorXd>(init["group_rate"])),
		_init_contem_local(CAST<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_contem_rate(CAST<Eigen::VectorXd>(init["contem_rate"])),
		_init_gamma_shape(CAST_DOUBLE(init["gamma_shape"])), _init_gamma_rate(CAST_DOUBLE(init["gamma_rate"])),
		_init_contem_gamma_shape(CAST_DOUBLE(init["contem_gamma_shape"])), _init_contem_gamma_rate(CAST_DOUBLE(init["contem_gamma_rate"])) {}
};

/**
 * @brief MCMC records for `McmcTriangular`
 * 
 */
struct RegRecords {
	Eigen::MatrixXd coef_record; // alpha in VAR
	Eigen::MatrixXd contem_coef_record; // a = a21, a31, a32, ..., ak1, ..., ak(k-1)

	RegRecords() : coef_record(), contem_coef_record() {}

	RegRecords(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_coef)),
		contem_coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol)) {}
	
	RegRecords(const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& a_record)
	: coef_record(alpha_record), contem_coef_record(a_record) {}

	virtual ~RegRecords() = default;

	/**
	 * @brief Assign MCMC draw to the draw matrix
	 * 
	 * @param id MCMC step
	 * @param coef_vec Coefficient vector draw
	 * @param contem_coef Contemporaneous coefficient draw
	 */
	void assignRecords(int id, const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef) {
		coef_record.row(id) = coef_vec;
		contem_coef_record.row(id) = contem_coef;
	}
	
	/**
	 * @copydoc assignRecords(int, const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& coef_vec)
	 * 
	 * @param diag_vec Diagonal term draw
	 */
	virtual void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef, const Eigen::VectorXd& diag_vec
	) = 0;

	/**
	 * @copydoc assignRecords(int, const Eigen::VectorXd&, const Eigen::VectorXd&)
	 * 
	 * @param lvol_draw Log volatilities draw
	 * @param lvol_sig Variance draw of log volatilities
	 * @param lvol_init Initial log volatlity draw
	 */
	virtual void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef,
		const Eigen::MatrixXd& lvol_draw, const Eigen::VectorXd& lvol_sig, const Eigen::VectorXd& lvol_init
	) = 0;

	/**
	 * @brief Return the MCMC record `LIST`
	 * 
	 * @param dim Time series dimension
	 * @param num_alpha The number of coefficient elements except constant term
	 * @param include_mean If `true`, constant term is included
	 * @return LIST A `LIST` containing MCMC records. If `include_mean` is `true`, it also includes a constant term record.
	 */
	LIST returnListRecords(int dim, int num_alpha, bool include_mean) const {
		LIST res = CREATE_LIST(
			NAMED("alpha_record") = coef_record.leftCols(num_alpha),
			NAMED("a_record") = contem_coef_record
		);
		if (include_mean) {
			res["c_record"] = CAST_MATRIX(coef_record.rightCols(dim));
		}
		return res;
	}

	/**
	 * @brief Append records to the MCMC record `LIST`
	 * 
	 * @param list MCMC record `LIST`
	 */
	virtual void appendRecords(LIST& list) = 0;

	/**
	 * @brief Return `LdltRecords`
	 * 
	 * @param sparse_record `SparseRecords` object
	 * @param num_iter Number of MCMC iteration
	 * @param num_burn Number of burn-in
	 * @param thin Thinning
	 * @param sparse If `true`, return sparsified draws.
	 * @return LdltRecords `LdltRecords` object
	 */
	virtual LdltRecords returnLdltRecords(const SparseRecords& sparse_record, int num_iter, int num_burn, int thin, bool sparse) const = 0;

	/**
	 * @brief Return `SvRecords`
	 * 
	 * @param sparse_record `SparseRecords` object
	 * @param num_iter Number of MCMC iteration
	 * @param num_burn Number of burn-in
	 * @param thin Thinning
	 * @param sparse If `true`, return sparsified draws.
	 * @return SvRecords `SvRecords` object
	 */
	virtual SvRecords returnSvRecords(const SparseRecords& sparse_record, int num_iter, int num_burn, int thin, bool sparse) const = 0;

	/**
	 * @brief Return `LdltRecords` or `SvRecords`
	 * 
	 * @tparam RecordType `LdltRecords` or `SvRecords`
	 * @param sparse_record `SparseRecords` object
	 * @param num_iter Number of MCMC iteration
	 * @param num_burn Number of burn-in
	 * @param thin Thinning
	 * @param sparse If `true`, return sparsified draws.
	 * @return RecordType `LdltRecords` or `SvRecords`
	 */
	template <typename RecordType = LdltRecords>
	RecordType returnRecords(const SparseRecords& sparse_record, int num_iter, int num_burn, int thin, bool sparse) const;

	/**
	 * @brief Get the dimension
	 * 
	 * @return int Time series dimension
	 */
	virtual int getDim() = 0;

	/**
	 * @brief Update parameters in D
	 * 
	 * @param i MCMC step
	 * @param sv_update State vector draw
	 */
	virtual void updateDiag(int i, Eigen::Ref<Eigen::VectorXd> sv_update) = 0;

	/**
	 * @brief Update parameters in D
	 * 
	 * @param i MCMC step
	 * @param id Timestamp
	 * @param sv_update State vector draw
	 */
	virtual void updateDiag(int i, int id, Eigen::Ref<Eigen::VectorXd> sv_update) = 0;

	/**
	 * @copydoc updateDiag(int, Eigen::Ref<Eigen::VectorXd>)
	 * 
	 * @param sv_sig Variance draw of AR log volatilties
	 */
	virtual void updateDiag(int i, Eigen::Ref<Eigen::VectorXd> sv_update, Eigen::Ref<Eigen::VectorXd> sv_sig) = 0;

	/**
	 * @brief Remove unstable coefficients draw
	 * 
	 * @param num_alpha The number of coefficient elements except constant term
	 * @param threshold Threashold to check stability
	 */
	virtual void subsetStable(int num_alpha, double threshold) = 0;

	/**
	 * @copydoc subsetStable(int, double)
	 * 
	 * @param har_trans Dense VHAR transformation matrix
	 */
	virtual void subsetStable(int num_alpha, double threshold, Eigen::Ref<const Eigen::MatrixXd> har_trans) = 0;

	/**
	 * @copydoc subsetStable(int, double)
	 * 
	 * @param har_trans Sprase VHAR transformation matrix
	 */
	virtual void subsetStable(int num_alpha, double threshold, Eigen::Ref<const Eigen::SparseMatrix<double>> har_trans) = 0;

	/**
	 * @brief Get sparse draw using credible interval
	 * 
	 * @param level Credible interval level
	 * @return Eigen::VectorXd Vector of 0 or 1. 0 if the credible interval includes 0.
	 */
	Eigen::VectorXd computeActivity(double level) {
		Eigen::VectorXd lower_ci(coef_record.cols());
		Eigen::VectorXd upper_ci(coef_record.cols());
		Eigen::VectorXd selection(coef_record.cols());
		for (int i = 0; i < coef_record.cols(); ++i) {
			// lower_ci[i] = quantile_lower(coef_record.col(i), level / 2);
			// upper_ci[i] = quantile_upper(coef_record.col(i), 1 - level / 2);
			selection[i] = quantile_lower(coef_record.col(i), level / 2) * quantile_upper(coef_record.col(i), 1 - level / 2) < 0 ? 0.0 : 1.1;
		}
		return selection;
	}
};

/**
 * @brief Signal adaptive variable selector records
 * 
 */
struct SparseRecords {
	Eigen::MatrixXd coef_record;
	Eigen::MatrixXd contem_coef_record;

	SparseRecords() : coef_record(), contem_coef_record() {}

	SparseRecords(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_coef)),
		contem_coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol)) {}
	
	SparseRecords(const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& a_record)
	: coef_record(alpha_record), contem_coef_record(a_record) {}
	
	/**
	 * @brief Assign MCMC draw to the draw matrix
	 * 
	 * @param id MCMC step
	 * @param coef_mat Coefficient matrix processed by SAVS
	 * @param contem_coef Contemporaneous vector processed by SAVS
	 */
	void assignRecords(int id, const Eigen::MatrixXd& coef_mat, const Eigen::VectorXd& contem_coef) {
		coef_record.row(id) = coef_mat.reshaped();
		contem_coef_record.row(id) = contem_coef;
	}

	/**
	 * @copydoc assignRecords(int, const Eigen::MatrixXd&, const Eigen::VectorXd&)
	 * 
	 * @param num_alpha The number of coefficient elements except constant term
	 * @param dim Time series dimension
	 * @param nrow_coef Number of rows of coefficient matrix
	 */
	void assignRecords(int id, int num_alpha, int dim, int nrow_coef, const Eigen::MatrixXd& coef_mat, const Eigen::VectorXd& contem_coef) {
		if (coef_mat.size() == num_alpha) {
			coef_record.row(id) = coef_mat.reshaped();
		} else {
			coef_record.row(id).head(num_alpha) = coef_mat.topRows(nrow_coef).reshaped();
			coef_record.row(id).tail(dim) = coef_mat.bottomRows(1).reshaped();
		}
		contem_coef_record.row(id) = contem_coef;
	}

	/**
	 * @brief Append sparse records to the MCMC record `LIST`
	 * 
	 * @param list MCMC record `LIST`
	 * @param dim Time series dimension
	 * @param num_alpha The number of coefficient elements except constant term
	 * @param include_mean If `true`, constant term is included
	 */
	void appendRecords(LIST& list, int dim, int num_alpha, bool include_mean) {
		list["alpha_sparse_record"] = CAST_MATRIX(coef_record.leftCols(num_alpha));
		list["a_sparse_record"] = contem_coef_record;
		if (include_mean) {
			list["c_sparse_record"] = CAST_MATRIX(coef_record.rightCols(dim));
		}
	}
};

/**
 * @brief MCMC records for `McmcReg`
 * 
 */
struct LdltRecords : public RegRecords {
	Eigen::MatrixXd fac_record; // d_1, ..., d_m in D of LDLT

	LdltRecords() : RegRecords(), fac_record() {}

	LdltRecords(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: RegRecords(num_iter, dim, num_design, num_coef, num_lowerchol),
		fac_record(Eigen::MatrixXd::Zero(num_iter + 1, dim)) {}
	
	LdltRecords(const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& d_record)
	: RegRecords(alpha_record, a_record), fac_record(d_record) {}

	LdltRecords(
		const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& c_record,
		const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& d_record
	)
	: RegRecords(Eigen::MatrixXd::Zero(alpha_record.rows(), alpha_record.cols() + c_record.cols()), a_record),
		fac_record(d_record) {
		coef_record << alpha_record, c_record;
	}

	virtual ~LdltRecords() = default;

	void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef, const Eigen::VectorXd& diag_vec
	) override {
		coef_record.row(id) = coef_vec;
		contem_coef_record.row(id) = contem_coef;
		fac_record.row(id) = diag_vec.array();
	}

	void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef,
		const Eigen::MatrixXd& lvol_draw, const Eigen::VectorXd& lvol_sig, const Eigen::VectorXd& lvol_init
	) override {}

	void appendRecords(LIST& list) override {
		list["d_record"] = fac_record;
	}

	LdltRecords returnLdltRecords(const SparseRecords& sparse_record, int num_iter, int num_burn, int thin, bool sparse) const override;
	SvRecords returnSvRecords(const SparseRecords& sparse_record, int num_iter, int num_burn, int thin, bool sparse) const override;

	int getDim() override {
		return fac_record.cols();
	}

	void updateDiag(int i, Eigen::Ref<Eigen::VectorXd> sv_update) override {
		sv_update = fac_record.row(i).transpose().cwiseSqrt(); // D^1/2
	}
	void updateDiag(int i, int id, Eigen::Ref<Eigen::VectorXd> sv_update) override {
		sv_update = fac_record.row(i).transpose().cwiseSqrt(); // D^1/2
	}
	void updateDiag(int i, Eigen::Ref<Eigen::VectorXd> sv_update, Eigen::Ref<Eigen::VectorXd> sv_sig) override {}

	void subsetStable(int num_alpha, double threshold) override {
		int dim = fac_record.cols();
		int nrow_coef = num_alpha / dim;
		std::vector<int> stable_id;
		for (int i = 0; i < coef_record.rows(); ++i) {
			if (is_stable(coef_record.row(i).head(num_alpha).reshaped(nrow_coef, dim), threshold)) {
				stable_id.push_back(i);
			}
		}
		coef_record = std::move(coef_record(stable_id, Eigen::all));
		contem_coef_record = std::move(contem_coef_record(stable_id, Eigen::all));
		fac_record = std::move(fac_record(stable_id, Eigen::all));
	}

	void subsetStable(int num_alpha, double threshold, Eigen::Ref<const Eigen::MatrixXd> har_trans) override {
		int dim = fac_record.cols();
		int nrow_coef = num_alpha / dim;
		std::vector<int> stable_id;
		// Eigen::MatrixXd var_record = coef_record.leftCols(num_alpha) * kronecker_eigen(Eigen::MatrixXd::Identity(dim, dim), har_trans);
		for (int i = 0; i < coef_record.rows(); ++i) {
			// if (is_stable(var_record.row(i).reshaped(nrow_coef, dim))) {
			// 	stable_id.push_back(i);
			// }
			if (is_stable(coef_record.row(i).head(num_alpha).reshaped(nrow_coef, dim), threshold, har_trans)) {
				stable_id.push_back(i);
			}
		}
		coef_record = std::move(coef_record(stable_id, Eigen::all));
		contem_coef_record = std::move(contem_coef_record(stable_id, Eigen::all));
		fac_record = std::move(fac_record(stable_id, Eigen::all));
	}

	void subsetStable(int num_alpha, double threshold, Eigen::Ref<const Eigen::SparseMatrix<double>> har_trans) override {
		int dim = fac_record.cols();
		int nrow_coef = num_alpha / dim;
		std::vector<int> stable_id;
		for (int i = 0; i < coef_record.rows(); ++i) {
			if (is_stable(har_trans.transpose() * coef_record.row(i).head(num_alpha).reshaped(nrow_coef, dim), threshold)) {
				stable_id.push_back(i);
			}
		}
		coef_record = std::move(coef_record(stable_id, Eigen::all));
		contem_coef_record = std::move(contem_coef_record(stable_id, Eigen::all));
		fac_record = std::move(fac_record(stable_id, Eigen::all));
	}
};

struct SvRecords : public RegRecords {
	Eigen::MatrixXd lvol_sig_record; // sigma_h^2 = (sigma_(h1i)^2, ..., sigma_(hki)^2)
	Eigen::MatrixXd lvol_init_record; // h0 = h10, ..., hk0
	Eigen::MatrixXd lvol_record; // time-varying h = (h_1, ..., h_k) with h_j = (h_j1, ..., h_jn), row-binded
	
	SvRecords() : RegRecords(), lvol_sig_record(), lvol_init_record(), lvol_record() {}

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

	virtual ~SvRecords() = default;

	void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef, const Eigen::VectorXd& diag_vec
	) override {}

	void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef,
		const Eigen::MatrixXd& lvol_draw, const Eigen::VectorXd& lvol_sig, const Eigen::VectorXd& lvol_init
	) override {
		coef_record.row(id) = coef_vec;
		contem_coef_record.row(id) = contem_coef;
		lvol_record.row(id) = lvol_draw.transpose().reshaped();
		lvol_sig_record.row(id) = lvol_sig;
		lvol_init_record.row(id) = lvol_init;
	}

	void appendRecords(LIST& list) override {
		list["h_record"] = lvol_record;
		list["h0_record"] = lvol_init_record;
		list["sigh_record"] = lvol_sig_record;
	}

	LdltRecords returnLdltRecords(const SparseRecords& sparse_record, int num_iter, int num_burn, int thin, bool sparse) const override;
	SvRecords returnSvRecords(const SparseRecords& sparse_record, int num_iter, int num_burn, int thin, bool sparse) const override;

	int getDim() override {
		return lvol_sig_record.cols();
	}

	void updateDiag(int i, Eigen::Ref<Eigen::VectorXd> sv_update) override {
		int dim = getDim();
		int num_design = lvol_record.cols() / dim;
		sv_update.setZero();
		for (int id = 0; id < num_design; ++id) {
			sv_update += (lvol_record.block(i, id * dim, 1, dim) / 2).array().exp().matrix();
		}
		sv_update /= num_design;
	}

	void updateDiag(int i, int id, Eigen::Ref<Eigen::VectorXd> sv_update) override {
		if (id >= 0) {
			sv_update = (lvol_record.middleCols(id * getDim(), getDim()).row(i) / 2).array().exp().matrix();
		} else {
			updateDiag(i, sv_update);
		}
	}

	void updateDiag(int i, Eigen::Ref<Eigen::VectorXd> sv_update, Eigen::Ref<Eigen::VectorXd> sv_sig) override {
		sv_update = lvol_record.rightCols(lvol_sig_record.cols()).row(i).transpose();
		sv_sig = lvol_sig_record.row(i).cwiseSqrt();
	}

	void subsetStable(int num_alpha, double threshold) override {
		int dim = lvol_sig_record.cols();
		int nrow_coef = num_alpha / dim;
		std::vector<int> stable_id;
		for (int i = 0; i < coef_record.rows(); ++i) {
			if (is_stable(coef_record.row(i).head(num_alpha).reshaped(nrow_coef, dim), threshold)) {
				stable_id.push_back(i);
			}
		}
		coef_record = std::move(coef_record(stable_id, Eigen::all));
		contem_coef_record = std::move(contem_coef_record(stable_id, Eigen::all));
		lvol_record = std::move(lvol_record(stable_id, Eigen::all));
		lvol_sig_record = std::move(lvol_sig_record(stable_id, Eigen::all));
		lvol_init_record = std::move(lvol_init_record(stable_id, Eigen::all));
	}

	void subsetStable(int num_alpha, double threshold, Eigen::Ref<const Eigen::MatrixXd> har_trans) override {
		int dim = lvol_sig_record.cols();
		int nrow_coef = num_alpha / dim;
		std::vector<int> stable_id;
		Eigen::MatrixXd var_record = coef_record.leftCols(num_alpha) * kronecker_eigen(Eigen::MatrixXd::Identity(dim, dim), har_trans);
		for (int i = 0; i < coef_record.rows(); ++i) {
			if (is_stable(var_record.row(i).reshaped(nrow_coef, dim), threshold)) {
				stable_id.push_back(i);
			}
		}
		coef_record = std::move(coef_record(stable_id, Eigen::all));
		contem_coef_record = std::move(contem_coef_record(stable_id, Eigen::all));
		lvol_record = std::move(lvol_record(stable_id, Eigen::all));
		lvol_sig_record = std::move(lvol_sig_record(stable_id, Eigen::all));
		lvol_init_record = std::move(lvol_init_record(stable_id, Eigen::all));
	}

	void subsetStable(int num_alpha, double threshold, Eigen::Ref<const Eigen::SparseMatrix<double>> har_trans) override {
		int dim = lvol_sig_record.cols();
		int nrow_coef = num_alpha / dim;
		std::vector<int> stable_id;
		for (int i = 0; i < coef_record.rows(); ++i) {
			if (is_stable(har_trans.transpose() * coef_record.row(i).head(num_alpha).reshaped(nrow_coef, dim), threshold)) {
				stable_id.push_back(i);
			}
		}
		coef_record = std::move(coef_record(stable_id, Eigen::all));
		contem_coef_record = std::move(contem_coef_record(stable_id, Eigen::all));
		lvol_record = std::move(lvol_record(stable_id, Eigen::all));
		lvol_sig_record = std::move(lvol_sig_record(stable_id, Eigen::all));
		lvol_init_record = std::move(lvol_init_record(stable_id, Eigen::all));
	}
};

/**
 * @brief MCMC records for `McmcSsvs`
 * 
 */
struct SsvsRecords {
	Eigen::MatrixXd coef_dummy_record;
	Eigen::MatrixXd coef_weight_record;
	Eigen::MatrixXd contem_dummy_record;
	Eigen::MatrixXd contem_weight_record;

	SsvsRecords() : coef_dummy_record(), coef_weight_record(), contem_dummy_record(), contem_weight_record() {}
	
	SsvsRecords(int num_iter, int num_alpha, int num_grp, int num_lowerchol)
	: coef_dummy_record(Eigen::MatrixXd::Ones(num_iter + 1, num_alpha)),
		coef_weight_record(Eigen::MatrixXd::Zero(num_iter + 1, num_grp)),
		contem_dummy_record(Eigen::MatrixXd::Ones(num_iter + 1, num_lowerchol)),
		contem_weight_record(Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol)) {}
	
	SsvsRecords(
		const Eigen::MatrixXd& coef_dummy_record, const Eigen::MatrixXd& coef_weight_record,
		const Eigen::MatrixXd& contem_dummy_record, const Eigen::MatrixXd& contem_weight_record
	)
	: coef_dummy_record(coef_dummy_record), coef_weight_record(coef_weight_record),
		contem_dummy_record(contem_dummy_record), contem_weight_record(contem_weight_record) {}
	
	/**
	 * @brief Assign MCMC draw to the draw matrix
	 * 
	 * @param id MCMC step
	 * @param coef_dummy Dummy parameter corresponding to coefficient draw
	 * @param coef_weight Weight parameter corresponding to coefficient draw
	 * @param contem_dummy Dummy parameter corresponding to contemporaneous coefficient draw
	 * @param contem_weight Weight parameter corresponding to contemporaneous coefficient draw
	 */
	void assignRecords(int id, const Eigen::VectorXd& coef_dummy, const Eigen::VectorXd& coef_weight, const Eigen::VectorXd& contem_dummy, const Eigen::VectorXd& contem_weight) {
		coef_dummy_record.row(id) = coef_dummy;
		coef_weight_record.row(id) = coef_weight;
		contem_dummy_record.row(id) = contem_dummy;
		contem_weight_record.row(id) = contem_weight;
	}

	/**
	 * @brief Return `SsvsRecords`
	 * 
	 * @param num_iter Number of MCMC iteration
	 * @param num_burn Number of burn-in
	 * @param thin Thinning
	 * @return SsvsRecords `SsvsRecords`
	 */
	SsvsRecords returnRecords(int num_iter, int num_burn, int thin) const {
		SsvsRecords res_record(
			thin_record(coef_dummy_record, num_iter, num_burn, thin).derived(),
			thin_record(coef_weight_record, num_iter, num_burn, thin).derived(),
			thin_record(contem_dummy_record, num_iter, num_burn, thin).derived(),
			thin_record(contem_weight_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
};

/**
 * @brief MCMC records for global-local shrinkage prior
 * `McmcDl` directly uses this.
 * 
 */
struct GlobalLocalRecords {
	Eigen::MatrixXd local_record;
	Eigen::VectorXd global_record;

	GlobalLocalRecords() : local_record(), global_record() {}
	
	GlobalLocalRecords(int num_iter, int num_alpha)
	: local_record(Eigen::MatrixXd::Zero(num_iter + 1, num_alpha)),
		global_record(Eigen::VectorXd::Zero(num_iter + 1)) {}
	
	GlobalLocalRecords(const Eigen::MatrixXd& local_record, const Eigen::VectorXd& global_record)
	: local_record(local_record), global_record(global_record) {}
	
	/**
	 * @brief Assign MCMC draw to the draw matrix
	 * 
	 * @param id MCMC step
	 * @param local_lev Local shrinkage parameter draw
	 * @param global_lev Global shrinkage parameter draw
	 */
	virtual void assignRecords(int id, const Eigen::VectorXd& local_lev, const double global_lev) {
		local_record.row(id) = local_lev;
		global_record[id] = global_lev;
	}

	/**
	 * @copydoc assignRecords(int, const Eigen::VectorXd&, const double)
	 * 
	 * @param id MCMC step
	 * @param group_lev Group shrinkage parameter draw
	 */
	virtual void assignRecords(int id, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) {
		assignRecords(id, local_lev, global_lev);
	}

	/**
	 * @copydoc assignRecords(int, const Eigen::VectorXd&, const double)
	 * 
	 * @param id MCMC step
	 * @param shrink_fac Shrinkage factor draw
	 */
	virtual void assignRecords(int id, const Eigen::VectorXd& shrink_fac, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) {
		assignRecords(id, local_lev, global_lev);
	}

	/**
	 * @brief Return `GlobalLocalRecords`
	 * 
	 * @param num_iter Number of MCMC iteration
	 * @param num_burn Number of burn-in
	 * @param thin Thinning
	 * @return GlobalLocalRecords `GlobalLocalRecords`
	 */
	GlobalLocalRecords returnGlRecords(int num_iter, int num_burn, int thin) const {
		GlobalLocalRecords res_record(
			thin_record(local_record, num_iter, num_burn, thin).derived(),
			thin_record(global_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
};

/**
 * @brief MCMC records for `McmcHorseshoe`
 * 
 */
struct HorseshoeRecords : public GlobalLocalRecords {
	Eigen::MatrixXd group_record;
	Eigen::MatrixXd shrink_record;

	HorseshoeRecords() : GlobalLocalRecords(), group_record(), shrink_record() {}
	
	HorseshoeRecords(int num_iter, int num_alpha, int num_grp)
	: GlobalLocalRecords(num_iter, num_alpha),
		group_record(Eigen::MatrixXd::Zero(num_iter + 1, num_grp)),
		shrink_record(Eigen::MatrixXd::Zero(num_iter + 1, num_alpha)) {}
	
	HorseshoeRecords(const Eigen::MatrixXd& local_record, const Eigen::MatrixXd& group_record, const Eigen::VectorXd& global_record, const Eigen::MatrixXd& shrink_record)
	: GlobalLocalRecords(local_record, global_record),
		group_record(group_record), shrink_record(shrink_record) {}
	
	void assignRecords(int id, const Eigen::VectorXd& shrink_fac, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) override {
		shrink_record.row(id) = shrink_fac;
		local_record.row(id) = local_lev;
		group_record.row(id) = group_lev;
		global_record[id] = global_lev;
	}

	void assignRecords(int id, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) override {}

	/**
	 * @brief Return `HorseshoeRecords`
	 * 
	 * @param num_iter Number of MCMC iteration
	 * @param num_burn Number of burn-in
	 * @param thin Thinning
	 * @return HorseshoeRecords `HorseshoeRecords`
	 */
	HorseshoeRecords returnHsRecords(int num_iter, int num_burn, int thin) const {
		HorseshoeRecords res_record(
			thin_record(local_record, num_iter, num_burn, thin).derived(),
			thin_record(group_record, num_iter, num_burn, thin).derived(),
			thin_record(global_record, num_iter, num_burn, thin).derived(),
			thin_record(shrink_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
};

/**
 * @brief MCMC records for `McmcNg`
 * 
 */
struct NgRecords : public GlobalLocalRecords {
	Eigen::MatrixXd group_record;

	NgRecords() : GlobalLocalRecords(), group_record() {}
	
	NgRecords(int num_iter, int num_alpha, int num_grp)
	: GlobalLocalRecords(num_iter, num_alpha),
		group_record(Eigen::MatrixXd::Zero(num_iter + 1, num_grp)) {}
	
	NgRecords(const Eigen::MatrixXd& local_record, const Eigen::MatrixXd& group_record, const Eigen::VectorXd& global_record)
	: GlobalLocalRecords(local_record, global_record), group_record(group_record) {}
	
	void assignRecords(int id, const Eigen::VectorXd& shrink_fac, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) override {}
	
	void assignRecords(int id, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) override {
		local_record.row(id) = local_lev;
		group_record.row(id) = group_lev;
		global_record[id] = global_lev;
	}

	/**
	 * @brief Return `NgRecords`
	 * 
	 * @param num_iter Number of MCMC iteration
	 * @param num_burn Number of burn-in
	 * @param thin Thinning
	 * @return NgRecords `NgRecords`
	 */
	NgRecords returnNgRecords(int num_iter, int num_burn, int thin) const {
		NgRecords res_record(
			thin_record(local_record, num_iter, num_burn, thin).derived(),
			thin_record(group_record, num_iter, num_burn, thin).derived(),
			thin_record(global_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
};

inline LdltRecords LdltRecords::returnLdltRecords(const SparseRecords& sparse_record, int num_iter, int num_burn, int thin, bool sparse) const {
	if (sparse) {
		return LdltRecords(
			thin_record(sparse_record.coef_record, num_iter, num_burn, thin).derived(),
			thin_record(sparse_record.contem_coef_record, num_iter, num_burn, thin).derived(),
			thin_record(fac_record, num_iter, num_burn, thin).derived()
		);
	}
	LdltRecords res_record(
		thin_record(coef_record, num_iter, num_burn, thin).derived(),
		thin_record(contem_coef_record, num_iter, num_burn, thin).derived(),
		thin_record(fac_record, num_iter, num_burn, thin).derived()
	);
	return res_record;
}

inline SvRecords LdltRecords::returnSvRecords(const SparseRecords& sparse_record, int num_iter, int num_burn, int thin, bool sparse) const {
	return SvRecords();
}

inline LdltRecords SvRecords::returnLdltRecords(const SparseRecords& sparse_record, int num_iter, int num_burn, int thin, bool sparse) const {
	return LdltRecords();
}

inline SvRecords SvRecords::returnSvRecords(const SparseRecords& sparse_record, int num_iter, int num_burn, int thin, bool sparse) const {
	if (sparse) {
		return SvRecords(
			thin_record(sparse_record.coef_record, num_iter, num_burn, thin).derived(),
			thin_record(lvol_record, num_iter, num_burn, thin).derived(),
			thin_record(sparse_record.contem_coef_record, num_iter, num_burn, thin).derived(),
			thin_record(lvol_sig_record, num_iter, num_burn, thin).derived()
		);
	}
	SvRecords res_record(
		thin_record(coef_record, num_iter, num_burn, thin).derived(),
		thin_record(lvol_record, num_iter, num_burn, thin).derived(),
		thin_record(contem_coef_record, num_iter, num_burn, thin).derived(),
		thin_record(lvol_sig_record, num_iter, num_burn, thin).derived()
	);
	return res_record;
}

template<>
inline LdltRecords RegRecords::returnRecords(const SparseRecords& sparse_record, int num_iter, int num_burn, int thin, bool sparse) const {
	return returnLdltRecords(sparse_record, num_iter, num_burn, thin, sparse);
}

template<>
inline SvRecords RegRecords::returnRecords(const SparseRecords& sparse_record, int num_iter, int num_burn, int thin, bool sparse) const {
	return returnSvRecords(sparse_record, num_iter, num_burn, thin, sparse);
}

/**
 * @brief Initialize MCMC record used in forecast classes
 * 
 * @param record Smart pointer of `LdltRecords` or `SvRecords`
 * @param chain_id Chain id
 * @param fit_record `LIST` of MCMC draw
 * @param include_mean Include constant term?
 * @param coef_name Element name for the coefficient in `fit_record`
 * @param a_name Element name for the contemporaneous coefficient in `fit_record`
 * @param c_name Element name for the constant term in `fit_record`
 */
inline void initialize_record(std::unique_ptr<LdltRecords>& record, int chain_id, LIST& fit_record, bool include_mean, STRING& coef_name, STRING& a_name, STRING& c_name) {
	PY_LIST coef_list = fit_record[coef_name];
	PY_LIST a_list = fit_record[a_name];
	PY_LIST d_list = fit_record["d_record"];
	if (include_mean) {
		PY_LIST c_list = fit_record[c_name];
		record = std::make_unique<LdltRecords>(
			CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			CAST<Eigen::MatrixXd>(c_list[chain_id]),
			CAST<Eigen::MatrixXd>(a_list[chain_id]),
			CAST<Eigen::MatrixXd>(d_list[chain_id])
		);
	} else {
		record = std::make_unique<LdltRecords>(
			CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			CAST<Eigen::MatrixXd>(a_list[chain_id]),
			CAST<Eigen::MatrixXd>(d_list[chain_id])
		);
	}
}

/**
 * @copydoc initialize_record(std::unique_ptr<LdltRecords>&, int, LIST&, bool, STRING&, STRING&, STRING&)
 * 
 */
inline void initialize_record(std::unique_ptr<SvRecords>& record, int chain_id, LIST& fit_record, bool include_mean, STRING& coef_name, STRING& a_name, STRING& c_name) {
	PY_LIST coef_list = fit_record[coef_name];
	PY_LIST a_list = fit_record[a_name];
	PY_LIST h_list = fit_record["h_record"];
	PY_LIST sigh_list = fit_record["sigh_record"];
	if (include_mean) {
		PY_LIST c_list = fit_record[c_name];
		record = std::make_unique<SvRecords>(
			CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			CAST<Eigen::MatrixXd>(c_list[chain_id]),
			CAST<Eigen::MatrixXd>(h_list[chain_id]),
			CAST<Eigen::MatrixXd>(a_list[chain_id]),
			CAST<Eigen::MatrixXd>(sigh_list[chain_id])
		);
	} else {
		record = std::make_unique<SvRecords>(
			CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			CAST<Eigen::MatrixXd>(h_list[chain_id]),
			CAST<Eigen::MatrixXd>(a_list[chain_id]),
			CAST<Eigen::MatrixXd>(sigh_list[chain_id])
		);
	}
}

} // namespace bvhar

#endif // BVHARCONFIG_H