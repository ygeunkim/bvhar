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
// Initialization
struct RegInits;
// MCMC records
struct RegRecords;
struct SparseRecords;
struct SsvsRecords;
struct GlobalLocalRecords;
struct HorseshoeRecords;
struct NgRecords;

struct RegParams {
	int _iter;
	Eigen::MatrixXd _x, _y;
	Eigen::VectorXd _sig_shp, _sig_scl, _mean_non;
	double _sd_non;
	bool _mean;
	int _dim, _dim_design, _num_design, _num_lowerchol, _num_coef, _num_alpha, _nrow;

	RegParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& spec, LIST& intercept,
		bool include_mean
	)
	: _iter(num_iter), _x(x), _y(y),
		_sig_shp(CAST<Eigen::VectorXd>(spec["shape"])),
		_sig_scl(CAST<Eigen::VectorXd>(spec["scale"])),
		_mean_non(CAST<Eigen::VectorXd>(intercept["mean_non"])),
		_sd_non(CAST_DOUBLE(intercept["sd_non"])), _mean(include_mean),
		_dim(y.cols()), _dim_design(x.cols()), _num_design(y.rows()),
		_num_lowerchol(_dim * (_dim - 1) / 2), _num_coef(_dim * _dim_design),
		_num_alpha(_mean ? _num_coef - _dim : _num_coef), _nrow(_num_alpha / _dim) {}
};

struct SvParams : public RegParams {
	Eigen::VectorXd _init_mean;
	Eigen::MatrixXd _init_prec;

	SvParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& spec, LIST& intercept,
		bool include_mean
	)
	: RegParams(num_iter, x, y, spec, intercept, include_mean),
		_init_mean(CAST<Eigen::VectorXd>(spec["initial_mean"])),
		_init_prec(CAST<Eigen::MatrixXd>(spec["initial_prec"])) {}
};

template <typename BaseRegParams = RegParams>
struct MinnParams : public BaseRegParams {
	Eigen::MatrixXd _prec_diag;
	Eigen::MatrixXd _prior_mean;
	Eigen::MatrixXd _prior_prec;
	MinnParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& reg_spec, LIST& priors, LIST& intercept,
		bool include_mean
	)
	: BaseRegParams(num_iter, x, y, reg_spec, intercept, include_mean),
		_prec_diag(Eigen::MatrixXd::Zero(y.cols(), y.cols())) {
		int lag = CAST_INT(priors["p"]); // append to bayes_spec, p = 3 in VHAR
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

template <typename BaseRegParams = RegParams>
struct HierminnParams : public BaseRegParams {
	double shape;
	double rate;
	Eigen::MatrixXd _prec_diag;
	Eigen::MatrixXd _prior_mean;
	Eigen::MatrixXd _prior_prec;
	Eigen::MatrixXi _grp_mat;
	bool _minnesota;
	std::set<int> _own_id;
	std::set<int> _cross_id;

	HierminnParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& reg_spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		LIST& priors, LIST& intercept,
		bool include_mean
	)
	: BaseRegParams(num_iter, x, y, reg_spec, intercept, include_mean),
		shape(CAST_DOUBLE(priors["shape"])), rate(CAST_DOUBLE(priors["rate"])),
		_prec_diag(Eigen::MatrixXd::Zero(y.cols(), y.cols())) {
		int lag = CAST_INT(priors["p"]); // append to bayes_spec, p = 3 in VHAR
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
		_grp_mat = grp_mat;
		_minnesota = true;
		std::set<int> unique_grp(_grp_mat.data(), _grp_mat.data() + _grp_mat.size());
		if (unique_grp.size() == 1) {
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

template <typename BaseRegParams = RegParams>
struct SsvsParams : public BaseRegParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;
	Eigen::VectorXd _coef_s1, _coef_s2;
	double _contem_s1, _contem_s2;
	double _coef_spike_scl, _contem_spike_scl;
	double _coef_slab_shape, _coef_slab_scl, _contem_slab_shape, _contem_slab_scl;

	SsvsParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& reg_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& ssvs_spec, LIST& intercept,
		bool include_mean
	)
	: BaseRegParams(num_iter, x, y, reg_spec, intercept, include_mean),
		_grp_id(grp_id), _grp_mat(grp_mat),
		_coef_s1(CAST<Eigen::VectorXd>(ssvs_spec["coef_s1"])), _coef_s2(CAST<Eigen::VectorXd>(ssvs_spec["coef_s2"])),
		_contem_s1(CAST_DOUBLE(ssvs_spec["chol_s1"])), _contem_s2(CAST_DOUBLE(ssvs_spec["chol_s2"])),
		_coef_spike_scl(CAST_DOUBLE(ssvs_spec["coef_spike_scl"])), _contem_spike_scl(CAST_DOUBLE(ssvs_spec["chol_spike_scl"])),
		_coef_slab_shape(CAST_DOUBLE(ssvs_spec["coef_slab_shape"])), _coef_slab_scl(CAST_DOUBLE(ssvs_spec["coef_slab_scl"])),
		_contem_slab_shape(CAST_DOUBLE(ssvs_spec["chol_slab_shape"])), _contem_slab_scl(CAST_DOUBLE(ssvs_spec["chol_slab_scl"])) {}
};

template <typename BaseRegParams = RegParams>
struct HorseshoeParams : public BaseRegParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;

	HorseshoeParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& reg_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& intercept, bool include_mean
	)
	: BaseRegParams(num_iter, x, y, reg_spec, intercept, include_mean), _grp_id(grp_id), _grp_mat(grp_mat) {}
};

template <typename BaseRegParams = RegParams>
struct NgParams : public BaseRegParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;
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
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& ng_spec, LIST& intercept,
		bool include_mean
	)
	: BaseRegParams(num_iter, x, y, reg_spec, intercept, include_mean), _grp_id(grp_id), _grp_mat(grp_mat),
		_mh_sd(CAST_DOUBLE(ng_spec["shape_sd"])),
		_group_shape(CAST_DOUBLE(ng_spec["group_shape"])), _group_scl(CAST_DOUBLE(ng_spec["group_scale"])),
		_global_shape(CAST_DOUBLE(ng_spec["global_shape"])), _global_scl(CAST_DOUBLE(ng_spec["global_scale"])),
		_contem_global_shape(CAST_DOUBLE(ng_spec["contem_global_shape"])), _contem_global_scl(CAST_DOUBLE(ng_spec["contem_global_scale"])) {}
};

template <typename BaseRegParams = RegParams>
struct DlParams : public BaseRegParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;
	int _grid_size;
	double _shape;
	double _rate;

	DlParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& reg_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& dl_spec, LIST& intercept,
		bool include_mean
	)
	: BaseRegParams(num_iter, x, y, reg_spec, intercept, include_mean),
		_grp_id(grp_id), _grp_mat(grp_mat),
		_grid_size(CAST_INT(dl_spec["grid_size"])), _shape(CAST_DOUBLE(dl_spec["shape"])), _rate(CAST_DOUBLE(dl_spec["rate"])) {}
};

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
	
	// void updateCoef()

	// void updateImpact()

	// virtual void updateState()
};

struct RegRecords {
	Eigen::MatrixXd coef_record; // alpha in VAR
	Eigen::MatrixXd contem_coef_record; // a = a21, a31, a32, ..., ak1, ..., ak(k-1)

	RegRecords() : coef_record(), contem_coef_record() {}

	RegRecords(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_coef)),
		contem_coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol)) {}
	
	RegRecords(const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& a_record)
	: coef_record(alpha_record), contem_coef_record(a_record) {}

	void assignRecords(int id, const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef) {
		coef_record.row(id) = coef_vec;
		contem_coef_record.row(id) = contem_coef;
	}
	
	virtual void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef, const Eigen::VectorXd& diag_vec
	) = 0;

	virtual void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef,
		const Eigen::MatrixXd& lvol_draw, const Eigen::VectorXd& lvol_sig, const Eigen::VectorXd& lvol_init
	) = 0;

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
	virtual void appendRecords(LIST& list) = 0;
	virtual void subsetStable(int num_alpha, double threshold) = 0;
	virtual void subsetStable(int num_alpha, double threshold, Eigen::Ref<const Eigen::MatrixXd> har_trans) = 0;
	virtual void subsetStable(int num_alpha, double threshold, Eigen::Ref<const Eigen::SparseMatrix<double>> har_trans) = 0;

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

struct SparseRecords {
	Eigen::MatrixXd coef_record;
	Eigen::MatrixXd contem_coef_record;

	SparseRecords(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_coef)),
		contem_coef_record(Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol)) {}
	
	SparseRecords(const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& a_record)
	: coef_record(alpha_record), contem_coef_record(a_record) {}
	
	void assignRecords(int id, const Eigen::MatrixXd& coef_mat, const Eigen::VectorXd& contem_coef) {
		coef_record.row(id) = coef_mat.reshaped();
		contem_coef_record.row(id) = contem_coef;
	}

	void appendRecords(LIST& list, int dim, int num_alpha, bool include_mean) {
		list["alpha_sparse_record"] = coef_record.leftCols(num_alpha);
		list["a_sparse_record"] = contem_coef_record;
		if (include_mean) {
			list["c_sparse_record"] = CAST_MATRIX(coef_record.rightCols(dim));
		}
	}
};

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
	
	void assignRecords(int id, const Eigen::VectorXd& coef_dummy, const Eigen::VectorXd& coef_weight, const Eigen::VectorXd& contem_dummy, const Eigen::VectorXd& contem_weight) {
		coef_dummy_record.row(id) = coef_dummy;
		coef_weight_record.row(id) = coef_weight;
		contem_dummy_record.row(id) = contem_dummy;
		contem_weight_record.row(id) = contem_weight;
	}

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

struct GlobalLocalRecords {
	Eigen::MatrixXd local_record;
	Eigen::VectorXd global_record;

	GlobalLocalRecords() : local_record(), global_record() {}
	
	GlobalLocalRecords(int num_iter, int num_alpha)
	: local_record(Eigen::MatrixXd::Zero(num_iter + 1, num_alpha)),
		global_record(Eigen::VectorXd::Zero(num_iter + 1)) {}
	
	GlobalLocalRecords(const Eigen::MatrixXd& local_record, const Eigen::VectorXd& global_record)
	: local_record(local_record), global_record(global_record) {}
	
	virtual void assignRecords(int id, const Eigen::VectorXd& local_lev, const double global_lev) {
		local_record.row(id) = local_lev;
		global_record[id] = global_lev;
	}

	virtual void assignRecords(int id, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) {
		assignRecords(id, local_lev, global_lev);
	}

	virtual void assignRecords(int id, const Eigen::VectorXd& shrink_fac, const Eigen::VectorXd& local_lev, const Eigen::VectorXd& group_lev, const double global_lev) {
		assignRecords(id, local_lev, global_lev);
	}

	GlobalLocalRecords returnGlRecords(int num_iter, int num_burn, int thin) const {
		GlobalLocalRecords res_record(
			thin_record(local_record, num_iter, num_burn, thin).derived(),
			thin_record(global_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
};

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

	NgRecords returnNgRecords(int num_iter, int num_burn, int thin) const {
		NgRecords res_record(
			thin_record(local_record, num_iter, num_burn, thin).derived(),
			thin_record(group_record, num_iter, num_burn, thin).derived(),
			thin_record(global_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
};

} // namespace bvhar

#endif // BVHARCONFIG_H