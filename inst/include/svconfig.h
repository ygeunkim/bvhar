#ifndef SVCONFIG_H
#define SVCONFIG_H

#include "bvharconfig.h"
#include "bvhardesign.h"

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
struct GlSvInits;
struct HsSvInits;
struct NgSvInits;
// MCMC records
struct SvRecords;

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

struct MinnSvParams : public SvParams {
	Eigen::MatrixXd _prec_diag;
	Eigen::MatrixXd _prior_mean;
	Eigen::MatrixXd _prior_prec;

	MinnSvParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& sv_spec, LIST& priors, LIST& intercept,
		bool include_mean
	)
	: SvParams(num_iter, x, y, sv_spec, intercept, include_mean),
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
		// _prior_prec = (dummy_design.transpose() * dummy_design).inverse();
		_prior_prec = dummy_design.transpose() * dummy_design;
		_prior_mean = _prior_prec * dummy_design.transpose() * dummy_response;
		// _prec_diag.diagonal() = _sigma;
		_prec_diag.diagonal() = 1 / _sigma.array();
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
		LIST& sv_spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		LIST& priors, LIST& intercept,
		bool include_mean
	)
	: SvParams(num_iter, x, y, sv_spec, intercept, include_mean),
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
		// _prior_prec = (dummy_design.transpose() * dummy_design).inverse();
		_prior_prec = dummy_design.transpose() * dummy_design;
		_prior_mean = _prior_prec * dummy_design.transpose() * dummy_response;
		// _prec_diag.diagonal() = _sigma;
		_prec_diag.diagonal() = 1 / _sigma.array();
		_grp_mat = grp_mat;
		_minnesota = true;
		// if (own_id.size() == 1 && cross_id.size() == 1) {
		// 	_minnesota = false;
		// }
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

struct SsvsSvParams : public SvParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;
	// Eigen::VectorXd _coef_spike;
	// Eigen::VectorXd _coef_slab;
	// Eigen::VectorXd _coef_weight;
	// Eigen::VectorXd _contem_spike;
	// Eigen::VectorXd _contem_slab;
	// Eigen::VectorXd _contem_weight;
	Eigen::VectorXd _coef_s1, _coef_s2;
	double _contem_s1, _contem_s2;
	double _coef_spike_scl, _contem_spike_scl;
	double _coef_slab_shape, _coef_slab_scl, _contem_slab_shape, _contem_slab_scl;

	SsvsSvParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& sv_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& ssvs_spec, LIST& intercept,
		bool include_mean
	)
	: SvParams(num_iter, x, y, sv_spec, intercept, include_mean),
		_grp_id(grp_id), _grp_mat(grp_mat),
		// _coef_spike(Rcpp::as<Eigen::VectorXd>(ssvs_spec["coef_spike"])),
		// _coef_slab(Rcpp::as<Eigen::VectorXd>(ssvs_spec["coef_slab"])),
		// _coef_weight(Rcpp::as<Eigen::VectorXd>(ssvs_spec["coef_mixture"])),
		// _contem_spike(Rcpp::as<Eigen::VectorXd>(ssvs_spec["chol_spike"])),
		// _contem_slab(Rcpp::as<Eigen::VectorXd>(ssvs_spec["chol_slab"])),
		// _contem_weight(Rcpp::as<Eigen::VectorXd>(ssvs_spec["chol_mixture"])),
		_coef_s1(CAST<Eigen::VectorXd>(ssvs_spec["coef_s1"])), _coef_s2(CAST<Eigen::VectorXd>(ssvs_spec["coef_s2"])),
		_contem_s1(CAST_DOUBLE(ssvs_spec["chol_s1"])), _contem_s2(CAST_DOUBLE(ssvs_spec["chol_s2"])),
		_coef_spike_scl(CAST_DOUBLE(ssvs_spec["coef_spike_scl"])), _contem_spike_scl(CAST_DOUBLE(ssvs_spec["chol_spike_scl"])),
		_coef_slab_shape(CAST_DOUBLE(ssvs_spec["coef_slab_shape"])), _coef_slab_scl(CAST_DOUBLE(ssvs_spec["coef_slab_scl"])),
		_contem_slab_shape(CAST_DOUBLE(ssvs_spec["chol_slab_shape"])), _contem_slab_scl(CAST_DOUBLE(ssvs_spec["chol_slab_scl"])) {}
};

struct HsSvParams : public SvParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;

	HsSvParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& sv_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& intercept, bool include_mean
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
		LIST& sv_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& ng_spec, LIST& intercept,
		bool include_mean
	)
	: SvParams(num_iter, x, y, sv_spec, intercept, include_mean), _grp_id(grp_id), _grp_mat(grp_mat),
		// _local_shape(ng_spec["local_shape"]),
		// _contem_shape(ng_spec["contem_shape"]),
		_mh_sd(CAST_DOUBLE(ng_spec["shape_sd"])),
		_group_shape(CAST_DOUBLE(ng_spec["group_shape"])), _group_scl(CAST_DOUBLE(ng_spec["group_scale"])),
		_global_shape(CAST_DOUBLE(ng_spec["global_shape"])), _global_scl(CAST_DOUBLE(ng_spec["global_scale"])),
		_contem_global_shape(CAST_DOUBLE(ng_spec["contem_global_shape"])), _contem_global_scl(CAST_DOUBLE(ng_spec["contem_global_scale"])) {}
};

struct DlSvParams : public SvParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;
	// double _dl_concen;
	// double _contem_dl_concen;
	int _grid_size;
	double _shape;
	double _rate;

	DlSvParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& sv_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& dl_spec, LIST& intercept,
		bool include_mean
	)
	: SvParams(num_iter, x, y, sv_spec, intercept, include_mean),
		_grp_id(grp_id), _grp_mat(grp_mat),
		// _dl_concen(dl_spec["dirichlet"]), _contem_dl_concen(dl_spec["contem_dirichlet"]),
		_grid_size(CAST_INT(dl_spec["grid_size"])), _shape(CAST_DOUBLE(dl_spec["shape"])), _rate(CAST_DOUBLE(dl_spec["rate"])) {}
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
	
	// void updateState() override
};

struct HierminnSvInits : public SvInits {
	double _own_lambda;
	double _cross_lambda;
	double _contem_lambda;

	HierminnSvInits(LIST& init)
	: SvInits(init),
		_own_lambda(CAST_DOUBLE(init["own_lambda"])), _cross_lambda(CAST_DOUBLE(init["cross_lambda"])), _contem_lambda(CAST_DOUBLE(init["contem_lambda"])) {}
	
	HierminnSvInits(LIST& init, int num_design)
	: SvInits(init, num_design),
		_own_lambda(CAST_DOUBLE(init["own_lambda"])), _cross_lambda(CAST_DOUBLE(init["cross_lambda"])), _contem_lambda(CAST_DOUBLE(init["contem_lambda"])) {}
};

struct SsvsSvInits : public SvInits {
	Eigen::VectorXd _coef_dummy;
	Eigen::VectorXd _coef_weight; // in SsvsSvParams: move coef_mixture and chol_mixture in set_ssvs()?
	Eigen::VectorXd _contem_weight; // in SsvsSvParams
	Eigen::VectorXd _coef_slab;
	Eigen::VectorXd _contem_slab;
	
	SsvsSvInits(LIST& init)
	: SvInits(init),
		_coef_dummy(CAST<Eigen::VectorXd>(init["init_coef_dummy"])),
		_coef_weight(CAST<Eigen::VectorXd>(init["coef_mixture"])),
		_contem_weight(CAST<Eigen::VectorXd>(init["chol_mixture"])),
		_coef_slab(CAST<Eigen::VectorXd>(init["coef_slab"])),
		_contem_slab(CAST<Eigen::VectorXd>(init["contem_slab"])) {}
	
	SsvsSvInits(LIST& init, int num_design)
	: SvInits(init, num_design),
		_coef_dummy(CAST<Eigen::VectorXd>(init["init_coef_dummy"])),
		_coef_weight(CAST<Eigen::VectorXd>(init["coef_mixture"])),
		_contem_weight(CAST<Eigen::VectorXd>(init["chol_mixture"])),
		_coef_slab(CAST<Eigen::VectorXd>(init["coef_slab"])),
		_contem_slab(CAST<Eigen::VectorXd>(init["contem_slab"])) {}
};

struct GlSvInits : public SvInits {
	Eigen::VectorXd _init_local;
	double _init_global;
	Eigen::VectorXd _init_contem_local;
	Eigen::VectorXd _init_conetm_global;
	
	GlSvInits(LIST& init)
	: SvInits(init),
		_init_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_init_global(CAST_DOUBLE(init["global_sparsity"])),
		_init_contem_local(CAST<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_conetm_global(CAST<Eigen::VectorXd>(init["contem_global_sparsity"])) {}
	
	GlSvInits(LIST& init, int num_design)
	: SvInits(init, num_design),
		_init_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_init_global(CAST_DOUBLE(init["global_sparsity"])),
		_init_contem_local(CAST<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_conetm_global(CAST<Eigen::VectorXd>(init["contem_global_sparsity"])) {}
};

struct HsSvInits : public GlSvInits {
	Eigen::VectorXd _init_group;
	
	HsSvInits(LIST& init)
	: GlSvInits(init),
		_init_group(CAST<Eigen::VectorXd>(init["group_sparsity"])) {}
	
	HsSvInits(LIST& init, int num_design)
	: GlSvInits(init, num_design),
		_init_group(CAST<Eigen::VectorXd>(init["group_sparsity"])) {}
};

struct NgSvInits : public HsSvInits {
	Eigen::VectorXd _init_local_shape;
	double _init_contem_shape;

	NgSvInits(LIST& init)
	: HsSvInits(init),
		_init_local_shape(CAST<Eigen::VectorXd>(init["local_shape"])),
		_init_contem_shape(CAST_DOUBLE(init["contem_shape"])) {}
	
	NgSvInits(LIST& init, int num_design)
	: HsSvInits(init, num_design),
		_init_local_shape(CAST<Eigen::VectorXd>(init["local_shape"])),
		_init_contem_shape(CAST_DOUBLE(init["contem_shape"])) {}
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

	void subsetStable(int num_alpha) {
		int dim = lvol_sig_record.cols();
		int nrow_coef = num_alpha / dim;
		std::vector<int> stable_id;
		for (int i = 0; i < coef_record.rows(); ++i) {
			if (is_stable(coef_record.row(i).head(num_alpha).reshaped(nrow_coef, dim))) {
				stable_id.push_back(i);
			}
		}
		coef_record = std::move(coef_record(stable_id, Eigen::all));
		contem_coef_record = std::move(contem_coef_record(stable_id, Eigen::all));
		lvol_record = std::move(lvol_record(stable_id, Eigen::all));
		lvol_sig_record = std::move(lvol_sig_record(stable_id, Eigen::all));
		lvol_init_record = std::move(lvol_init_record(stable_id, Eigen::all));
	}

	void subsetStable(int num_alpha, Eigen::Ref<const Eigen::MatrixXd> har_trans) {
		int dim = lvol_sig_record.cols();
		int nrow_coef = num_alpha / dim;
		std::vector<int> stable_id;
		for (int i = 0; i < coef_record.rows(); ++i) {
			if (is_stable(coef_record.row(i).head(num_alpha).reshaped(nrow_coef, dim), har_trans)) {
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

} // namespace bvhar

#endif // SVCONFIG_H