#ifndef BVHARMCMC_H
#define BVHARMCMC_H

#include "bvhardraw.h"
#include "bvhardesign.h"
#include "bvharprogress.h"
#include <type_traits>

namespace bvhar {

// Parameters
struct SvParams2;

template <typename BaseRegParams>
struct MinnParams2;

template <typename BaseRegParams>
struct HierMinnParams2;

template <typename BaseRegParams>
struct SsvsParams2;

template <typename BaseRegParams>
struct HorseshoeParams2;

template <typename BaseRegParams>
struct NgParams2;

template <typename BaseRegParams>
struct DlParams2;

// Initialization
struct LdltInits2;
struct SvInits2;

template <typename BaseRegInits>
struct HierminnInits2;

template <typename BaseRegInits>
struct SsvsInits2;

template <typename BaseRegInits>
struct GlInits2;

template <typename BaseRegInits>
struct HsInits2;

template <typename BaseRegInits>
struct NgInits2;

// MCMC records
struct LdltRecords2;
struct SvRecords2;

// MCMC algorithms
class McmcCta;

class McmcReg2;
class McmcSv2;

template <typename BaseCta, typename InitType>
class MinnReg2;

template <typename BaseCta>
class HierMinnReg2;

template <typename BaseCta>
class SsvsReg2;

template <typename BaseCta>
class HorseshoeReg2;

template <typename BaseCta>
class NgReg2;

template <typename BaseCta>
class DlReg2;

struct SvParams2 : public RegParams {
	Eigen::VectorXd _init_mean;
	Eigen::MatrixXd _init_prec;

	SvParams2(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& spec, LIST& intercept,
		bool include_mean
	)
	: RegParams(num_iter, x, y, spec, intercept, include_mean),
		_init_mean(CAST<Eigen::VectorXd>(spec["initial_mean"])),
		_init_prec(CAST<Eigen::MatrixXd>(spec["initial_prec"])) {}
};

template <typename BaseRegParams = RegParams>
struct MinnParams2 : public BaseRegParams {
	Eigen::MatrixXd _prec_diag;
	Eigen::MatrixXd _prior_mean;
	Eigen::MatrixXd _prior_prec;

	MinnParams2(
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
struct HierMinnParams2 : public MinnParams2<BaseRegParams> {
	double shape;
	double rate;
	// Eigen::MatrixXd _prec_diag;
	// Eigen::MatrixXd _prior_mean;
	// Eigen::MatrixXd _prior_prec;
	Eigen::MatrixXi _grp_mat;
	bool _minnesota;
	std::set<int> _own_id;
	std::set<int> _cross_id;

	HierMinnParams2(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& reg_spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
		LIST& priors, LIST& intercept,
		bool include_mean
	)
	: MinnParams2<BaseRegParams>(num_iter, x, y, reg_spec, priors, intercept, include_mean),
		shape(CAST_DOUBLE(priors["shape"])), rate(CAST_DOUBLE(priors["rate"])),
		_grp_mat(grp_mat), _minnesota(true) {
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
struct SsvsParams2 : public BaseRegParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;
	Eigen::VectorXd _coef_s1, _coef_s2;
	double _contem_s1, _contem_s2;
	double _coef_spike_scl, _contem_spike_scl;
	double _coef_slab_shape, _coef_slab_scl, _contem_slab_shape, _contem_slab_scl;

	SsvsParams2(
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
struct HorseshoeParams2 : public BaseRegParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;

	HorseshoeParams2(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& reg_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		LIST& intercept, bool include_mean
	)
	: BaseRegParams(num_iter, x, y, reg_spec, intercept, include_mean), _grp_id(grp_id), _grp_mat(grp_mat) {}
};

template <typename BaseRegParams = RegParams>
struct NgParams2 : public BaseRegParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;
	double _mh_sd;
	double _group_shape;
	double _group_scl;
	double _global_shape;
	double _global_scl;
	double _contem_global_shape;
	double _contem_global_scl;

	NgParams2(
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
struct DlParams2 : public BaseRegParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;
	int _grid_size;
	double _shape;
	double _rate;

	DlParams2(
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

struct LdltInits2 : public RegInits {
	Eigen::VectorXd _diag;

	LdltInits2(LIST& init)
	: RegInits(init),
		_diag(CAST<Eigen::VectorXd>(init["init_diag"])) {}
};

struct SvInits2 : public RegInits {
	Eigen::VectorXd _lvol_init;
	Eigen::MatrixXd _lvol;
	Eigen::VectorXd _lvol_sig;

	SvInits2(const SvParams2& params)
	: RegInits(params) {
		int dim = params._y.cols();
		int num_design = params._y.rows();
		_lvol_init = (params._y - params._x * _coef).transpose().array().square().rowwise().mean().log();
		_lvol = _lvol_init.transpose().replicate(num_design, 1);
		_lvol_sig = .1 * Eigen::VectorXd::Ones(dim);
	}

	SvInits2(LIST& init)
	: RegInits(init),
		_lvol_init(CAST<Eigen::VectorXd>(init["lvol_init"])),
		_lvol(CAST<Eigen::MatrixXd>(init["lvol"])),
		_lvol_sig(CAST<Eigen::VectorXd>(init["lvol_sig"])) {}
	
	SvInits2(LIST& init, int num_design)
	: RegInits(init),
		_lvol_init(CAST<Eigen::VectorXd>(init["lvol_init"])),
		_lvol(_lvol_init.transpose().replicate(num_design, 1)),
		_lvol_sig(CAST<Eigen::VectorXd>(init["lvol_sig"])) {}
};

template <typename BaseRegInits = LdltInits2>
struct HierminnInits2 : public BaseRegInits {
	double _own_lambda;
	double _cross_lambda;
	double _contem_lambda;

	HierminnInits2(LIST& init)
	: BaseRegInits(init),
		_own_lambda(CAST_DOUBLE(init["own_lambda"])), _cross_lambda(CAST_DOUBLE(init["cross_lambda"])), _contem_lambda(CAST_DOUBLE(init["contem_lambda"])) {}
	
	HierminnInits2(LIST& init, int num_design)
	: BaseRegInits(init, num_design),
		_own_lambda(CAST_DOUBLE(init["own_lambda"])), _cross_lambda(CAST_DOUBLE(init["cross_lambda"])), _contem_lambda(CAST_DOUBLE(init["contem_lambda"])) {}
};

template <typename BaseRegInits = LdltInits2>
struct SsvsInits2 : public BaseRegInits {
	Eigen::VectorXd _coef_dummy;
	Eigen::VectorXd _coef_weight;
	Eigen::VectorXd _contem_weight;
	Eigen::VectorXd _coef_slab;
	Eigen::VectorXd _contem_slab;

	SsvsInits2(LIST& init)
	: BaseRegInits(init),
		_coef_dummy(CAST<Eigen::VectorXd>(init["init_coef_dummy"])),
		_coef_weight(CAST<Eigen::VectorXd>(init["coef_mixture"])),
		_contem_weight(CAST<Eigen::VectorXd>(init["chol_mixture"])),
		_coef_slab(CAST<Eigen::VectorXd>(init["coef_slab"])),
		_contem_slab(CAST<Eigen::VectorXd>(init["contem_slab"])) {}
	
	SsvsInits2(LIST& init, int num_design)
	: BaseRegInits(init, num_design),
		_coef_dummy(CAST<Eigen::VectorXd>(init["init_coef_dummy"])),
		_coef_weight(CAST<Eigen::VectorXd>(init["coef_mixture"])),
		_contem_weight(CAST<Eigen::VectorXd>(init["chol_mixture"])),
		_coef_slab(CAST<Eigen::VectorXd>(init["coef_slab"])),
		_contem_slab(CAST<Eigen::VectorXd>(init["contem_slab"])) {}
};

template <typename BaseRegInits = LdltInits2>
struct GlInits2 : public BaseRegInits {
	Eigen::VectorXd _init_local;
	double _init_global;
	Eigen::VectorXd _init_contem_local;
	Eigen::VectorXd _init_conetm_global;
	
	GlInits2(LIST& init)
	: BaseRegInits(init),
		_init_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_init_global(CAST_DOUBLE(init["global_sparsity"])),
		_init_contem_local(CAST<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_conetm_global(CAST<Eigen::VectorXd>(init["contem_global_sparsity"])) {}
	
	GlInits2(LIST& init, int num_design)
	: BaseRegInits(init, num_design),
		_init_local(CAST<Eigen::VectorXd>(init["local_sparsity"])),
		_init_global(CAST_DOUBLE(init["global_sparsity"])),
		_init_contem_local(CAST<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_conetm_global(CAST<Eigen::VectorXd>(init["contem_global_sparsity"])) {}
};

template <typename BaseRegInits = LdltInits2>
struct HsInits2 : public GlInits2<BaseRegInits> {
	Eigen::VectorXd _init_group;
	
	HsInits2(LIST& init)
	: GlInits2<BaseRegInits>(init), _init_group(CAST<Eigen::VectorXd>(init["group_sparsity"])) {}

	HsInits2(LIST& init, int num_design)
	: GlInits2<BaseRegInits>(init, num_design), _init_group(CAST<Eigen::VectorXd>(init["group_sparsity"])) {}
};

template <typename BaseRegInits = LdltInits2>
struct NgInits2 : public HsInits2<BaseRegInits> {
	Eigen::VectorXd _init_local_shape;
	double _init_contem_shape;

	NgInits2(LIST& init)
	: HsInits2<BaseRegInits>(init),
		_init_local_shape(CAST<Eigen::VectorXd>(init["local_shape"])),
		_init_contem_shape(CAST_DOUBLE(init["contem_shape"])) {}
	
	NgInits2(LIST& init, int num_design)
	: HsInits2<BaseRegInits>(init, num_design),
		_init_local_shape(CAST<Eigen::VectorXd>(init["local_shape"])),
		_init_contem_shape(CAST_DOUBLE(init["contem_shape"])) {}
};

struct LdltRecords2 : public RegRecords {
	Eigen::MatrixXd fac_record; // d_1, ..., d_m in D of LDLT

	LdltRecords2() : RegRecords(), fac_record() {}
	
	LdltRecords2(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: RegRecords(num_iter, dim, num_design, num_coef, num_lowerchol),
		fac_record(Eigen::MatrixXd::Zero(num_iter + 1, dim)) {}
	
	LdltRecords2(const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& d_record)
	: RegRecords(alpha_record, a_record), fac_record(d_record) {}

	LdltRecords2(
		const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& c_record,
		const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& d_record
	)
	: RegRecords(Eigen::MatrixXd::Zero(alpha_record.rows(), alpha_record.cols() + c_record.cols()), a_record),
		fac_record(d_record) {
		coef_record << alpha_record, c_record;
	}

	void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef, const Eigen::VectorXd& diag_vec
	) {
		coef_record.row(id) = coef_vec;
		contem_coef_record.row(id) = contem_coef;
		fac_record.row(id) = diag_vec.array();
	}
};

struct SvRecords2 : public RegRecords {
	Eigen::MatrixXd lvol_sig_record; // sigma_h^2 = (sigma_(h1i)^2, ..., sigma_(hki)^2)
	Eigen::MatrixXd lvol_init_record; // h0 = h10, ..., hk0
	Eigen::MatrixXd lvol_record; // time-varying h = (h_1, ..., h_k) with h_j = (h_j1, ..., h_jn), row-binded

	SvRecords2() : RegRecords(), lvol_sig_record(), lvol_init_record(), lvol_record() {}
	
	SvRecords2(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: RegRecords(num_iter, dim, num_design, num_coef, num_lowerchol),
		lvol_sig_record(Eigen::MatrixXd::Ones(num_iter + 1, dim)),
		lvol_init_record(Eigen::MatrixXd::Zero(num_iter + 1, dim)),
		lvol_record(Eigen::MatrixXd::Zero(num_iter + 1, num_design * dim)) {}
	
	SvRecords2(
		const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& h_record,
		const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& sigh_record
	)
	: RegRecords(alpha_record, a_record),
		lvol_sig_record(sigh_record), lvol_init_record(Eigen::MatrixXd::Zero(coef_record.rows(), lvol_sig_record.cols())),
		lvol_record(h_record) {}
	
	SvRecords2(
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

class McmcCta {
public:
	McmcCta(RegParams& params, RegInits& inits, unsigned int seed)
	: include_mean(params._mean), x(params._x), y(params._y),
		num_iter(params._iter), dim(y.cols()), dim_design(x.cols()), num_design(y.rows()),
		num_lowerchol(dim * (dim - 1) / 2), num_coef(dim * dim_design), num_alpha(include_mean ? num_coef - dim : num_coef),
		sparse_record(num_iter, dim, num_design, num_alpha, num_lowerchol),
		mcmc_step(0), rng(seed),
		coef_vec(Eigen::VectorXd::Zero(num_coef)), contem_coef(inits._contem),
		prior_alpha_mean(Eigen::VectorXd::Zero(num_coef)), prior_alpha_prec(Eigen::VectorXd::Zero(num_coef)),
		prior_chol_mean(Eigen::VectorXd::Zero(num_lowerchol)), prior_chol_prec(Eigen::VectorXd::Ones(num_lowerchol)),
		coef_mat(inits._coef),
		contem_id(0), chol_lower(build_inv_lower(dim, contem_coef)),
		latent_innov(y - x * coef_mat), ortho_latent(Eigen::MatrixXd::Zero(num_design, dim)),
		response_contem(Eigen::VectorXd::Zero(num_design)), sqrt_sv(Eigen::MatrixXd::Zero(num_design, dim)),
		sparse_coef(Eigen::MatrixXd::Zero(num_alpha / dim, dim)), sparse_contem(Eigen::VectorXd::Zero(num_lowerchol)),
		prior_sig_shp(params._sig_shp), prior_sig_scl(params._sig_scl) {
		if (include_mean) {
			prior_alpha_mean.tail(dim) = params._mean_non;
			prior_alpha_prec.tail(dim) = 1 / (params._sd_non * Eigen::VectorXd::Ones(dim)).array().square();
		}
		coef_vec.head(num_alpha) = coef_mat.topRows(num_alpha / dim).reshaped();
		if (include_mean) {
			coef_vec.tail(dim) = coef_mat.bottomRows(1).transpose();
		}
		sparse_record.assignRecords(0, sparse_coef, sparse_contem);
	}
	virtual ~McmcCta() = default;
	virtual void doPosteriorDraws() = 0;
	virtual LIST gatherRecords() const = 0;
	virtual void appendRecords(LIST& list) = 0;
	// virtual LIST returnRecords(int num_burn, int thin) const = 0;
	LIST returnRecords(int num_burn, int thin) {
		LIST res = gatherRecords();
		appendRecords(res);
		for (auto& record : res) {
			if (IS_MATRIX(ACCESS_LIST(record, res))) {
				ACCESS_LIST(record, res) = thin_record(CAST<Eigen::MatrixXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			} else {
				ACCESS_LIST(record, res) = thin_record(CAST<Eigen::VectorXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			}
		}
		return res;
	}
	virtual LdltRecords2 returnLdltRecords(int num_burn, int thin, bool sparse = false) const = 0;
	virtual SvRecords2 returnSvRecords(int num_burn, int thin, bool sparse = false) const = 0;
	virtual SsvsRecords returnSsvsRecords(int num_burn, int thin) const = 0;
	virtual HorseshoeRecords returnHsRecords(int num_burn, int thin) const = 0;
	virtual NgRecords returnNgRecords(int num_burn, int thin) const = 0;
	virtual GlobalLocalRecords returnGlRecords(int num_burn, int thin) const = 0;

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
	SparseRecords sparse_record;
	std::atomic<int> mcmc_step; // MCMC step
	boost::random::mt19937 rng; // RNG instance for multi-chain
	Eigen::VectorXd coef_vec;
	Eigen::VectorXd contem_coef;
	Eigen::VectorXd prior_alpha_mean; // prior mean vector of alpha
	Eigen::VectorXd prior_alpha_prec; // Diagonal of alpha prior precision
	Eigen::VectorXd prior_chol_mean; // prior mean vector of a = 0
	Eigen::VectorXd prior_chol_prec; // Diagonal of prior precision of a = I
	Eigen::MatrixXd coef_mat;
	int contem_id;
	Eigen::MatrixXd chol_lower; // L in Sig_t^(-1) = L D_t^(-1) LT
	Eigen::MatrixXd latent_innov; // Z0 = Y0 - X0 A = (eps_p+1, eps_p+2, ..., eps_n+p)^T
	Eigen::MatrixXd ortho_latent; // orthogonalized Z0
	Eigen::VectorXd response_contem; // j-th column of Z0 = Y0 - X0 * A: n-dim
	Eigen::MatrixXd sqrt_sv; // stack sqrt of exp(h_t) = (exp(-h_1t / 2), ..., exp(-h_kt / 2)), t = 1, ..., n => n x k
	Eigen::MatrixXd sparse_coef;
	Eigen::VectorXd sparse_contem;
	Eigen::VectorXd prior_sig_shp;
	Eigen::VectorXd prior_sig_scl;
	void updateCoef() {
		for (int j = 0; j < dim; j++) {
			coef_mat.col(j).setZero(); // j-th column of A = 0: A(-j) = (alpha_1, ..., alpha_(j-1), 0, alpha_(j), ..., alpha_k)
			Eigen::MatrixXd chol_lower_j = chol_lower.bottomRows(dim - j); // L_(j:k) = a_jt to a_kt for t = 1, ..., j - 1
			Eigen::MatrixXd sqrt_sv_j = sqrt_sv.rightCols(dim - j); // use h_jt to h_kt for t = 1, .. n => (k - j + 1) x k
			Eigen::MatrixXd design_coef = kronecker_eigen(chol_lower_j.col(j), x).array().colwise() / sqrt_sv_j.reshaped().array(); // L_(j:k, j) otimes X0 scaled by D_(1:n, j:k): n(k - j + 1) x kp
			draw_coef(
				coef_mat.col(j),
				design_coef,
				(((y - x * coef_mat) * chol_lower_j.transpose()).array() / sqrt_sv_j.array()).reshaped(),
				prior_alpha_mean.segment(dim_design * j, dim_design), // Prior mean vector of j-th column of A
				prior_alpha_prec.segment(dim_design * j, dim_design), // Prior precision of j-th column of A
				rng
			);
			draw_savs(sparse_coef.col(j), coef_mat.col(j).head(num_alpha / dim), design_coef);
		}
		coef_vec.head(num_alpha) = coef_mat.topRows(num_alpha / dim).reshaped();
		if (include_mean) {
			coef_vec.tail(dim) = coef_mat.bottomRows(1).transpose();
		}
	}
	virtual void updateState() = 0; // Draw SV or diagonals
	void updateImpact() {
		for (int j = 2; j < dim + 1; j++) {
			response_contem = latent_innov.col(j - 2).array() / sqrt_sv.col(j - 2).array(); // n-dim
			Eigen::MatrixXd design_contem = latent_innov.leftCols(j - 1).array().colwise() / sqrt_sv.col(j - 2).reshaped().array(); // n x (j - 1)
			contem_id = (j - 1) * (j - 2) / 2;
			draw_coef(
				contem_coef.segment(contem_id, j - 1),
				design_contem, response_contem,
				prior_chol_mean.segment(contem_id, j - 1),
				prior_chol_prec.segment(contem_id, j - 1),
				rng
			);
			draw_savs(sparse_contem.segment(contem_id, j - 1), contem_coef.segment(contem_id, j - 1), design_contem);
		}
	}
	void addStep() { mcmc_step++; }
	void updateLatent() {
		latent_innov = y - x * coef_mat; // E_t before a
	}
	void updateChol() {
		chol_lower = build_inv_lower(dim, contem_coef); // L before h_t
	}
	virtual void updateSv() = 0;
	virtual void updateCoefPrec() = 0;
	virtual void updateCoefShrink() = 0;
	virtual void updateImpactPrec() = 0;
	virtual void updateRecords() = 0;
	virtual void updateCoefRecords() = 0;
};

class McmcReg2 : public McmcCta {
public:
	McmcReg2(RegParams& params, LdltInits2& inits, unsigned int seed)
	: McmcCta(params, inits, seed),
		reg_record(num_iter, dim, num_design, num_coef, num_lowerchol), diag_vec(inits._diag) {
		reg_record.assignRecords(0, coef_vec, contem_coef, diag_vec);
	}
	virtual ~McmcReg2() = default;
	void doPosteriorDraws() override {}
	LIST gatherRecords() const override {
		LIST res = CREATE_LIST(
			NAMED("alpha_record") = reg_record.coef_record.leftCols(num_alpha),
			NAMED("a_record") = reg_record.contem_coef_record,
			NAMED("d_record") = reg_record.fac_record,
			NAMED("alpha_sparse_record") = sparse_record.coef_record,
			NAMED("a_sparse_record") = sparse_record.contem_coef_record
		);
		if (include_mean) {
			res["c_record"] = CAST_MATRIX(reg_record.coef_record.rightCols(dim));
		}
		return res;
	}
	LdltRecords2 returnLdltRecords(int num_burn, int thin, bool sparse = false) const override {
		if (sparse) {
			Eigen::MatrixXd coef_record(num_iter + 1, num_coef);
			if (include_mean) {
				coef_record << sparse_record.coef_record, reg_record.coef_record.rightCols(dim);
			} else {
				coef_record = sparse_record.coef_record;
			}
			return LdltRecords2(
				thin_record(coef_record, num_iter, num_burn, thin).derived(),
				thin_record(sparse_record.contem_coef_record, num_iter, num_burn, thin).derived(),
				thin_record(reg_record.fac_record, num_iter, num_burn, thin).derived()
			);
		}
		LdltRecords2 res_record(
			thin_record(reg_record.coef_record, num_iter, num_burn, thin).derived(),
			thin_record(reg_record.contem_coef_record, num_iter, num_burn, thin).derived(),
			thin_record(reg_record.fac_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
	SvRecords2 returnSvRecords(int num_burn, int thin, bool sparse = false) const override {
		return SvRecords2();
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
	GlobalLocalRecords returnGlRecords(int num_burn, int thin) const override {
		return GlobalLocalRecords();
	}

protected:
	LdltRecords2 reg_record;
	void updateState() override {
		ortho_latent = latent_innov * chol_lower.transpose(); // L eps_t <=> Z0 U
		reg_ldlt_diag(diag_vec, prior_sig_shp, prior_sig_scl, ortho_latent, rng);
	}
	void updateSv() override {
		sqrt_sv = diag_vec.cwiseSqrt().transpose().replicate(num_design, 1);
	}
	void updateCoefRecords() override {
		reg_record.assignRecords(mcmc_step, coef_vec, contem_coef, diag_vec);
		sparse_record.assignRecords(mcmc_step, sparse_coef, sparse_contem);
	}

private:
	Eigen::VectorXd diag_vec; // inverse of d_i
};

class McmcSv2 : public McmcCta {
public:
	McmcSv2(SvParams2& params, SvInits2& inits, unsigned int seed)
	: McmcCta(params, inits, seed),
		reg_record(num_iter, dim, num_design, num_coef, num_lowerchol),
		lvol_draw(inits._lvol), lvol_init(inits._lvol_init), lvol_sig(inits._lvol_sig),
		prior_init_mean(params._init_mean), prior_init_prec(params._init_prec) {
		reg_record.assignRecords(0, coef_vec, contem_coef, lvol_draw, lvol_sig, lvol_init);
	}
	virtual ~McmcSv2() = default;
	void doPosteriorDraws() override {}
	LIST gatherRecords() const override {
		LIST res = CREATE_LIST(
			NAMED("alpha_record") = reg_record.coef_record.leftCols(num_alpha),
			NAMED("h_record") = reg_record.lvol_record,
			NAMED("a_record") = reg_record.contem_coef_record,
			NAMED("h0_record") = reg_record.lvol_init_record,
			NAMED("sigh_record") = reg_record.lvol_sig_record,
			NAMED("alpha_sparse_record") = sparse_record.coef_record,
			NAMED("a_sparse_record") = sparse_record.contem_coef_record
		);
		if (include_mean) {
			res["c_record"] = CAST_MATRIX(reg_record.coef_record.rightCols(dim));
		}
		return res;
	}
	LdltRecords2 returnLdltRecords(int num_burn, int thin, bool sparse = false) const override {
		return LdltRecords2();
	}
	SvRecords2 returnSvRecords(int num_burn, int thin, bool sparse = false) const override {
		if (sparse) {
			Eigen::MatrixXd coef_record(num_iter + 1, num_coef);
			if (include_mean) {
				coef_record << sparse_record.coef_record, reg_record.coef_record.rightCols(dim);
			} else {
				coef_record = sparse_record.coef_record;
			}
			return SvRecords2(
				thin_record(coef_record, num_iter, num_burn, thin).derived(),
				thin_record(reg_record.lvol_record, num_iter, num_burn, thin).derived(),
				thin_record(sparse_record.contem_coef_record, num_iter, num_burn, thin).derived(),
				thin_record(reg_record.lvol_sig_record, num_iter, num_burn, thin).derived()
			);
		}
		SvRecords2 res_record(
			thin_record(reg_record.coef_record, num_iter, num_burn, thin).derived(),
			thin_record(reg_record.lvol_record, num_iter, num_burn, thin).derived(),
			thin_record(reg_record.contem_coef_record, num_iter, num_burn, thin).derived(),
			thin_record(reg_record.lvol_sig_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
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
	GlobalLocalRecords returnGlRecords(int num_burn, int thin) const override {
		return GlobalLocalRecords();
	}

protected:
	SvRecords2 reg_record;
	void updateState() override {
		ortho_latent = latent_innov * chol_lower.transpose(); // L eps_t <=> Z0 U
		ortho_latent = (ortho_latent.array().square() + .0001).array().log(); // adjustment log(e^2 + c) for some c = 10^(-4) against numerical problems
		for (int t = 0; t < dim; t++) {
			varsv_ht(lvol_draw.col(t), lvol_init[t], lvol_sig[t], ortho_latent.col(t), rng);
		}
		updateStateVar();
		updateInitState();
	}
	void updateStateVar() { varsv_sigh(lvol_sig, prior_sig_shp, prior_sig_scl, lvol_init, lvol_draw, rng); }
	void updateInitState() { varsv_h0(lvol_init, prior_init_mean, prior_init_prec, lvol_draw.row(0), lvol_sig, rng); }
	void updateSv() override {
		sqrt_sv = (-lvol_draw / 2).array().exp();
	}
	void updateCoefRecords() override {
		reg_record.assignRecords(mcmc_step, coef_vec, contem_coef, lvol_draw, lvol_sig, lvol_init);
		sparse_record.assignRecords(mcmc_step, sparse_coef, sparse_contem);
	}

private:
	Eigen::MatrixXd lvol_draw; // h_j = (h_j1, ..., h_jn)
	Eigen::VectorXd lvol_init;
	Eigen::VectorXd lvol_sig;
	Eigen::VectorXd prior_init_mean;
	Eigen::MatrixXd prior_init_prec;
};

template <
	typename BaseCta = McmcReg2,
	typename InitType = typename std::conditional<
		std::is_same<BaseCta, McmcReg2>::value,
		LdltInits2,
		SvInits2
	>::type
>
class MinnReg2 : public BaseCta {
public:
	template <typename BaseRegParams>
	MinnReg2(MinnParams2<BaseRegParams>& params, InitType& inits, unsigned int seed) : BaseCta(params, inits, seed) {
	// MinnReg2(MinnParams2<>& params, InitType& inits, unsigned int seed) : BaseCta(params, inits, seed) {
		prior_alpha_mean.head(num_alpha) = params._prior_mean.reshaped();
		prior_alpha_prec.head(num_alpha) = kronecker_eigen(params._prec_diag, params._prior_prec).diagonal();
		if (include_mean) {
			prior_alpha_mean.tail(dim) = params._mean_non;
		}
	}
	virtual ~MinnReg2() = default;
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateSv(); // D_t before coef
		updateCoef();
		updateLatent(); // E_t before a
		updateImpact();
		updateChol(); // L before h_t
		updateState();
		updateRecords();
	}
	void appendRecords(LIST& list) override {}
	// SsvsRecords returnSsvsRecords(int num_burn, int thin) const override {
	// 	return SsvsRecords();
	// }
	// HorseshoeRecords returnHsRecords(int num_burn, int thin) const override {
	// 	return HorseshoeRecords();
	// }
	// NgRecords returnNgRecords(int num_burn, int thin) const override {
	// 	return NgRecords();
	// }
	// GlobalLocalRecords returnGlRecords(int num_burn, int thin) const override {
	// 	return GlobalLocalRecords();
	// }
	// LdltRecords2 returnLdltRecords(int num_burn, int thin, bool sparse = false) const override {
	// 	return BaseCta::returnLdltRecords(num_burn, thin);
	// }
	// SvRecords2 returnSvRecords(int num_burn, int thin, bool sparse = false) const override {
	// 	return BaseCta::returnSvRecords(num_burn, thin);
	// }
	// using BaseCta::returnRecords;

protected:
	using BaseCta::include_mean;
	using BaseCta::mtx;
	using BaseCta::num_iter;
	using BaseCta::dim;
	using BaseCta::num_alpha;
	using BaseCta::mcmc_step;
	using BaseCta::prior_alpha_mean;
	using BaseCta::prior_alpha_prec;
	using BaseCta::addStep;
	using BaseCta::updateCoef;
	using BaseCta::updateImpact;
	using BaseCta::updateState;
	using BaseCta::updateSv;
	using BaseCta::updateLatent;
	using BaseCta::updateChol;
	using BaseCta::updateCoefRecords;
	void updateCoefPrec() override {};
	void updateCoefShrink() override {};
	void updateImpactPrec() override {};
	void updateRecords() override { updateCoefRecords(); }
};

template <typename BaseCta = McmcReg2>
class HierMinnReg2 : public BaseCta {
public:
	template <typename BaseRegParams, typename BaseRegInits>
	HierMinnReg2(HierMinnParams2<BaseRegParams>& params, HierminnInits2<BaseRegInits>& inits, unsigned int seed)
	// HierMinnReg2(HierMinnParams2<>& params, HierminnInits2<>& inits, unsigned int seed)
	: BaseCta(params, inits, seed),
		own_id(params._own_id), cross_id(params._cross_id), coef_minnesota(params._minnesota), grp_mat(params._grp_mat), grp_vec(grp_mat.reshaped()),
		own_lambda(inits._own_lambda), cross_lambda(inits._cross_lambda), contem_lambda(inits._contem_lambda),
		own_shape(params.shape), own_rate(params.rate),
		cross_shape(params.shape), cross_rate(params.rate),
		contem_shape(params.shape), contem_rate(params.rate) {
		prior_alpha_mean.head(num_alpha) = params._prior_mean.reshaped();
		prior_alpha_prec.head(num_alpha) = kronecker_eigen(params._prec_diag, params._prior_prec).diagonal();
		for (int i = 0; i < num_alpha; ++i) {
			if (own_id.find(grp_vec[i]) != own_id.end()) {
				prior_alpha_prec[i] /= own_lambda; // divide because it is precision
			}
			if (cross_id.find(grp_vec[i]) != cross_id.end()) {
				prior_alpha_prec[i] /= cross_lambda; // divide because it is precision
			}
		}
		if (include_mean) {
			prior_alpha_mean.tail(dim) = params._mean_non;
		}
		prior_chol_prec.array() /= contem_lambda; // divide because it is precision
	}
	virtual ~HierMinnReg2() = default;
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		// updateCoefShrink();
		updateCoefPrec();
		updateSv();
		updateCoef();
		updateImpactPrec();
		updateLatent(); // E_t before a
		updateImpact();
		updateChol(); // L before h_t
		updateState();
		updateRecords();
	}
	void appendRecords(LIST& list) override {}
	// SsvsRecords returnSsvsRecords(int num_burn, int thin) const override {
	// 	return SsvsRecords();
	// }
	// HorseshoeRecords returnHsRecords(int num_burn, int thin) const override {
	// 	return HorseshoeRecords();
	// }
	// NgRecords returnNgRecords(int num_burn, int thin) const override {
	// 	return NgRecords();
	// }
	// GlobalLocalRecords returnGlRecords(int num_burn, int thin) const override {
	// 	return GlobalLocalRecords();
	// }
	// LdltRecords2 returnLdltRecords(int num_burn, int thin, bool sparse = false) const override {
	// 	return BaseCta::returnLdltRecords(num_burn, thin);
	// }
	// SvRecords2 returnSvRecords(int num_burn, int thin, bool sparse = false) const override {
	// 	return BaseCta::returnSvRecords(num_burn, thin);
	// }
	// using BaseCta::returnRecords;

protected:
	using BaseCta::include_mean;
	using BaseCta::mtx;
	using BaseCta::num_iter;
	using BaseCta::dim;
	using BaseCta::num_alpha;
	using BaseCta::mcmc_step;
	using BaseCta::coef_vec;
	using BaseCta::contem_coef;
	using BaseCta::prior_alpha_mean;
	using BaseCta::prior_alpha_prec;
	using BaseCta::prior_chol_mean;
	using BaseCta::prior_chol_prec;
	using BaseCta::addStep;
	using BaseCta::updateCoef;
	using BaseCta::updateImpact;
	using BaseCta::updateState;
	using BaseCta::updateSv;
	using BaseCta::updateLatent;
	using BaseCta::updateChol;
	using BaseCta::updateCoefRecords;
	using BaseCta::rng;
	void updateCoefPrec() override {
		updateCoefShrink();
		for (int i = 0; i < num_alpha; ++i) {
			if (own_id.find(grp_vec[i]) != own_id.end()) {
				prior_alpha_prec[i] /= own_lambda; // divide because it is precision
			}
			if (cross_id.find(grp_vec[i]) != cross_id.end()) {
				prior_alpha_prec[i] /= cross_lambda; // divide because it is precision
			}
		}
	}
	void updateCoefShrink() override {
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
	};
	void updateImpactPrec() override {
		minnesota_contem_lambda(
			contem_lambda, contem_shape, contem_rate,
			contem_coef, prior_chol_mean, prior_chol_prec,
			rng
		);
	};
	void updateRecords() override { updateCoefRecords(); }

private:
	std::set<int> own_id;
	std::set<int> cross_id;
	bool coef_minnesota;
	Eigen::MatrixXi grp_mat;
	Eigen::VectorXi grp_vec;
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

template <typename BaseCta = McmcReg2>
class SsvsReg2 : public BaseCta {
public:
	template <typename BaseRegParams, typename BaseRegInits>
	SsvsReg2(SsvsParams2<BaseRegParams>& params, SsvsInits2<BaseRegInits>& inits, unsigned int seed)
	// SsvsReg2(SsvsParams2<>& params, SsvsInits2<>& inits, unsigned int seed)
	: BaseCta(params, inits, seed),
		grp_id(params._grp_id), grp_vec(params._grp_mat.reshaped()), num_grp(grp_id.size()),
		ssvs_record(num_iter, num_alpha, num_grp, num_lowerchol),
		coef_dummy(inits._coef_dummy), coef_weight(inits._coef_weight),
		contem_dummy(Eigen::VectorXd::Ones(num_lowerchol)), contem_weight(inits._contem_weight),
		coef_slab(inits._coef_slab),
		spike_scl(params._coef_spike_scl), contem_spike_scl(params._coef_spike_scl),
		ig_shape(params._coef_slab_shape), ig_scl(params._coef_slab_scl),
		contem_ig_shape(params._contem_slab_shape), contem_ig_scl(params._contem_slab_scl),
		contem_slab(inits._contem_slab),
		coef_s1(params._coef_s1), coef_s2(params._coef_s2),
		contem_s1(params._contem_s1), contem_s2(params._contem_s2),
		slab_weight(Eigen::VectorXd::Ones(num_alpha)) {
		ssvs_record.assignRecords(0, coef_dummy, coef_weight, contem_dummy, contem_weight);
	}
	virtual ~SsvsReg2() = default;
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefPrec();
		updateSv();
		updateCoef();
		updateCoefShrink();
		updateImpactPrec();
		updateLatent(); // E_t before a
		updateImpact();
		updateChol(); // L before d_i
		updateState();
		updateRecords();
	}
	void appendRecords(LIST& list) override {
		list["gamma_record"] = ssvs_record.coef_dummy_record;
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
	// HorseshoeRecords returnHsRecords(int num_burn, int thin) const override {
	// 	return HorseshoeRecords();
	// }
	// NgRecords returnNgRecords(int num_burn, int thin) const override {
	// 	return NgRecords();
	// }
	// GlobalLocalRecords returnGlRecords(int num_burn, int thin) const override {
	// 	return GlobalLocalRecords();
	// }
	// LdltRecords2 returnLdltRecords(int num_burn, int thin, bool sparse = false) const override {
	// 	return BaseCta::returnLdltRecords(num_burn, thin);
	// }
	// SvRecords2 returnSvRecords(int num_burn, int thin, bool sparse = false) const override {
	// 	return BaseCta::returnSvRecords(num_burn, thin);
	// }
	// using BaseCta::returnRecords;

protected:
	using BaseCta::include_mean;
	using BaseCta::mtx;
	using BaseCta::num_iter;
	using BaseCta::dim;
	using BaseCta::num_alpha;
	using BaseCta::num_lowerchol;
	using BaseCta::mcmc_step;
	using BaseCta::coef_vec;
	using BaseCta::contem_coef;
	using BaseCta::prior_alpha_mean;
	using BaseCta::prior_alpha_prec;
	using BaseCta::prior_chol_mean;
	using BaseCta::prior_chol_prec;
	using BaseCta::addStep;
	using BaseCta::updateCoef;
	using BaseCta::updateImpact;
	using BaseCta::updateState;
	using BaseCta::updateSv;
	using BaseCta::updateLatent;
	using BaseCta::updateChol;
	using BaseCta::updateCoefRecords;
	using BaseCta::rng;
	void updateCoefPrec() override {
		ssvs_local_slab(coef_slab, coef_dummy, coef_vec.head(num_alpha), ig_shape, ig_scl, spike_scl, rng);
		prior_alpha_prec.head(num_alpha).array() = 1 / (spike_scl * (1 - coef_dummy.array()) * coef_slab.array() + coef_dummy.array() * coef_slab.array());
	}
	void updateCoefShrink() override {
		for (int j = 0; j < num_grp; j++) {
			slab_weight = (grp_vec.array() == grp_id[j]).select(
				coef_weight[j],
				slab_weight
			);
		}
		ssvs_dummy(
			coef_dummy,
			coef_vec.head(num_alpha),
			coef_slab, spike_scl * coef_slab, slab_weight,
			rng
		);
		ssvs_mn_weight(coef_weight, grp_vec, grp_id, coef_dummy, coef_s1, coef_s2, rng);
	}
	void updateImpactPrec() override {
		ssvs_local_slab(contem_slab, contem_dummy, contem_coef, contem_ig_shape, contem_ig_scl, contem_spike_scl, rng);
		ssvs_dummy(contem_dummy, contem_coef, contem_slab, contem_spike_scl * contem_slab, contem_weight, rng);
		ssvs_weight(contem_weight, contem_dummy, contem_s1, contem_s2, rng);
		prior_chol_prec = 1 / build_ssvs_sd(contem_spike_scl * contem_slab, contem_slab, contem_dummy).array().square();
	}
	void updateRecords() override {
		updateCoefRecords();
		ssvs_record.assignRecords(mcmc_step, coef_dummy, coef_weight, contem_dummy, contem_weight);
	}

private:
	Eigen::VectorXi grp_id;
	Eigen::VectorXi grp_vec;
	int num_grp;
	SsvsRecords ssvs_record;
	Eigen::VectorXd coef_dummy;
	Eigen::VectorXd coef_weight;
	Eigen::VectorXd contem_dummy;
	Eigen::VectorXd contem_weight;
	Eigen::VectorXd coef_slab;
	double spike_scl, contem_spike_scl; // scaling factor between 0 and 1: spike_sd = c * slab_sd
	double ig_shape, ig_scl, contem_ig_shape, contem_ig_scl; // IG hyperparameter for spike sd
	Eigen::VectorXd contem_slab;
	Eigen::VectorXd coef_s1, coef_s2;
	double contem_s1, contem_s2;
	Eigen::VectorXd slab_weight; // pij vector
};

template <typename BaseCta = McmcReg2>
class HorseshoeReg2 : public BaseCta {
public:
	template <typename BaseRegParams, typename BaseRegInits>
	HorseshoeReg2(HorseshoeParams2<BaseRegParams>& params, HsInits2<BaseRegInits>& inits, unsigned int seed)
	// HorseshoeReg2(HorseshoeParams2<>& params, HsInits2<>& inits, unsigned int seed)
	: BaseCta(params, inits, seed),
		grp_id(params._grp_id), grp_vec(params._grp_mat.reshaped()), num_grp(grp_id.size()),
		hs_record(num_iter, num_alpha, num_grp),
		local_lev(inits._init_local), group_lev(inits._init_group), global_lev(inits._init_global),
		shrink_fac(Eigen::VectorXd::Zero(num_alpha)),
		latent_local(Eigen::VectorXd::Zero(num_alpha)), latent_group(Eigen::VectorXd::Zero(num_grp)), latent_global(0.0),
		coef_var(Eigen::VectorXd::Zero(num_alpha)),
		contem_local_lev(inits._init_contem_local), contem_global_lev(inits._init_conetm_global),
		contem_var(Eigen::VectorXd::Zero(num_lowerchol)),
		latent_contem_local(Eigen::VectorXd::Zero(num_lowerchol)), latent_contem_global(Eigen::VectorXd::Zero(1)) {
		hs_record.assignRecords(0, shrink_fac, local_lev, group_lev, global_lev);
	}
	virtual ~HorseshoeReg2() = default;
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefPrec();
		updateSv();
		updateCoef();
		updateCoefShrink();
		updateImpactPrec();
		updateLatent(); // E_t before a
		updateImpact();
		updateChol(); // L before d_i
		updateState();
		updateRecords();
	}
	void appendRecords(LIST& list) override {
		list["lambda_record"] = hs_record.local_record;
		list["eta_record"] = hs_record.group_record;
		list["tau_record"] = hs_record.global_record;
		list["kappa_record"] = hs_record.shrink_record;
	}
	// SsvsRecords returnSsvsRecords(int num_burn, int thin) const override {
	// 	return SsvsRecords();
	// }
	HorseshoeRecords returnHsRecords(int num_burn, int thin) const override {
		HorseshoeRecords res_record(
			thin_record(hs_record.local_record, num_iter, num_burn, thin).derived(),
			thin_record(hs_record.group_record, num_iter, num_burn, thin).derived(),
			thin_record(hs_record.global_record, num_iter, num_burn, thin).derived(),
			thin_record(hs_record.shrink_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
	// NgRecords returnNgRecords(int num_burn, int thin) const override {
	// 	return NgRecords();
	// }
	// GlobalLocalRecords returnGlRecords(int num_burn, int thin) const override {
	// 	return GlobalLocalRecords();
	// }
	// LdltRecords2 returnLdltRecords(int num_burn, int thin, bool sparse = false) const override {
	// 	return BaseCta::returnLdltRecords(num_burn, thin);
	// }
	// SvRecords2 returnSvRecords(int num_burn, int thin, bool sparse = false) const override {
	// 	return BaseCta::returnSvRecords(num_burn, thin);
	// }
	// using BaseCta::returnRecords;

protected:
	using BaseCta::include_mean;
	using BaseCta::mtx;
	using BaseCta::num_iter;
	using BaseCta::dim;
	using BaseCta::num_alpha;
	using BaseCta::num_lowerchol;
	using BaseCta::mcmc_step;
	using BaseCta::coef_vec;
	using BaseCta::contem_coef;
	using BaseCta::prior_alpha_mean;
	using BaseCta::prior_alpha_prec;
	using BaseCta::prior_chol_mean;
	using BaseCta::prior_chol_prec;
	using BaseCta::addStep;
	using BaseCta::updateCoef;
	using BaseCta::updateImpact;
	using BaseCta::updateState;
	using BaseCta::updateSv;
	using BaseCta::updateLatent;
	using BaseCta::updateChol;
	using BaseCta::updateCoefRecords;
	using BaseCta::rng;
	void updateCoefPrec() override {
		for (int j = 0; j < num_grp; j++) {
			coef_var = (grp_vec.array() == grp_id[j]).select(
				group_lev[j],
				coef_var
			);
		}
		prior_alpha_prec.head(num_alpha) = 1 / (global_lev * coef_var.array() * local_lev.array()).square();
		shrink_fac = 1 / (1 + prior_alpha_prec.head(num_alpha).array());
	}
	void updateCoefShrink() override {
		horseshoe_latent(latent_local, local_lev, rng);
		horseshoe_latent(latent_group, group_lev, rng);
		horseshoe_latent(latent_global, global_lev, rng);
		global_lev = horseshoe_global_sparsity(latent_global, coef_var.array() * local_lev.array(), coef_vec.head(num_alpha), 1, rng);
		horseshoe_mn_sparsity(group_lev, grp_vec, grp_id, latent_group, global_lev, local_lev, coef_vec.head(num_alpha), 1, rng);
		horseshoe_local_sparsity(local_lev, latent_local, coef_var, coef_vec.head(num_alpha), global_lev * global_lev, rng);
	}
	void updateImpactPrec() override {
		horseshoe_latent(latent_contem_local, contem_local_lev, rng);
		horseshoe_latent(latent_contem_global, contem_global_lev, rng);
		contem_var = contem_global_lev.replicate(1, num_lowerchol).reshaped();
		horseshoe_local_sparsity(contem_local_lev, latent_contem_local, contem_var, contem_coef, 1, rng);
		contem_global_lev[0] = horseshoe_global_sparsity(latent_contem_global[0], latent_contem_local, contem_coef, 1, rng);
		prior_chol_prec.setZero();
		prior_chol_prec = 1 / (contem_var.array() * contem_local_lev.array()).square();
	}
	void updateRecords() override {
		updateCoefRecords();
		hs_record.assignRecords(mcmc_step, shrink_fac, local_lev, group_lev, global_lev);
	}

private:
	Eigen::VectorXi grp_id;
	Eigen::VectorXi grp_vec;
	int num_grp;
	HorseshoeRecords hs_record;
	Eigen::VectorXd local_lev;
	Eigen::VectorXd group_lev;
	double global_lev;
	Eigen::VectorXd shrink_fac;
	Eigen::VectorXd latent_local;
	Eigen::VectorXd latent_group;
	double latent_global;
	Eigen::VectorXd coef_var;
	Eigen::VectorXd contem_local_lev;
	Eigen::VectorXd contem_global_lev; // -> double
	Eigen::VectorXd contem_var;
	Eigen::VectorXd latent_contem_local;
	Eigen::VectorXd latent_contem_global; // -> double
};

template <typename BaseCta = McmcReg2>
class NgReg2 : public BaseCta {
public:
	template <typename BaseRegParams, typename BaseRegInits>
	NgReg2(NgParams2<BaseRegParams>& params, NgInits2<BaseRegInits>& inits, unsigned int seed)
	// NgReg2(NgParams2<>& params, NgInits2<>& inits, unsigned int seed)
	: BaseCta(params, inits, seed),
		grp_id(params._grp_id), grp_vec(params._grp_mat.reshaped()), num_grp(grp_id.size()),
		ng_record(num_iter, num_alpha, num_grp),
		mh_sd(params._mh_sd),
		local_shape(inits._init_local_shape), local_shape_fac(Eigen::VectorXd::Ones(num_alpha)),
		contem_shape(inits._init_contem_shape),
		group_shape(params._group_shape), group_scl(params._global_scl),
		global_shape(params._global_shape), global_scl(params._global_scl),
		contem_global_shape(params._contem_global_shape), contem_global_scl(params._contem_global_scl),
		local_lev(inits._init_local), group_lev(inits._init_group), global_lev(inits._init_global),
		coef_var(Eigen::VectorXd::Zero(num_alpha)),
		contem_global_lev(inits._init_conetm_global),
		contem_fac(contem_global_lev[0] * inits._init_contem_local) {
		ng_record.assignRecords(0, local_lev, group_lev, global_lev);
	}
	virtual ~NgReg2() = default;
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefPrec();
		updateSv();
		updateCoef();
		updateImpactPrec();
		updateLatent(); // E_t before a
		updateImpact();
		updateChol(); // L before d_i
		updateState();
		updateRecords();
	}
	void appendRecords(LIST& list) override {
		list["lambda_record"] = ng_record.local_record;
		list["eta_record"] = ng_record.group_record;
		list["tau_record"] = ng_record.global_record;
	}
	// SsvsRecords returnSsvsRecords(int num_burn, int thin) const override {
	// 	return SsvsRecords();
	// }
	// HorseshoeRecords returnHsRecords(int num_burn, int thin) const override {
	// 	return HorseshoeRecords();
	// }
	NgRecords returnNgRecords(int num_burn, int thin) const override {
		NgRecords res_record(
			thin_record(ng_record.local_record, num_iter, num_burn, thin).derived(),
			thin_record(ng_record.group_record, num_iter, num_burn, thin).derived(),
			thin_record(ng_record.global_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
	// GlobalLocalRecords returnGlRecords(int num_burn, int thin) const override {
	// 	return GlobalLocalRecords();
	// }
	// LdltRecords2 returnLdltRecords(int num_burn, int thin, bool sparse = false) const override {
	// 	return BaseCta::returnLdltRecords(num_burn, thin);
	// }
	// SvRecords2 returnSvRecords(int num_burn, int thin, bool sparse = false) const override {
	// 	return BaseCta::returnSvRecords(num_burn, thin);
	// }
	// using BaseCta::returnRecords;

protected:
	using BaseCta::include_mean;
	using BaseCta::mtx;
	using BaseCta::num_iter;
	using BaseCta::dim;
	using BaseCta::num_alpha;
	using BaseCta::num_lowerchol;
	using BaseCta::mcmc_step;
	using BaseCta::coef_vec;
	using BaseCta::contem_coef;
	using BaseCta::prior_alpha_mean;
	using BaseCta::prior_alpha_prec;
	using BaseCta::prior_chol_mean;
	using BaseCta::prior_chol_prec;
	using BaseCta::addStep;
	using BaseCta::updateCoef;
	using BaseCta::updateImpact;
	using BaseCta::updateState;
	using BaseCta::updateSv;
	using BaseCta::updateLatent;
	using BaseCta::updateChol;
	using BaseCta::updateCoefRecords;
	using BaseCta::rng;
	void updateCoefPrec() override {
		ng_mn_shape_jump(local_shape, local_lev, group_lev, grp_vec, grp_id, global_lev, mh_sd, rng);
		for (int j = 0; j < num_grp; j++) {
			coef_var = (grp_vec.array() == grp_id[j]).select(
				group_lev[j],
				coef_var
			);
			local_shape_fac = (grp_vec.array() == grp_id[j]).select(
				local_shape[j],
				local_shape_fac
			);
		}
		updateCoefShrink();
		prior_alpha_prec.head(num_alpha) = 1 / local_lev.array().square();
	}
	void updateCoefShrink() override {
		ng_local_sparsity(local_lev, local_shape_fac, coef_vec.head(num_alpha), global_lev * coef_var, rng);
		global_lev = ng_global_sparsity(local_lev.array() / coef_var.array(), local_shape_fac, global_shape, global_scl, rng);
		ng_mn_sparsity(group_lev, grp_vec, grp_id, local_shape, global_lev, local_lev, group_shape, group_scl, rng);
	}
	void updateImpactPrec() override {
		contem_shape = ng_shape_jump(contem_shape, contem_fac, contem_global_lev[0], mh_sd, rng);
		ng_local_sparsity(contem_fac, contem_shape, contem_coef, contem_global_lev.replicate(1, num_lowerchol).reshaped(), rng);
		contem_global_lev[0] = ng_global_sparsity(contem_fac, contem_shape, contem_global_shape, contem_global_scl, rng);
		prior_chol_prec = 1 / contem_fac.array().square();
	}
	void updateRecords() override {
		updateCoefRecords();
		ng_record.assignRecords(mcmc_step, local_lev, group_lev, global_lev);
	}

private:
	Eigen::VectorXi grp_id;
	Eigen::VectorXi grp_vec;
	int num_grp;
	NgRecords ng_record;
	double mh_sd;
	Eigen::VectorXd local_shape, local_shape_fac;
	double contem_shape;
	double group_shape, group_scl, global_shape, global_scl, contem_global_shape, contem_global_scl;
	Eigen::VectorXd local_lev;
	Eigen::VectorXd group_lev;
	double global_lev;
	Eigen::VectorXd coef_var;
	Eigen::VectorXd contem_global_lev;
	Eigen::VectorXd contem_fac;
};

template <typename BaseCta>
class DlReg2 : public BaseCta {
public:
	template <typename BaseRegParams, typename BaseRegInits>
	DlReg2(DlParams2<BaseRegParams>& params, GlInits2<BaseRegInits>& inits, unsigned int seed)
	// DlReg2(DlParams2<>& params, GlInits2<>& inits, unsigned int seed)
	: BaseCta(params, inits, seed),
		grp_id(params._grp_id), grp_vec(params._grp_mat.reshaped()), num_grp(grp_id.size()),
		dl_record(num_iter, num_alpha),
		dir_concen(0.0), contem_dir_concen(0.0),
		shape(params._shape), rate(params._rate),
		grid_size(params._grid_size),
		local_lev(inits._init_local), group_lev(Eigen::VectorXd::Zero(num_grp)), global_lev(inits._init_global),
		latent_local(Eigen::VectorXd::Zero(num_alpha)),
		coef_var(Eigen::VectorXd::Zero(num_alpha)),
		contem_local_lev(inits._init_contem_local), contem_global_lev(inits._init_conetm_global),
		latent_contem_local(Eigen::VectorXd::Zero(num_lowerchol)) {
		dl_record.assignRecords(0, local_lev, global_lev);
	}
	virtual ~DlReg2() = default;
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefPrec();
		updateSv();
		updateCoef();
		// updateCoefShrink();
		updateImpactPrec();
		updateLatent(); // E_t before a
		updateImpact();
		updateChol(); // L before d_i
		updateState();
		updateRecords();
	}
	void appendRecords(LIST& list) override {
		list["lambda_record"] = dl_record.local_record;
		list["tau_record"] = dl_record.global_record;
	}
	// SsvsRecords returnSsvsRecords(int num_burn, int thin) const override {
	// 	return SsvsRecords();
	// }
	// HorseshoeRecords returnHsRecords(int num_burn, int thin) const override {
	// 	return HorseshoeRecords();
	// }
	// NgRecords returnNgRecords(int num_burn, int thin) const override {
	// 	return NgRecords();
	// }
	GlobalLocalRecords returnGlRecords(int num_burn, int thin) const override {
		GlobalLocalRecords res_record(
			thin_record(dl_record.local_record, num_iter, num_burn, thin).derived(),
			thin_record(dl_record.global_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
	// LdltRecords2 returnLdltRecords(int num_burn, int thin, bool sparse = false) const override {
	// 	return BaseCta::returnLdltRecords(num_burn, thin);
	// }
	// SvRecords2 returnSvRecords(int num_burn, int thin, bool sparse = false) const override {
	// 	return BaseCta::returnSvRecords(num_burn, thin);
	// }
	// using BaseCta::returnRecords;

protected:
	using BaseCta::include_mean;
	using BaseCta::mtx;
	using BaseCta::num_iter;
	using BaseCta::dim;
	using BaseCta::num_alpha;
	using BaseCta::num_lowerchol;
	using BaseCta::mcmc_step;
	using BaseCta::coef_vec;
	using BaseCta::contem_coef;
	using BaseCta::prior_alpha_mean;
	using BaseCta::prior_alpha_prec;
	using BaseCta::prior_chol_mean;
	using BaseCta::prior_chol_prec;
	using BaseCta::addStep;
	using BaseCta::updateCoef;
	using BaseCta::updateImpact;
	using BaseCta::updateState;
	using BaseCta::updateSv;
	using BaseCta::updateLatent;
	using BaseCta::updateChol;
	using BaseCta::updateCoefRecords;
	using BaseCta::rng;
	void updateCoefPrec() override {
		dl_mn_sparsity(group_lev, grp_vec, grp_id, global_lev, local_lev, shape, rate, coef_vec.head(num_alpha), rng);
		for (int j = 0; j < num_grp; j++) {
			coef_var = (grp_vec.array() == grp_id[j]).select(
				group_lev[j],
				coef_var
			);
		}
		dl_latent(latent_local, global_lev * local_lev.array() * coef_var.array(), coef_vec.head(num_alpha), rng);
		updateCoefShrink();
		prior_alpha_prec.head(num_alpha) = 1 / ((global_lev * local_lev.array() * coef_var.array()).square() * latent_local.array());
	}
	void updateCoefShrink() override {
		dl_dir_griddy(dir_concen, grid_size, local_lev, global_lev, rng);
		dl_local_sparsity(local_lev, dir_concen, coef_vec.head(num_alpha).array() / coef_var.array(), rng);
		global_lev = dl_global_sparsity(local_lev.array() * coef_var.array(), dir_concen, coef_vec.head(num_alpha), rng);
	}
	void updateImpactPrec() override {
		dl_dir_griddy(contem_dir_concen, grid_size, contem_local_lev, contem_global_lev[0], rng);
		dl_latent(latent_contem_local, contem_local_lev, contem_coef, rng);
		dl_local_sparsity(contem_local_lev, contem_dir_concen, contem_coef, rng);
		contem_global_lev[0] = dl_global_sparsity(contem_local_lev, contem_dir_concen, contem_coef, rng);
		prior_chol_prec = 1 / ((contem_global_lev[0] * contem_local_lev.array()).square() * latent_contem_local.array());
	}
	void updateRecords() override {
		updateCoefRecords();
		dl_record.assignRecords(mcmc_step, local_lev, global_lev);
	}

private:
	Eigen::VectorXi grp_id;
	Eigen::VectorXi grp_vec;
	int num_grp;
	GlobalLocalRecords dl_record;
	double dir_concen, contem_dir_concen, shape, rate;
	int grid_size;
	Eigen::VectorXd local_lev;
	Eigen::VectorXd group_lev;
	double global_lev;
	Eigen::VectorXd latent_local;
	Eigen::VectorXd coef_var;
	Eigen::VectorXd contem_local_lev;
	Eigen::VectorXd contem_global_lev;
	Eigen::VectorXd latent_contem_local;
};

}; // namespace bvhar

#endif // BVHARMCMC_H