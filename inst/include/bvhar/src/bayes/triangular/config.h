/**
 * @file config.h
 * @author your name (you@domain.com)
 * @brief Headers including MCMC configuration structs
 */

#ifndef BVHAR_BAYES_TRIANGULAR_CONFIG_H
#define BVHAR_BAYES_TRIANGULAR_CONFIG_H

#include "../misc/draw.h"
#include "../bayes.h"
#include "../../math/design.h"
#include <utility>

namespace bvhar {

// Parameters
struct RegParams;
struct SvParams;
// Initialization
struct RegInits;
struct LdltInits;
struct SvInits;
// MCMC records
struct RegRecords;
struct SparseRecords;
struct LdltRecords;
struct SvRecords;

/**
 * @brief Hyperparameters for `McmcReg`
 * 
 */
struct RegParams : McmcParams {
	Eigen::MatrixXd _x, _y;
	bool _mean;
	int _dim, _dim_design, _num_design, _num_lowerchol, _num_coef, _num_alpha, _nrow;
	int _nrow_exogen, _num_exogen;
	int _size_factor, _num_factor;
	Eigen::VectorXd _alpha_mean, _alpha_prec, _chol_mean, _chol_prec, _sig_shp, _sig_scl, _mean_non;
	double _sd_non;
	std::set<int> _own_id;
	std::set<int> _cross_id;
	Eigen::VectorXi _grp_id;
	// Eigen::MatrixXi _grp_mat;
	Eigen::VectorXi _grp_vec;

	RegParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		BVHAR_LIST& spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		BVHAR_LIST& intercept,
		bool include_mean,
		BVHAR_OPTIONAL<int> exogen_cols = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<int> factor_size = BVHAR_NULLOPT
	)
	: McmcParams(num_iter),
		_x(x), _y(y),
		_mean(include_mean),
		_dim(y.cols()), _dim_design(x.cols()), _num_design(y.rows()),
		_num_lowerchol(_dim * (_dim - 1) / 2), _num_coef(_dim * _dim_design),
		_num_alpha(_mean ? _num_coef - _dim : _num_coef), _nrow(_num_alpha / _dim),
		_nrow_exogen(exogen_cols ? *exogen_cols : 0), _num_exogen(_nrow_exogen * _dim),
		_size_factor(factor_size ? *factor_size : 0), _num_factor(_size_factor * _dim),
		_alpha_mean(Eigen::VectorXd::Zero(_num_coef)),
		_alpha_prec(Eigen::VectorXd::Ones(_num_coef)),
		_chol_mean(Eigen::VectorXd::Zero(_num_lowerchol)),
		_chol_prec(Eigen::VectorXd::Ones(_num_lowerchol)),
		_sig_shp(BVHAR_CAST<Eigen::VectorXd>(spec["shape"])),
		_sig_scl(BVHAR_CAST<Eigen::VectorXd>(spec["scale"])),
		_mean_non(BVHAR_CAST<Eigen::VectorXd>(intercept["mean_non"])),
		_sd_non(BVHAR_CAST_DOUBLE(intercept["sd_non"])),
		_grp_id(grp_id), _grp_vec(grp_mat.reshaped()) {
		set_grp_id(_own_id, _cross_id, own_id, cross_id);
		_num_alpha -= _num_exogen + _num_factor;
		_nrow = _num_alpha / _dim;
	}
};

/**
 * @brief Hyperparameters for `McmcSv`
 * 
 */
struct SvParams : public RegParams {
	Eigen::VectorXd _init_mean;
	Eigen::VectorXd _init_prec;

	SvParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		BVHAR_LIST& spec,
		const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		BVHAR_LIST& intercept,
		bool include_mean,
		BVHAR_OPTIONAL<int> exogen_cols = BVHAR_NULLOPT
	)
	: RegParams(num_iter, x, y, spec, own_id, cross_id, grp_id, grp_mat, intercept, include_mean, exogen_cols),
		_init_mean(BVHAR_CAST<Eigen::VectorXd>(spec["initial_mean"])),
		_init_prec(BVHAR_CAST<Eigen::VectorXd>(spec["initial_prec"])) {}
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

	RegInits(BVHAR_LIST& init)
	: _coef(BVHAR_CAST<Eigen::MatrixXd>(init["init_coef"])),
		_contem(BVHAR_CAST<Eigen::VectorXd>(init["init_contem"])) {}
};

/**
 * @brief MCMC initial values for `McmcReg`
 * 
 */
struct LdltInits : public RegInits {
	Eigen::VectorXd _diag;

	LdltInits(BVHAR_LIST& init)
	: RegInits(init),
		_diag(BVHAR_CAST<Eigen::VectorXd>(init["init_diag"])) {}
	
	LdltInits(BVHAR_LIST& init, int num_design)
	: RegInits(init),
		_diag(BVHAR_CAST<Eigen::VectorXd>(init["init_diag"])) {}
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
	
	SvInits(BVHAR_LIST& init)
	: RegInits(init),
		_lvol_init(BVHAR_CAST<Eigen::VectorXd>(init["lvol_init"])),
		_lvol(BVHAR_CAST<Eigen::MatrixXd>(init["lvol"])),
		_lvol_sig(BVHAR_CAST<Eigen::VectorXd>(init["lvol_sig"])) {}
	
	SvInits(BVHAR_LIST& init, int num_design)
	: RegInits(init),
		_lvol_init(BVHAR_CAST<Eigen::VectorXd>(init["lvol_init"])),
		_lvol(_lvol_init.transpose().replicate(num_design, 1)),
		_lvol_sig(BVHAR_CAST<Eigen::VectorXd>(init["lvol_sig"])) {}
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
	 * @brief Return the MCMC record `BVHAR_LIST`
	 * 
	 * @param dim Time series dimension
	 * @param num_alpha The number of coefficient elements except constant term
	 * @param include_mean If `true`, constant term is included
	 * @return BVHAR_LIST A `BVHAR_LIST` containing MCMC records. If `include_mean` is `true`, it also includes a constant term record.
	 */
	BVHAR_LIST returnListRecords(int dim, int num_alpha, int num_exogen, bool include_mean) const {
		BVHAR_LIST res = BVHAR_CREATE_LIST(
			BVHAR_NAMED("alpha_record") = coef_record.leftCols(num_alpha),
			BVHAR_NAMED("a_record") = contem_coef_record
		);
		if (include_mean) {
			res["c_record"] = BVHAR_CAST_MATRIX(coef_record.middleCols(num_alpha, dim));
		}
		if (num_exogen > 0) {
			res["b_record"] = BVHAR_CAST_MATRIX(coef_record.rightCols(num_exogen));
		}
		return res;
	}

	/**
	 * @brief Append records to the MCMC record `BVHAR_LIST`
	 * 
	 * @param list MCMC record `BVHAR_LIST`
	 */
	virtual void appendRecords(BVHAR_LIST& list) = 0;

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
	void assignRecords(int id, int num_alpha, int dim, int nrow_coef, int num_exogen, int nrow_exogen, const Eigen::MatrixXd& coef_mat, const Eigen::VectorXd& contem_coef) {
		coef_record.row(id).head(num_alpha) = coef_mat.topRows(nrow_coef).reshaped();
		if (coef_mat.rows() > nrow_coef) {
			coef_record.row(id).segment(num_alpha, dim) = coef_mat.middleRows<1>(nrow_coef).reshaped();
			if (nrow_exogen > 0) {
				coef_record.row(id).tail(num_exogen) = coef_mat.bottomRows(nrow_exogen).reshaped();
			}
		}
		// if (coef_mat.size() == num_alpha) {
		// 	coef_record.row(id) = coef_mat.reshaped();
		// } else {
		// 	coef_record.row(id).head(num_alpha) = coef_mat.topRows(nrow_coef).reshaped();
		// 	coef_record.row(id).segment(num_alpha, dim) = coef_mat.bottomRows(1).reshaped();
		// }
		contem_coef_record.row(id) = contem_coef;
	}

	/**
	 * @brief Append sparse records to the MCMC record `BVHAR_LIST`
	 * 
	 * @param list MCMC record `BVHAR_LIST`
	 * @param dim Time series dimension
	 * @param num_alpha The number of coefficient elements except constant term
	 * @param include_mean If `true`, constant term is included
	 */
	void appendRecords(BVHAR_LIST& list, int dim, int num_alpha, int num_exogen, bool include_mean) {
		list["alpha_sparse_record"] = BVHAR_CAST_MATRIX(coef_record.leftCols(num_alpha));
		list["a_sparse_record"] = contem_coef_record;
		if (include_mean) {
			list["c_sparse_record"] = BVHAR_CAST_MATRIX(coef_record.middleCols(num_alpha, dim));
		}
		if (num_exogen > 0) {
			list["b_sparse_record"] = BVHAR_CAST_MATRIX(coef_record.rightCols(num_exogen));
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

	LdltRecords(
		const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& c_record, const Eigen::MatrixXd& b_record,
		const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& d_record
	)
	: RegRecords(Eigen::MatrixXd::Zero(alpha_record.rows(), alpha_record.cols() + c_record.cols() + b_record.cols()), a_record),
		fac_record(d_record) {
		coef_record << alpha_record, c_record, b_record;
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

	void appendRecords(BVHAR_LIST& list) override {
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

	SvRecords(
		const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& c_record, const Eigen::MatrixXd& b_record,
		const Eigen::MatrixXd& h_record, const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& sigh_record
	)
	: RegRecords(Eigen::MatrixXd::Zero(alpha_record.rows(), alpha_record.cols() + c_record.cols() + b_record.cols()), a_record),
		lvol_sig_record(sigh_record), lvol_init_record(Eigen::MatrixXd::Zero(coef_record.rows(), lvol_sig_record.cols())),
		lvol_record(h_record) {
		coef_record << alpha_record, c_record, b_record;
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

	void appendRecords(BVHAR_LIST& list) override {
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
 * @param fit_record `BVHAR_LIST` of MCMC draw
 * @param include_mean Include constant term?
 * @param coef_name Element name for the coefficient in `fit_record`
 * @param a_name Element name for the contemporaneous coefficient in `fit_record`
 * @param c_name Element name for the constant term in `fit_record`
 */
inline void initialize_record(
	std::unique_ptr<LdltRecords>& record, int chain_id, BVHAR_LIST& fit_record, bool include_mean,
	BVHAR_STRING& coef_name, BVHAR_STRING& a_name, BVHAR_STRING& c_name, BVHAR_OPTIONAL<BVHAR_STRING> b_name = BVHAR_NULLOPT
) {
	BVHAR_PY_LIST coef_list = fit_record[coef_name];
	BVHAR_PY_LIST a_list = fit_record[a_name];
	BVHAR_PY_LIST d_list = fit_record["d_record"];
	if (include_mean && b_name) {
		BVHAR_PY_LIST c_list = fit_record[c_name];
		BVHAR_PY_LIST b_list = fit_record[*b_name];
		record = std::make_unique<LdltRecords>(
			BVHAR_CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(c_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(b_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(a_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(d_list[chain_id])
		);
	} else if (include_mean && !b_name) {
		BVHAR_PY_LIST c_list = fit_record[c_name];
		record = std::make_unique<LdltRecords>(
			BVHAR_CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(c_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(a_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(d_list[chain_id])
		);
	} else if (!include_mean && b_name) {
		BVHAR_PY_LIST b_list = fit_record[*b_name];
		record = std::make_unique<LdltRecords>(
			BVHAR_CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(b_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(a_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(d_list[chain_id])
		);
	} else {
		record = std::make_unique<LdltRecords>(
			BVHAR_CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(a_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(d_list[chain_id])
		);
	}
}

/**
 * @copydoc initialize_record(std::unique_ptr<LdltRecords>&, int, BVHAR_LIST&, bool, BVHAR_STRING&, BVHAR_STRING&, BVHAR_STRING&)
 * 
 */
inline void initialize_record(
	std::unique_ptr<SvRecords>& record, int chain_id, BVHAR_LIST& fit_record, bool include_mean,
	BVHAR_STRING& coef_name, BVHAR_STRING& a_name, BVHAR_STRING& c_name, BVHAR_OPTIONAL<BVHAR_STRING> b_name = BVHAR_NULLOPT
) {
	BVHAR_PY_LIST coef_list = fit_record[coef_name];
	BVHAR_PY_LIST a_list = fit_record[a_name];
	BVHAR_PY_LIST h_list = fit_record["h_record"];
	BVHAR_PY_LIST sigh_list = fit_record["sigh_record"];
	if (include_mean && b_name) {
		BVHAR_PY_LIST c_list = fit_record[c_name];
		BVHAR_PY_LIST b_list = fit_record[*b_name];
		record = std::make_unique<SvRecords>(
			BVHAR_CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(c_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(b_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(h_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(a_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(sigh_list[chain_id])
		);
	} else if (include_mean && !b_name) {
		BVHAR_PY_LIST c_list = fit_record[c_name];
		record = std::make_unique<SvRecords>(
			BVHAR_CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(c_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(h_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(a_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(sigh_list[chain_id])
		);
	} else if (!include_mean && b_name) {
		BVHAR_PY_LIST b_list = fit_record[*b_name];
		record = std::make_unique<SvRecords>(
			BVHAR_CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(b_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(h_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(a_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(sigh_list[chain_id])
		);
	} else {
		record = std::make_unique<SvRecords>(
			BVHAR_CAST<Eigen::MatrixXd>(coef_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(h_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(a_list[chain_id]),
			BVHAR_CAST<Eigen::MatrixXd>(sigh_list[chain_id])
		);
	}
}

} // namespace bvhar

#endif // BVHAR_BAYES_TRIANGULAR_CONFIG_H