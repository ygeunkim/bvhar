/**
 * @file updater.h
 * @brief Pluggable draw strategies for the coefficient, impact, and variance blocks of `McmcTriangular`
 */

#ifndef BVHAR_BAYES_TRIANGULAR_UPDATER_H
#define BVHAR_BAYES_TRIANGULAR_UPDATER_H

#include "./config.h"
#include "../shrinkage/shrinkage.h"
#include "../dfm/augment.h"

namespace baecon {
namespace bvhar {

class CoefUpdater;
class StaticCoefUpdater;
class ImpactUpdater;
class StaticImpactUpdater;
class VarianceUpdater;
class LdltVarianceUpdater;
class SvVarianceUpdater;

/**
 * @brief Draw strategy for the autoregressive coefficient matrix
 *
 * Implementations own only their own extra state (e.g. a time-varying implementation would hold the
 * per-time coefficient history and its state-innovation prior); everything shared with the impact and
 * variance blocks travels through `TriangularState`.
 */
class CoefUpdater {
public:
	CoefUpdater() {}
	virtual ~CoefUpdater() = default;

	/**
	 * @brief Draw the coefficient matrix and its SAVS-sparsified counterpart
	 *
	 * @param state Shared mutable MCMC state
	 * @param favar_updater FAVAR factor augmenter, if enabled
	 * @param exogen_updater Exogenous-block shrinkage prior, if enabled
	 * @param factor_updater Factor-block shrinkage prior, if enabled
	 * @param rng RNG
	 */
	virtual void updateCoef(
		TriangularState& state,
		std::unique_ptr<FactorAugmenter>& favar_updater,
		std::unique_ptr<ShrinkageUpdater>& exogen_updater,
		std::unique_ptr<ShrinkageUpdater>& factor_updater,
		BVHAR_BHRNG& rng
	) = 0;
};

/**
 * @brief Time-invariant coefficient draw (column-wise Gaussian regression, current `McmcTriangular::updateCoef()`)
 */
class StaticCoefUpdater : public CoefUpdater {
public:
	StaticCoefUpdater() {}
	virtual ~StaticCoefUpdater() = default;

	void updateCoef(
		TriangularState& state,
		std::unique_ptr<FactorAugmenter>& favar_updater,
		std::unique_ptr<ShrinkageUpdater>& exogen_updater,
		std::unique_ptr<ShrinkageUpdater>& factor_updater,
		BVHAR_BHRNG& rng
	) override {
		if (favar_updater) {
			favar_updater->updateResid(state.x, state.y, state.coef_mat);
			favar_updater->updateFactor(state.coef_mat, state.chol_lower, state.sqrt_sv, rng);
			favar_updater->appendDesign(state.x);
		}
		for (int j = 0; j < state.dim; ++j) {
			state.coef_mat.col(j).setZero(); // j-th column of A = 0
			Eigen::MatrixXd chol_lower_j = state.chol_lower.bottomRows(state.dim - j); // L_(j:k) = a_jt to a_kt for t = 1, ..., j - 1
			Eigen::MatrixXd sqrt_sv_j = state.sqrt_sv.rightCols(state.dim - j); // use h_jt to h_kt for t = 1, .. n => (k - j + 1) x k
			Eigen::MatrixXd design_coef = kronecker_eigen(chol_lower_j.col(j), state.x).array().colwise() / sqrt_sv_j.reshaped().array(); // L_(j:k, j) otimes X0 scaled by D_(1:n, j:k): n(k - j + 1) x kp
			Eigen::VectorXd prior_mean_j(state.dim_design);
			Eigen::VectorXd prior_prec_j(state.dim_design);
			Eigen::VectorXd penalty_j = Eigen::VectorXd::Zero(state.dim_design);
			prior_mean_j.head(state.nrow_coef) = state.prior_alpha_mean.segment(j * state.nrow_coef, state.nrow_coef);
			prior_prec_j.head(state.nrow_coef) = state.prior_alpha_prec.segment(j * state.nrow_coef, state.nrow_coef);
			penalty_j.head(state.nrow_coef) = state.alpha_penalty.segment(j * state.nrow_coef, state.nrow_coef);
			if (state.include_mean) {
				prior_mean_j[state.nrow_coef] = state.prior_alpha_mean.segment(state.num_alpha, state.dim)[j];
				prior_prec_j[state.nrow_coef] = state.prior_alpha_prec.segment(state.num_alpha, state.dim)[j];
				if (exogen_updater) {
					prior_mean_j.segment(state.nrow_endog, state.nrow_exogen) = state.prior_alpha_mean.segment(state.num_endog + j * state.nrow_exogen, state.nrow_exogen);
					prior_prec_j.segment(state.nrow_endog, state.nrow_exogen) = state.prior_alpha_prec.segment(state.num_endog + j * state.nrow_exogen, state.nrow_exogen);
				}
				if (factor_updater) {
					prior_mean_j.tail(state.size_factor) = state.prior_alpha_mean.segment(state.num_endog + state.num_exogen + j * state.size_factor, state.size_factor);
					prior_prec_j.tail(state.size_factor) = state.prior_alpha_prec.segment(state.num_endog + state.num_exogen + j * state.size_factor, state.size_factor);
				}
				draw_coef(
					state.coef_mat.col(j), design_coef,
					(((state.y - state.x * state.coef_mat) * chol_lower_j.transpose()).array() / sqrt_sv_j.array()).reshaped(), // Hadamard product between: (Y - X0 A(-j))L_(j:k)^T and D_(1:n, j:k)
					prior_mean_j, prior_prec_j, rng
				);
				state.coef_vec.head(state.num_alpha) = state.coef_mat.topRows(state.nrow_coef).reshaped();
				state.coef_vec.segment(state.num_alpha, state.dim) = state.coef_mat.middleRows<1>(state.nrow_coef).transpose();
			} else {
				if (exogen_updater) {
					prior_mean_j.segment(state.nrow_endog, state.nrow_exogen) = state.prior_alpha_mean.segment(state.num_endog + j * state.nrow_exogen, state.nrow_exogen);
					prior_prec_j.segment(state.nrow_endog, state.nrow_exogen) = state.prior_alpha_prec.segment(state.num_endog + j * state.nrow_exogen, state.nrow_exogen);
				}
				if (factor_updater) {
					prior_mean_j.tail(state.size_factor) = state.prior_alpha_mean.segment(state.num_endog + state.num_exogen + j * state.size_factor, state.size_factor);
					prior_prec_j.tail(state.size_factor) = state.prior_alpha_prec.segment(state.num_endog + state.num_exogen + j * state.size_factor, state.size_factor);
				}
				draw_coef(
					state.coef_mat.col(j),
					design_coef,
					(((state.y - state.x * state.coef_mat) * chol_lower_j.transpose()).array() / sqrt_sv_j.array()).reshaped(),
					prior_mean_j, prior_prec_j, rng
				);
				state.coef_vec.head(state.num_alpha) = state.coef_mat.topRows(state.nrow_coef).reshaped();
			}
			if (exogen_updater) {
				state.coef_vec.segment(state.num_endog, state.num_exogen) = state.coef_mat.middleRows(state.nrow_endog, state.nrow_exogen).reshaped();
			}
			if (factor_updater) {
				state.coef_vec.tail(state.num_factor) = state.coef_mat.bottomRows(state.size_factor).reshaped();
			}
			draw_mn_savs(state.sparse_coef.col(j), state.coef_mat.col(j), state.x, penalty_j);
		}
	}
};

/**
 * @brief Draw strategy for the contemporaneous (impact) coefficients
 */
class ImpactUpdater {
public:
	ImpactUpdater() {}
	virtual ~ImpactUpdater() = default;

	/**
	 * @brief Draw the contemporaneous coefficients and their SAVS-sparsified counterpart
	 *
	 * @param state Shared mutable MCMC state
	 * @param rng RNG
	 */
	virtual void updateImpact(TriangularState& state, BVHAR_BHRNG& rng) = 0;
};

/**
 * @brief Time-invariant impact draw (current `McmcTriangular::updateImpact()`)
 */
class StaticImpactUpdater : public ImpactUpdater {
public:
	StaticImpactUpdater(int num_design) : response_contem(Eigen::VectorXd::Zero(num_design)) {}
	virtual ~StaticImpactUpdater() = default;

	void updateImpact(TriangularState& state, BVHAR_BHRNG& rng) override {
		for (int j = 1; j < state.dim; ++j) {
			response_contem = state.latent_innov.col(j).array() / state.sqrt_sv.col(j).array(); // n-dim
			Eigen::MatrixXd design_contem = state.latent_innov.leftCols(j).array().colwise() / state.sqrt_sv.col(j).reshaped().array(); // n x (j - 1)
			int contem_id = j * (j - 1) / 2;
			draw_coef(
				state.contem_coef.segment(contem_id, j),
				design_contem, response_contem,
				state.prior_chol_mean.segment(contem_id, j),
				state.prior_chol_prec.segment(contem_id, j),
				rng
			);
			draw_savs(state.sparse_contem.segment(contem_id, j), state.contem_coef.segment(contem_id, j), state.latent_innov.leftCols(j));
		}
	}

private:
	Eigen::VectorXd response_contem; // j-th column of Z0 = Y0 - X0 * A: n-dim
};

/**
 * @brief Draw strategy for the innovation covariance block (D in the LDLT decomposition)
 *
 * Also owns writing the combined per-step record row, since which `RegRecords::assignRecords()`
 * overload applies (LDLT vs. SV) depends on which variance state this updater carries.
 */
class VarianceUpdater {
public:
	VarianceUpdater() {}
	virtual ~VarianceUpdater() = default;

	/**
	 * @brief Recompute the diagonal scale used by the coefficient/impact draws (D^(1/2) in `sqrt_sv`)
	 */
	virtual void updateSv(TriangularState& state) = 0;

	/**
	 * @brief Draw the next variance state (diag_vec, or the SV latent volatility path)
	 */
	virtual void updateState(TriangularState& state, BVHAR_BHRNG& rng) = 0;

	/**
	 * @brief Save this step's full posterior draw (coefficient, impact, and variance state)
	 */
	virtual void updateRecords(int step, TriangularState& state, RegRecords& reg_record, SparseRecords& sparse_record) = 0;
};

/**
 * @brief Homoskedastic LDLT variance (current `McmcReg` variance logic)
 */
class LdltVarianceUpdater : public VarianceUpdater {
public:
	LdltVarianceUpdater(const Eigen::VectorXd& diag_init, const Eigen::VectorXd& sig_shp, const Eigen::VectorXd& sig_scl)
	: diag_vec(diag_init), prior_sig_shp(sig_shp), prior_sig_scl(sig_scl) {}
	virtual ~LdltVarianceUpdater() = default;

	void updateSv(TriangularState& state) override {
		state.sqrt_sv = diag_vec.cwiseSqrt().transpose().replicate(state.num_design, 1);
	}

	void updateState(TriangularState& state, BVHAR_BHRNG& rng) override {
		reg_ldlt_diag(diag_vec, prior_sig_shp, prior_sig_scl, state.latent_innov * state.chol_lower.transpose(), rng);
	}

	void updateRecords(int step, TriangularState& state, RegRecords& reg_record, SparseRecords& sparse_record) override {
		reg_record.assignRecords(step, state.coef_vec, state.contem_coef, diag_vec);
		sparse_record.assignRecords(step, state.num_alpha, state.dim, state.nrow_coef, state.num_exogen, state.nrow_exogen, state.sparse_coef, state.sparse_contem);
	}

	const Eigen::VectorXd& getDiag() const { return diag_vec; }

private:
	Eigen::VectorXd diag_vec; // inverse of d_i
	Eigen::VectorXd prior_sig_shp;
	Eigen::VectorXd prior_sig_scl;
};

/**
 * @brief Stochastic-volatility variance (current `McmcSv` variance logic)
 */
class SvVarianceUpdater : public VarianceUpdater {
public:
	SvVarianceUpdater(const SvParams& params, const SvInits& inits)
	: ortho_latent(Eigen::MatrixXd::Zero(params._num_design, params._dim)),
		lvol_draw(inits._lvol), lvol_init(inits._lvol_init), lvol_sig(inits._lvol_sig),
		prior_sig_shp(params._sig_shp), prior_sig_scl(params._sig_scl),
		prior_init_mean(params._init_mean), prior_init_prec(params._init_prec) {}
	virtual ~SvVarianceUpdater() = default;

	void updateSv(TriangularState& state) override {
		state.sqrt_sv = (lvol_draw / 2).array().exp();
	}

	void updateState(TriangularState& state, BVHAR_BHRNG& rng) override {
		ortho_latent = state.latent_innov * state.chol_lower.transpose(); // L eps_t <=> Z0 U
		ortho_latent = (ortho_latent.array().square() + .0001).array().log(); // adjustment log(e^2 + c) for some c = 10^(-4) against numerical problems
		for (int t = 0; t < state.dim; ++t) {
			varsv_ht(lvol_draw.col(t), lvol_init[t], lvol_sig[t], ortho_latent.col(t), rng);
		}
		varsv_sigh(lvol_sig, prior_sig_shp, prior_sig_scl, lvol_init, lvol_draw, rng);
		varsv_h0(lvol_init, prior_init_mean, prior_init_prec, lvol_draw.row(0), 1 / lvol_sig.array(), rng);
	}

	void updateRecords(int step, TriangularState& state, RegRecords& reg_record, SparseRecords& sparse_record) override {
		reg_record.assignRecords(step, state.coef_vec, state.contem_coef, lvol_draw, lvol_sig, lvol_init);
		sparse_record.assignRecords(step, state.num_alpha, state.dim, state.nrow_coef, state.num_exogen, state.nrow_exogen, state.sparse_coef, state.sparse_contem);
	}

	const Eigen::MatrixXd& getLvolDraw() const { return lvol_draw; }
	const Eigen::VectorXd& getLvolSig() const { return lvol_sig; }
	const Eigen::VectorXd& getLvolInit() const { return lvol_init; }

private:
	Eigen::MatrixXd ortho_latent; // orthogonalized Z0
	Eigen::MatrixXd lvol_draw; // h_j = (h_j1, ..., h_jn)
	Eigen::VectorXd lvol_init;
	Eigen::VectorXd lvol_sig;
	Eigen::VectorXd prior_sig_shp;
	Eigen::VectorXd prior_sig_scl;
	Eigen::VectorXd prior_init_mean;
	Eigen::VectorXd prior_init_prec;
};

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_TRIANGULAR_UPDATER_H
