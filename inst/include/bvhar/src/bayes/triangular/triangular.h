/**
 * @file triangular.h
 * @brief Header including MCMC algorithm
 */

#ifndef BVHAR_BAYES_TRIANGULAR_TRIANGULAR_H
#define BVHAR_BAYES_TRIANGULAR_TRIANGULAR_H

#include "./config.h"
#include "../shrinkage/shrinkage.h"
#include "../dfm/augment.h"
#include <type_traits>

namespace bvhar {

// MCMC algorithms
class McmcTriangular;
class McmcReg;
class McmcSv;
// Running MCMC
template <typename BaseMcmc, bool isGroup> class CtaRun;

/**
 * @brief Corrected Triangular Algorithm (CTA)
 * 
 * This class is a base class to conduct corrected triangular algorithm.
 * 
 */
class McmcTriangular : public McmcAlgo {
public:
	McmcTriangular(
		const RegParams& params, const RegInits& inits,
		std::unique_ptr<ShrinkageUpdater> coef_prior,
		std::unique_ptr<ShrinkageUpdater> contem_prior,
		unsigned int seed,
		BVHAR_OPTIONAL<std::unique_ptr<ShrinkageUpdater>> exogen_prior = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<std::unique_ptr<FactorAugmenter>> favar = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<std::unique_ptr<ShrinkageUpdater>> factor_prior = BVHAR_NULLOPT
	)
	: McmcAlgo(params, seed),
		include_mean(params._mean), x(params._x), y(params._y),
		dim(params._dim), dim_design(params._dim_design), num_design(params._num_design),
		num_lowerchol(params._num_lowerchol), num_coef(params._num_coef), num_alpha(params._num_alpha), nrow_coef(params._nrow),
		nrow_exogen(params._nrow_exogen), num_exogen(params._num_exogen),// num_endog(num_coef - num_exogen), nrow_endog(num_endog / dim),
		size_factor(params._size_factor), num_factor(params._num_factor),
		num_endog(params._num_endog), nrow_endog(num_endog / dim),
		nrow_varx(nrow_endog + nrow_exogen),
		coef_updater(std::move(coef_prior)), contem_updater(std::move(contem_prior)),
		own_id(params._own_id), grp_id(params._grp_id), grp_vec(params._grp_vec), num_grp(grp_id.size()),
		// reg_record(std::make_unique<RegRecords>(num_iter, dim, num_design, num_coef, num_lowerchol)),
		sparse_record(num_iter, dim, num_design, num_coef, num_lowerchol),
		coef_vec(Eigen::VectorXd::Zero(num_coef)), contem_coef(inits._contem),
		// prior_alpha_mean(Eigen::VectorXd::Zero(num_coef)),
		// prior_alpha_prec(Eigen::VectorXd::Zero(num_coef)),
		prior_alpha_mean(params._alpha_mean), prior_alpha_prec(params._alpha_prec),
		alpha_penalty(Eigen::VectorXd::Zero(num_alpha)),
		// prior_chol_mean(Eigen::VectorXd::Zero(num_lowerchol)),
		// prior_chol_prec(Eigen::VectorXd::Ones(num_lowerchol)),
		prior_chol_mean(params._chol_mean), prior_chol_prec(params._chol_prec),
		coef_mat(inits._coef), contem_id(0),
		sparse_coef(Eigen::MatrixXd::Zero(dim_design, dim)), sparse_contem(Eigen::VectorXd::Zero(num_lowerchol)),
		chol_lower(build_inv_lower(dim, contem_coef)),
		latent_innov(y - x * coef_mat),
		response_contem(Eigen::VectorXd::Zero(num_design)),
		sqrt_sv(Eigen::MatrixXd::Zero(num_design, dim)),
		prior_sig_shp(params._sig_shp), prior_sig_scl(params._sig_scl) {
		if (include_mean) {
			prior_alpha_mean.segment(num_alpha, dim) = params._mean_non;
			prior_alpha_prec.segment(num_alpha, dim) = 1 / (params._sd_non * Eigen::VectorXd::Ones(dim)).array().square();
		}
		coef_vec.head(num_alpha) = coef_mat.topRows(nrow_coef).reshaped();
		if (include_mean) {
			coef_vec.segment(num_alpha, dim) = coef_mat.middleRows<1>(nrow_coef).transpose();
		}
		if (exogen_prior) {
			exogen_updater = std::move(*exogen_prior);
			// coef_vec.tail(num_exogen) = coef_mat.bottomRows(nrow_exogen).reshaped();
			coef_vec.segment(num_endog, num_exogen) = coef_mat.middleRows(nrow_endog, nrow_exogen).reshaped();
		}
		if (factor_prior) {
			factor_updater = std::move(*factor_prior);
			coef_vec.tail(num_factor) = coef_mat.bottomRows(size_factor).reshaped();
		}
		if (favar) {
			favar_updater = std::move(*favar);
		}
		// reg_record->assignRecords(0, coef_vec, contem_coef, diag_vec);
		sparse_record.assignRecords(0, sparse_coef, sparse_contem);
		coef_updater->updateRecords(0);
		contem_updater->updateRecords(0);
	}
	virtual ~McmcTriangular() = default;

	/**
	 * @brief Append each class's additional record to the result `BVHAR_LIST`
	 * 
	 * @param list `BVHAR_LIST` containing MCMC record result
	 */
	void appendRecords(BVHAR_LIST& list) {
		coef_updater->appendCoefRecords(list);
		contem_updater->appendContemRecords(list);
		if (favar_updater) {
			favar_updater->appendRecords(list);
		}
	}

	void doWarmUp() override {
		std::lock_guard<std::mutex> lock(mtx);
		updateCoefPrec();
		updatePenalty();
		updateSv();
		updateCoef();
		updateImpactPrec();
		updateLatent();
		updateImpact();
		updateChol();
		updateState();
	}

	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefPrec();
		updatePenalty();
		updateSv(); // D before coef
		updateCoef();
		updateImpactPrec();
		updateLatent(); // E_t before a
		updateImpact();
		updateChol(); // L before d_i
		updateState();
		updateRecords();
	}

	BVHAR_LIST returnRecords(int num_burn, int thin) override {
		BVHAR_LIST res = gatherRecords();
		appendRecords(res);
		for (auto& record : res) {
			if (BVHAR_IS_MATRIX(BVHAR_ACCESS_LIST(record, res))) {
				BVHAR_ACCESS_LIST(record, res) = thin_record(BVHAR_CAST<Eigen::MatrixXd>(BVHAR_ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			} else {
				BVHAR_ACCESS_LIST(record, res) = thin_record(BVHAR_CAST<Eigen::VectorXd>(BVHAR_ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			}
		}
		return res;
	}

	/**
	 * @brief Return `LdltRecords`
	 * 
	 * @param num_burn Number of burn-in
	 * @param thin Thinning
	 * @param sparse If `true`, return sparsified draws.
	 * @return LdltRecords `LdltRecords` object
	 */
	LdltRecords returnLdltRecords(int num_burn, int thin, bool sparse = false) const {
		return reg_record->returnLdltRecords(sparse_record, num_iter, num_burn, thin, sparse);
	}

	/**
	 * @brief Return `SvRecords`
	 * 
	 * @param num_burn Number of burn-in
	 * @param thin Thinning
	 * @param sparse If `true`, return sparsified draws.
	 * @return SvRecords `SvRecords` object
	 */
	SvRecords returnSvRecords(int num_burn, int thin, bool sparse = false) const {
		return reg_record->returnSvRecords(sparse_record, num_iter, num_burn, thin, sparse);
	}

	/**
	 * @brief Return `LdltRecords` or `SvRecords`
	 * 
	 * @tparam RecordType `LdltRecords` or `SvRecords` 
	 * @param num_burn Number of burn-in
	 * @param thin Thinning
	 * @param sparse If `true`, return sparsified draws.
	 * @return RecordType `LdltRecords` or `SvRecords` 
	 */
	template <typename RecordType>
	RecordType returnStructRecords(int num_burn, int thin, bool sparse = false) const {
		return reg_record->returnRecords<RecordType>(sparse_record, num_iter, num_burn, thin, sparse);
	}

	template <typename RecordType>
	RecordType returnFactorRecords(int num_burn, int thin) const {
		return favar_updater->returnStructRecords<RecordType>(num_burn, thin);
	}

protected:
	bool include_mean;
	Eigen::MatrixXd x;
	Eigen::MatrixXd y;
	int dim; // k
  int dim_design; // kp(+1)
  int num_design; // n = T - p
  int num_lowerchol;
  int num_coef;
	int num_alpha;
	int nrow_coef;
	// int nrow_exogen, num_exogen, num_endog;
	int nrow_exogen, num_exogen;
	int size_factor, num_factor, num_endog, nrow_endog, nrow_varx;
	std::unique_ptr<ShrinkageUpdater> coef_updater;
	std::unique_ptr<ShrinkageUpdater> contem_updater;
	std::unique_ptr<ShrinkageUpdater> exogen_updater;
	std::unique_ptr<ShrinkageUpdater> factor_updater;
	std::unique_ptr<FactorAugmenter> favar_updater;
	// std::unique_ptr<FactorAugmenter>
	std::set<int> own_id;
	Eigen::VectorXi grp_id;
	Eigen::VectorXi grp_vec;
	int num_grp;
	std::unique_ptr<RegRecords> reg_record;
	SparseRecords sparse_record;
	Eigen::VectorXd coef_vec;
	Eigen::VectorXd contem_coef;
	Eigen::VectorXd prior_alpha_mean; // prior mean vector of alpha
	Eigen::VectorXd prior_alpha_prec; // Diagonal of alpha prior precision
	Eigen::VectorXd alpha_penalty; // SAVS penalty vector
	Eigen::VectorXd prior_chol_mean; // prior mean vector of a = 0
	Eigen::VectorXd prior_chol_prec; // Diagonal of prior precision of a = I
	Eigen::MatrixXd coef_mat;
	int contem_id;
	Eigen::MatrixXd sparse_coef;
	Eigen::VectorXd sparse_contem;
	Eigen::MatrixXd chol_lower; // L in Sig_t^(-1) = L D_t^(-1) LT
	Eigen::MatrixXd latent_innov; // Z0 = Y0 - X0 A = (eps_p+1, eps_p+2, ..., eps_n+p)^T
	Eigen::VectorXd response_contem; // j-th column of Z0 = Y0 - X0 * A: n-dim
	Eigen::MatrixXd sqrt_sv; // stack sqrt of exp(h_t) = (exp(-h_1t / 2), ..., exp(-h_kt / 2)), t = 1, ..., n => n x k
	Eigen::VectorXd prior_sig_shp;
	Eigen::VectorXd prior_sig_scl;

	/**
	 * @brief Draw state vector
	 * 
	 */
	virtual void updateState() = 0;

	/**
	 * @brief Compute D
	 * 
	 */
	virtual void updateSv() = 0;

	/**
	 * @brief Save coefficient records
	 * 
	 */
	virtual void updateCoefRecords() = 0;

	/**
	 * @brief Draw precision of coefficient based on each shrinkage priors
	 * 
	 */
	void updateCoefPrec() {
		coef_updater->updateCoefPrec(
			prior_alpha_prec.head(num_alpha), coef_vec.head(num_alpha),
      num_grp, grp_vec, grp_id,
      rng
		);
		if (exogen_updater) {
			// exogen_updater->updateImpactPrec(prior_alpha_prec.tail(num_exogen), coef_vec.tail(num_exogen), rng);
			exogen_updater->updateImpactPrec(prior_alpha_prec.segment(num_endog, num_exogen), coef_vec.segment(num_endog, num_exogen), rng);
		}
		if (factor_updater) {
			factor_updater->updateImpactPrec(prior_alpha_prec.tail(num_factor), coef_vec.tail(num_factor), rng);
		}
	}

	/**
	 * @brief Update SAVS penalty
	 * 
	 */
	void updatePenalty() {
		for (int i = 0; i < num_alpha; ++i) {
			if (own_id.find(grp_vec[i]) != own_id.end()) {
				alpha_penalty[i] = 0;
			} else {
				alpha_penalty[i] = 1;
			}
		}
	}
	// virtual void updatePenalty() = 0;

	/**
	 * @brief Draw precision of contemporaneous coefficient based on each shrinkage priors
	 * 
	 */
	void updateImpactPrec() {
		contem_updater->updateImpactPrec(prior_chol_prec, contem_coef, rng);
	}

	/**
	 * @brief Save MCMC records
	 * 
	 */
	void updateRecords() {
		updateCoefRecords();
		coef_updater->updateRecords(mcmc_step);
		contem_updater->updateRecords(mcmc_step);
	}
	// virtual void updateRecords() = 0;

	/**
	 * @brief Draw coefficients
	 * 
	 */
	void updateCoef() {
		if (favar_updater) {
			favar_updater->updateResid(x, y, coef_mat);
			favar_updater->updateFactor(coef_mat, chol_lower, sqrt_sv, rng);
			favar_updater->appendDesign(x);
		}
		for (int j = 0; j < dim; ++j) {
			coef_mat.col(j).setZero(); // j-th column of A = 0
			Eigen::MatrixXd chol_lower_j = chol_lower.bottomRows(dim - j); // L_(j:k) = a_jt to a_kt for t = 1, ..., j - 1
			Eigen::MatrixXd sqrt_sv_j = sqrt_sv.rightCols(dim - j); // use h_jt to h_kt for t = 1, .. n => (k - j + 1) x k
			Eigen::MatrixXd design_coef = kronecker_eigen(chol_lower_j.col(j), x).array().colwise() / sqrt_sv_j.reshaped().array(); // L_(j:k, j) otimes X0 scaled by D_(1:n, j:k): n(k - j + 1) x kp
			Eigen::VectorXd prior_mean_j(dim_design);
			Eigen::VectorXd prior_prec_j(dim_design);
			Eigen::VectorXd penalty_j = Eigen::VectorXd::Zero(dim_design);
			prior_mean_j.head(nrow_coef) = prior_alpha_mean.segment(j * nrow_coef, nrow_coef);
			prior_prec_j.head(nrow_coef) = prior_alpha_prec.segment(j * nrow_coef, nrow_coef);
			penalty_j.head(nrow_coef) = alpha_penalty.segment(j * nrow_coef, nrow_coef);
			if (include_mean) {
				// prior_mean_j << prior_alpha_mean.segment(j * nrow_coef, nrow_coef), prior_alpha_mean.segment(num_alpha, dim)[j];
				// prior_prec_j << prior_alpha_prec.segment(j * nrow_coef, nrow_coef), prior_alpha_prec.segment(num_alpha, dim)[j];
				// penalty_j << alpha_penalty.segment(j * nrow_coef, nrow_coef), alpha_penalty.tail(dim)[j];
				// penalty_j.head(nrow_coef) = alpha_penalty.segment(j * nrow_coef, nrow_coef);
				prior_mean_j[nrow_coef] = prior_alpha_mean.segment(num_alpha, dim)[j];
				prior_prec_j[nrow_coef] = prior_alpha_prec.segment(num_alpha, dim)[j];
				if (exogen_updater) {
					// prior_mean_j.tail(nrow_exogen) = prior_alpha_mean.segment(num_endog + j * nrow_exogen, nrow_exogen);
					// prior_prec_j.tail(nrow_exogen) = prior_alpha_prec.segment(num_endog + j * nrow_exogen, nrow_exogen);
					prior_mean_j.segment(nrow_endog, nrow_exogen) = prior_alpha_mean.segment(num_endog + j * nrow_exogen, nrow_exogen);
					prior_prec_j.segment(nrow_endog, nrow_exogen) = prior_alpha_prec.segment(num_endog + j * nrow_exogen, nrow_exogen);
					// penalty_j.tail(nrow_exogen): current alpha_penalty only covers VAR
				}
				if (factor_updater) {
					prior_mean_j.tail(size_factor) = prior_alpha_mean.segment(num_endog + num_exogen + j * size_factor, size_factor);
					prior_prec_j.tail(size_factor) = prior_alpha_prec.segment(num_endog + num_exogen + j * size_factor, size_factor);
				}
				draw_coef(
					coef_mat.col(j), design_coef,
					(((y - x * coef_mat) * chol_lower_j.transpose()).array() / sqrt_sv_j.array()).reshaped(), // Hadamard product between: (Y - X0 A(-j))L_(j:k)^T and D_(1:n, j:k)
					prior_mean_j, prior_prec_j, rng
				);
				coef_vec.head(num_alpha) = coef_mat.topRows(nrow_coef).reshaped();
				coef_vec.segment(num_alpha, dim) = coef_mat.middleRows<1>(nrow_coef).transpose();
			} else {
				// prior_mean_j = prior_alpha_mean.segment(dim_design * j, dim_design);
				// prior_prec_j = prior_alpha_prec.segment(dim_design * j, dim_design);
				// penalty_j = alpha_penalty.segment(dim_design * j, dim_design);
				if (exogen_updater) {
					// prior_mean_j.tail(nrow_exogen) = prior_alpha_mean.segment(num_endog + j * nrow_exogen, nrow_exogen);
					// prior_prec_j.tail(nrow_exogen) = prior_alpha_prec.segment(num_endog + j * nrow_exogen, nrow_exogen);
					prior_mean_j.segment(nrow_endog, nrow_exogen) = prior_alpha_mean.segment(num_endog + j * nrow_exogen, nrow_exogen);
					prior_prec_j.segment(nrow_endog, nrow_exogen) = prior_alpha_prec.segment(num_endog + j * nrow_exogen, nrow_exogen);
				}
				if (factor_updater) {
					prior_mean_j.tail(size_factor) = prior_alpha_mean.segment(num_endog + num_exogen + j * size_factor, size_factor);
					prior_prec_j.tail(size_factor) = prior_alpha_prec.segment(num_endog + num_exogen + j * size_factor, size_factor);
				}
				draw_coef(
					coef_mat.col(j),
					design_coef,
					(((y - x * coef_mat) * chol_lower_j.transpose()).array() / sqrt_sv_j.array()).reshaped(),
					prior_mean_j, prior_prec_j, rng
				);
				// coef_vec = coef_mat.reshaped();
				coef_vec.head(num_alpha) = coef_mat.topRows(nrow_coef).reshaped();
			}
			if (exogen_updater) {
				// coef_vec.tail(num_exogen) = coef_mat.bottomRows(nrow_exogen).reshaped();
				coef_vec.segment(num_endog, num_exogen) = coef_mat.middleRows(nrow_endog, nrow_exogen).reshaped();
			}
			if (factor_updater) {
				coef_vec.tail(num_factor) = coef_mat.bottomRows(size_factor).reshaped();
			}
			draw_mn_savs(sparse_coef.col(j), coef_mat.col(j), x, penalty_j);
		}
	}

	/**
	 * @brief Draw contemporaneous coefficients
	 * 
	 */
	void updateImpact() {
		for (int j = 1; j < dim; ++j) {
			response_contem = latent_innov.col(j).array() / sqrt_sv.col(j).array(); // n-dim
			Eigen::MatrixXd design_contem = latent_innov.leftCols(j).array().colwise() / sqrt_sv.col(j).reshaped().array(); // n x (j - 1)
			contem_id = j * (j - 1) / 2;
			draw_coef(
				contem_coef.segment(contem_id, j),
				design_contem, response_contem,
				prior_chol_mean.segment(contem_id, j),
				prior_chol_prec.segment(contem_id, j),
				rng
			);
			draw_savs(sparse_contem.segment(contem_id, j), contem_coef.segment(contem_id, j), latent_innov.leftCols(j));
		}
	}

	/**
	 * @brief Compute residual matrix for orthogonalization
	 * 
	 */
	void updateLatent() { latent_innov = y - x * coef_mat; }

	/**
	 * @brief Compute L
	 * 
	 */
	void updateChol() { chol_lower = build_inv_lower(dim, contem_coef); }

	/**
	 * @brief Gather MCMC records
	 * 
	 * @return BVHAR_LIST 
	 */
	BVHAR_LIST gatherRecords() {
		BVHAR_LIST res = reg_record->returnListRecords(dim, num_alpha, num_exogen, include_mean);
		reg_record->appendRecords(res);
		sparse_record.appendRecords(res, dim, num_alpha, num_exogen, include_mean);
		return res;
	}
};

/**
 * @brief MCMC for homoskedastic LDLT parameterization
 * 
 */
class McmcReg : public McmcTriangular {
public:
	McmcReg(
		const RegParams& params, const LdltInits& inits,
		std::unique_ptr<ShrinkageUpdater> coef_prior,
		std::unique_ptr<ShrinkageUpdater> contem_prior,
		unsigned int seed,
		BVHAR_OPTIONAL<std::unique_ptr<ShrinkageUpdater>> exogen_prior = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<std::unique_ptr<FactorAugmenter>> favar = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<std::unique_ptr<ShrinkageUpdater>> factor_prior = BVHAR_NULLOPT
	)
	: McmcTriangular(params, inits, std::move(coef_prior), std::move(contem_prior), seed, std::move(exogen_prior), std::move(favar), std::move(factor_prior)),
		diag_vec(inits._diag) {
		reg_record = std::make_unique<LdltRecords>(num_iter, dim, num_design, num_coef, num_lowerchol);
		reg_record->assignRecords(0, coef_vec, contem_coef, diag_vec);
	}
	virtual ~McmcReg() = default;

protected:
	void updateState() override { reg_ldlt_diag(diag_vec, prior_sig_shp, prior_sig_scl, latent_innov * chol_lower.transpose(), rng); }
	void updateSv() override { sqrt_sv = diag_vec.cwiseSqrt().transpose().replicate(num_design, 1); }
	void updateCoefRecords() override {
		reg_record->assignRecords(mcmc_step, coef_vec, contem_coef, diag_vec);
		sparse_record.assignRecords(mcmc_step, num_alpha, dim, nrow_coef, num_exogen, nrow_exogen, sparse_coef, sparse_contem);
	}

private:
	Eigen::VectorXd diag_vec; // inverse of d_i
};

/**
 * @brief MCMC for stochastic volatility
 * 
 */
class McmcSv : public McmcTriangular {
public:
	McmcSv(
		const SvParams& params, const SvInits& inits,
		std::unique_ptr<ShrinkageUpdater> coef_prior,
		std::unique_ptr<ShrinkageUpdater> contem_prior,
		unsigned int seed,
		BVHAR_OPTIONAL<std::unique_ptr<ShrinkageUpdater>> exogen_prior = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<std::unique_ptr<FactorAugmenter>> favar = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<std::unique_ptr<ShrinkageUpdater>> factor_prior = BVHAR_NULLOPT
	)
	: McmcTriangular(params, inits, std::move(coef_prior), std::move(contem_prior), seed, std::move(exogen_prior), std::move(favar), std::move(factor_prior)),
		ortho_latent(Eigen::MatrixXd::Zero(num_design, dim)),
		lvol_draw(inits._lvol), lvol_init(inits._lvol_init), lvol_sig(inits._lvol_sig),
		prior_init_mean(params._init_mean), prior_init_prec(params._init_prec) {
		reg_record = std::make_unique<SvRecords>(num_iter, dim, num_design, num_coef, num_lowerchol);
		reg_record->assignRecords(0, coef_vec, contem_coef, lvol_draw, lvol_sig, lvol_init);
		sparse_record.assignRecords(0, sparse_coef, sparse_contem);
	}
	virtual ~McmcSv() = default;

protected:
	void updateState() override {
		ortho_latent = latent_innov * chol_lower.transpose(); // L eps_t <=> Z0 U
		ortho_latent = (ortho_latent.array().square() + .0001).array().log(); // adjustment log(e^2 + c) for some c = 10^(-4) against numerical problems
		for (int t = 0; t < dim; t++) {
			varsv_ht(lvol_draw.col(t), lvol_init[t], lvol_sig[t], ortho_latent.col(t), rng);
		}
		varsv_sigh(lvol_sig, prior_sig_shp, prior_sig_scl, lvol_init, lvol_draw, rng);
		varsv_h0(lvol_init, prior_init_mean, prior_init_prec, lvol_draw.row(0), 1 / lvol_sig.array(), rng);
	}
	void updateSv() override { sqrt_sv = (lvol_draw / 2).array().exp(); }
	void updateCoefRecords() override {
		reg_record->assignRecords(mcmc_step, coef_vec, contem_coef, lvol_draw, lvol_sig, lvol_init);
		sparse_record.assignRecords(mcmc_step, num_alpha, dim, nrow_coef, num_exogen, nrow_exogen, sparse_coef, sparse_contem);
	}

private:
	Eigen::MatrixXd ortho_latent; // orthogonalized Z0
	Eigen::MatrixXd lvol_draw; // h_j = (h_j1, ..., h_jn)
	Eigen::VectorXd lvol_init;
	Eigen::VectorXd lvol_sig;
	Eigen::VectorXd prior_init_mean;
	Eigen::VectorXd prior_init_prec;
};

/**
 * @brief Function to initialize `McmcReg` or `McmcSv`
 * 
 * @tparam BaseMcmc `McmcReg` or `McmcSv`
 * @tparam isGroup If `true`, use group shrinkage parameter
 * @param num_chains Number of MCMC chains
 * @param num_iter MCMC iteration
 * @param x Design matrix in multivariate regression form
 * @param y Response matrix in multivariat regression form
 * @param param_reg Covariance configuration
 * @param param_prior Shrinkage prior configuration
 * @param param_intercept Constant term configuration
 * @param param_init MCMC initial values
 * @param prior_type Prior number to use
 * @param contem_prior Contemporaneous shrinkage prior configuration
 * @param contem_init MCMC initial values for Contemporaneous shrinkage prior
 * @param contem_prior_type Contemporaneous shrinkage prior number to use
 * @param grp_id Minnesota group unique ids
 * @param own_id Own-lag id
 * @param cross_id Cross-lag id
 * @param grp_mat Minnesota group matrix
 * @param include_mean If `true`, include constant term
 * @param seed_chain Seed for each chain
 * @param num_design Number of samples
 * @param exogen_prior Exogenous shrinkage prior configuration
 * @param exogen_init MCMC initial values for Exogenous shrinkage prior
 * @param exogen_prior_type Exogenous shrinkage prior number to use
 * @param exogen_cols The number of exogenous design matrix columns
 * @return std::vector<std::unique_ptr<BaseMcmc>> 
 */
template <typename BaseMcmc = McmcReg, bool isGroup = true>
inline std::vector<std::unique_ptr<BaseMcmc>> initialize_mcmc(
	int num_chains, int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
	BVHAR_LIST& param_reg, BVHAR_LIST& param_prior, BVHAR_LIST& param_intercept, BVHAR_LIST_OF_LIST& param_init, int prior_type,
	BVHAR_LIST& contem_prior, BVHAR_LIST_OF_LIST& contem_init, int contem_prior_type,
  const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
  bool include_mean, Eigen::Ref<const Eigen::VectorXi> seed_chain, BVHAR_OPTIONAL<int> num_design = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<BVHAR_LIST> exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_cols = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<BVHAR_LIST> factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> factor_prior_type = BVHAR_NULLOPT,
	BVHAR_OPTIONAL<int> size_factor = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> factor_lag = BVHAR_NULLOPT
) {
	using PARAMS = typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, RegParams, SvParams>::type;
	using INITS = typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, LdltInits, SvInits>::type;
	// PARAMS base_params = exogen_prior ? PARAMS(
	// 	num_iter, x, y,
	// 	param_reg,
	// 	own_id, cross_id,
	// 	grp_id, grp_mat,
	// 	param_intercept, include_mean,
	// 	*exogen_cols
	// ) : PARAMS(
	// 	num_iter, x, y,
	// 	param_reg,
	// 	own_id, cross_id,
	// 	grp_id, grp_mat,
	// 	param_intercept, include_mean
	// );
	PARAMS base_params(
		num_iter, x, y,
		param_reg,
		own_id, cross_id,
		grp_id, grp_mat,
		param_intercept, include_mean,
		exogen_cols, size_factor
	);
	std::vector<std::unique_ptr<BaseMcmc>> mcmc_ptr(num_chains);
	for (int i = 0; i < num_chains; ++i) {
		BVHAR_LIST init_spec = param_init[i];
		auto coef_updater = initialize_shrinkageupdater<isGroup>(num_iter, param_prior, init_spec, prior_type);
		coef_updater->initCoefMean(base_params._alpha_mean.head(base_params._num_alpha));
		coef_updater->initCoefPrec(base_params._alpha_prec.head(base_params._num_alpha), base_params._grp_vec, base_params._cross_id);
		BVHAR_LIST contem_init_spec = contem_init[i];
		auto contem_updater = initialize_shrinkageupdater<false>(num_iter, contem_prior, contem_init_spec, contem_prior_type);
		contem_updater->initImpactPrec(base_params._chol_prec);
		INITS ldlt_inits = num_design ? INITS(init_spec, *num_design) : INITS(init_spec);
		BVHAR_OPTIONAL<std::unique_ptr<ShrinkageUpdater>> exogen_updater = BVHAR_NULLOPT;
		BVHAR_OPTIONAL<std::unique_ptr<ShrinkageUpdater>> factor_updater = BVHAR_NULLOPT;
		BVHAR_OPTIONAL<std::unique_ptr<FactorAugmenter>> favar_updater = BVHAR_NULLOPT;
		if (exogen_prior) {
			BVHAR_LIST exogen_init_spec = (*exogen_init)[i];
			// auto exogen_updater = initialize_shrinkageupdater<false>(num_iter, *exogen_prior, exogen_init_spec, *exogen_prior_type);
			// exogen_updater->initCoefMean(base_params._alpha_mean.tail(base_params._num_exogen));
			// exogen_updater->initImpactPrec(base_params._alpha_prec.tail(base_params._num_exogen));
			exogen_updater = initialize_shrinkageupdater<false>(num_iter, *exogen_prior, exogen_init_spec, *exogen_prior_type);
			(*exogen_updater)->initCoefMean(base_params._alpha_mean.segment(base_params._num_endog, base_params._num_exogen));
			(*exogen_updater)->initImpactPrec(base_params._alpha_prec.segment(base_params._num_endog, base_params._num_exogen));
			// mcmc_ptr[i] = std::make_unique<BaseMcmc>(
			// 	base_params, ldlt_inits,
			// 	std::move(coef_updater), std::move(contem_updater),
			// 	static_cast<unsigned int>(seed_chain[i]),
			// 	std::move(exogen_updater)
			// );
		} else {
			// mcmc_ptr[i] = std::make_unique<BaseMcmc>(
			// 	base_params, ldlt_inits,
			// 	std::move(coef_updater), std::move(contem_updater),
			// 	static_cast<unsigned int>(seed_chain[i])
			// );
		}
		if (factor_prior) {
			BVHAR_LIST factor_init_spec = (*factor_init)[i];
			factor_updater = initialize_shrinkageupdater<false>(num_iter, *factor_prior, factor_init_spec, *factor_prior_type);
			(*exogen_updater)->initCoefMean(base_params._alpha_mean.tail(base_params._num_factor));
			(*exogen_updater)->initImpactPrec(base_params._alpha_prec.tail(base_params._num_factor));
			DfmParams dfm_params(0, *size_factor, base_params._dim);
			favar_updater = std::make_unique<FactorNormalAugmenter>(num_iter, base_params._num_design, dfm_params);
		}
		mcmc_ptr[i] = std::make_unique<BaseMcmc>(
			base_params, ldlt_inits,
			std::move(coef_updater), std::move(contem_updater),
			static_cast<unsigned int>(seed_chain[i]),
			std::move(exogen_updater),
			std::move(favar_updater), std::move(factor_updater)
		);
	}
	return mcmc_ptr;
}

/**
 * @brief Class that conducts MCMC using CTA
 * 
 * @tparam BaseMcmc `McmcReg` or `McmcSv`
 * @tparam isGroup If `true`, use group shrinkage parameter
 */
template <typename BaseMcmc = McmcReg, bool isGroup = true>
class CtaRun : public McmcRun {
public:
	CtaRun(
		int num_chains, int num_iter, int num_burn, int thin,
    const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		BVHAR_LIST& param_cov, BVHAR_LIST& param_prior, BVHAR_LIST& param_intercept,
		BVHAR_LIST_OF_LIST& param_init, int prior_type,
		BVHAR_LIST& contem_prior, BVHAR_LIST_OF_LIST& contem_init, int contem_prior_type,
    const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
    bool include_mean, const Eigen::VectorXi& seed_chain, bool display_progress, int nthreads,
		BVHAR_OPTIONAL<BVHAR_LIST> exogen_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> exogen_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_prior_type = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> exogen_cols = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<BVHAR_LIST> factor_prior = BVHAR_NULLOPT, BVHAR_OPTIONAL<BVHAR_LIST_OF_LIST> factor_init = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> factor_prior_type = BVHAR_NULLOPT,
		BVHAR_OPTIONAL<int> size_factor = BVHAR_NULLOPT, BVHAR_OPTIONAL<int> factor_lag = BVHAR_NULLOPT
	)
	: McmcRun(num_chains, num_iter, num_burn, thin, display_progress, nthreads) {
		auto temp_mcmc = initialize_mcmc<BaseMcmc, isGroup>(
			num_chains, num_iter - num_burn, x, y,
			param_cov, param_prior, param_intercept, param_init, prior_type,
			contem_prior, contem_init, contem_prior_type,
			grp_id, own_id, cross_id, grp_mat,
			include_mean, seed_chain,
			BVHAR_NULLOPT, exogen_prior, exogen_init, exogen_prior_type, exogen_cols,
			factor_prior, factor_init, factor_prior_type, size_factor, factor_lag
		);
		for (int i = 0; i < num_chains; ++i) {
			mcmc_ptr[i] = std::move(temp_mcmc[i]);
		}
	}
	virtual ~CtaRun() = default;
};

} // namespace bvhar

#endif // BVHAR_BAYES_TRIANGULAR_TRIANGULAR_H