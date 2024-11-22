#ifndef BVHARMCMC_H
#define BVHARMCMC_H

#include "bvharconfig.h"
#include "bvharprogress.h"
#include "bvharinterrupt.h"
#include <type_traits>

namespace bvhar {

// MCMC algorithms
class McmcTriangular;
class McmcReg;
class McmcSv;
// Shrinkage priors
template <typename BaseMcmc> class McmcMinn;
template <typename BaseMcmc> class McmcHierminn;
template <typename BaseMcmc> class McmcSsvs;
template <typename BaseMcmc, bool isGroup> class McmcHorseshoe;
template <typename BaseMcmc, bool isGroup> class McmcNg;
template <typename BaseMcmc, bool isGroup> class McmcDl;
// Running MCMC
class McmcInterface;
template <typename BaseMcmc, bool isGroup> class McmcRun;

class McmcTriangular {
public:
	McmcTriangular(const RegParams& params, const RegInits& inits, unsigned int seed)
	: include_mean(params._mean), x(params._x), y(params._y),
		num_iter(params._iter), dim(params._dim), dim_design(params._dim_design), num_design(params._num_design),
		num_lowerchol(params._num_lowerchol), num_coef(params._num_coef), num_alpha(params._num_alpha), nrow_coef(params._nrow),
		// reg_record(std::make_unique<RegRecords>(num_iter, dim, num_design, num_coef, num_lowerchol)),
		sparse_record(num_iter, dim, num_design, num_coef, num_lowerchol),
		mcmc_step(0), rng(seed),
		coef_vec(Eigen::VectorXd::Zero(num_coef)), contem_coef(inits._contem),
		prior_alpha_mean(Eigen::VectorXd::Zero(num_coef)),
		prior_alpha_prec(Eigen::VectorXd::Zero(num_coef)),
		alpha_penalty(Eigen::VectorXd::Zero(num_alpha)),
		prior_chol_mean(Eigen::VectorXd::Zero(num_lowerchol)),
		prior_chol_prec(Eigen::VectorXd::Ones(num_lowerchol)),
		coef_mat(inits._coef), contem_id(0),
		sparse_coef(Eigen::MatrixXd::Zero(dim_design, dim)), sparse_contem(Eigen::VectorXd::Zero(num_lowerchol)),
		chol_lower(build_inv_lower(dim, contem_coef)),
		latent_innov(y - x * coef_mat),
		response_contem(Eigen::VectorXd::Zero(num_design)),
		sqrt_sv(Eigen::MatrixXd::Zero(num_design, dim)),
		prior_sig_shp(params._sig_shp), prior_sig_scl(params._sig_scl) {
		if (include_mean) {
			prior_alpha_mean.tail(dim) = params._mean_non;
			prior_alpha_prec.tail(dim) = 1 / (params._sd_non * Eigen::VectorXd::Ones(dim)).array().square();
		}
		coef_vec.head(num_alpha) = coef_mat.topRows(nrow_coef).reshaped();
		if (include_mean) {
			coef_vec.tail(dim) = coef_mat.bottomRows(1).transpose();
		}
		// reg_record->assignRecords(0, coef_vec, contem_coef, diag_vec);
		sparse_record.assignRecords(0, sparse_coef, sparse_contem);
	}
	virtual ~McmcTriangular() = default;
	virtual void appendRecords(LIST& list) = 0;
	void doWarmUp() {
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
	void doPosteriorDraws() {
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
	LIST gatherRecords() {
		LIST res = reg_record->returnListRecords(dim, num_alpha, include_mean);
		reg_record->appendRecords(res);
		sparse_record.appendRecords(res, dim, num_alpha, include_mean);
		return res;
	}
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
	LdltRecords returnLdltRecords(int num_burn, int thin, bool sparse = false) const {
		return reg_record->returnLdltRecords(sparse_record, num_iter, num_burn, thin, sparse);
	}
	SvRecords returnSvRecords(int num_burn, int thin, bool sparse = false) const {
		return reg_record->returnSvRecords(sparse_record, num_iter, num_burn, thin, sparse);
	}
	template <typename RecordType>
	RecordType returnStructRecords(int num_burn, int thin, bool sparse = false) const {
		return reg_record->returnRecords<RecordType>(sparse_record, num_iter, num_burn, thin, sparse);
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
	int nrow_coef;
	std::unique_ptr<RegRecords> reg_record;
	SparseRecords sparse_record;
	std::atomic<int> mcmc_step; // MCMC step
	boost::random::mt19937 rng; // RNG instance for multi-chain
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
	virtual void updateState() = 0;
	virtual void updateSv() = 0;
	virtual void updateCoefRecords() = 0;
	virtual void updateCoefPrec() = 0;
	virtual void updatePenalty() = 0;
	virtual void updateImpactPrec() = 0;
	virtual void updateRecords() = 0;
	void updateCoef() {
		for (int j = 0; j < dim; ++j) {
			coef_mat.col(j).setZero(); // j-th column of A = 0
			Eigen::MatrixXd chol_lower_j = chol_lower.bottomRows(dim - j); // L_(j:k) = a_jt to a_kt for t = 1, ..., j - 1
			Eigen::MatrixXd sqrt_sv_j = sqrt_sv.rightCols(dim - j); // use h_jt to h_kt for t = 1, .. n => (k - j + 1) x k
			Eigen::MatrixXd design_coef = kronecker_eigen(chol_lower_j.col(j), x).array().colwise() / sqrt_sv_j.reshaped().array(); // L_(j:k, j) otimes X0 scaled by D_(1:n, j:k): n(k - j + 1) x kp
			Eigen::VectorXd prior_mean_j(dim_design);
			Eigen::VectorXd prior_prec_j(dim_design);
			Eigen::VectorXd penalty_j = Eigen::VectorXd::Zero(dim_design);
			if (include_mean) {
				prior_mean_j << prior_alpha_mean.segment(j * nrow_coef, nrow_coef), prior_alpha_mean.tail(dim)[j];
				prior_prec_j << prior_alpha_prec.segment(j * nrow_coef, nrow_coef), prior_alpha_prec.tail(dim)[j];
				// penalty_j << alpha_penalty.segment(j * nrow_coef, nrow_coef), alpha_penalty.tail(dim)[j];
				penalty_j.head(nrow_coef) = alpha_penalty.segment(j * nrow_coef, nrow_coef);
				draw_coef(
					coef_mat.col(j), design_coef,
					(((y - x * coef_mat) * chol_lower_j.transpose()).array() / sqrt_sv_j.array()).reshaped(), // Hadamard product between: (Y - X0 A(-j))L_(j:k)^T and D_(1:n, j:k)
					prior_mean_j, prior_prec_j, rng
				);
				coef_vec.head(num_alpha) = coef_mat.topRows(nrow_coef).reshaped();
				coef_vec.tail(dim) = coef_mat.bottomRows(1).transpose();
			} else {
				prior_mean_j = prior_alpha_mean.segment(dim_design * j, dim_design);
				prior_prec_j = prior_alpha_prec.segment(dim_design * j, dim_design);
				penalty_j = alpha_penalty.segment(dim_design * j, dim_design);
				draw_coef(
					coef_mat.col(j),
					design_coef,
					(((y - x * coef_mat) * chol_lower_j.transpose()).array() / sqrt_sv_j.array()).reshaped(),
					prior_mean_j, prior_prec_j, rng
				);
				coef_vec = coef_mat.reshaped();
			}
			draw_mn_savs(sparse_coef.col(j), coef_mat.col(j), x, penalty_j);
		}
	}
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
	void updateLatent() { latent_innov = y - x * coef_mat; }
	void updateChol() { chol_lower = build_inv_lower(dim, contem_coef); }
	void addStep() { mcmc_step++; }
};

class McmcReg : public McmcTriangular {
public:
	McmcReg(const RegParams& params, const LdltInits& inits, unsigned int seed)
	: McmcTriangular(params, inits, seed), diag_vec(inits._diag) {
		reg_record = std::make_unique<LdltRecords>(num_iter, dim, num_design, num_coef, num_lowerchol);
		reg_record->assignRecords(0, coef_vec, contem_coef, diag_vec);
	}
	virtual ~McmcReg() = default;

protected:
	void updateState() override { reg_ldlt_diag(diag_vec, prior_sig_shp, prior_sig_scl, latent_innov * chol_lower.transpose(), rng); }
	void updateSv() override { sqrt_sv = diag_vec.cwiseSqrt().transpose().replicate(num_design, 1); }
	void updateCoefRecords() override {
		reg_record->assignRecords(mcmc_step, coef_vec, contem_coef, diag_vec);
		sparse_record.assignRecords(mcmc_step, num_alpha, dim, nrow_coef, sparse_coef, sparse_contem);
	}

private:
	Eigen::VectorXd diag_vec; // inverse of d_i
};

class McmcSv : public McmcTriangular {
public:
	McmcSv(const SvParams& params, const SvInits& inits, unsigned int seed)
	: McmcTriangular(params, inits, seed),
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
		varsv_h0(lvol_init, prior_init_mean, prior_init_prec, lvol_draw.row(0), lvol_sig, rng);
	}
	void updateSv() override { sqrt_sv = (lvol_draw / 2).array().exp(); }
	void updateCoefRecords() override {
		reg_record->assignRecords(mcmc_step, coef_vec, contem_coef, lvol_draw, lvol_sig, lvol_init);
		sparse_record.assignRecords(mcmc_step, num_alpha, dim, nrow_coef, sparse_coef, sparse_contem);
	}

private:
	Eigen::MatrixXd ortho_latent; // orthogonalized Z0
	Eigen::MatrixXd lvol_draw; // h_j = (h_j1, ..., h_jn)
	Eigen::VectorXd lvol_init;
	Eigen::VectorXd lvol_sig;
	Eigen::VectorXd prior_init_mean;
	Eigen::MatrixXd prior_init_prec;
};

template <typename BaseMcmc = McmcReg>
class McmcMinn : public BaseMcmc {
public:
	McmcMinn(
		const MinnParams<typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, RegParams, SvParams>::type>& params,
		const typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, LdltInits, SvInits>::type& inits,
		unsigned int seed
	)
	: BaseMcmc(params, inits, seed) {
		prior_alpha_mean.head(num_alpha) = params._prior_mean.reshaped();
		prior_alpha_prec.head(num_alpha) = kronecker_eigen(params._prec_diag, params._prior_prec).diagonal();
		if (include_mean) {
			prior_alpha_mean.tail(dim) = params._mean_non;
		}
	}
	virtual ~McmcMinn() = default;
	void appendRecords(LIST& list) override {}

protected:
	using BaseMcmc::dim;
	using BaseMcmc::num_alpha;
	using BaseMcmc::include_mean;
	using BaseMcmc::prior_alpha_mean;
	using BaseMcmc::prior_alpha_prec;
	using BaseMcmc::alpha_penalty;
	using BaseMcmc::updateCoefRecords;
	void updateCoefPrec() override {};
	void updatePenalty() override {};
	void updateImpactPrec() override {};
	void updateRecords() override { updateCoefRecords(); }
};

template <typename BaseMcmc = McmcReg>
class McmcHierminn : public BaseMcmc {
public:
	McmcHierminn(
		const HierminnParams<typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, RegParams, SvParams>::type>& params,
		const HierminnInits<typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, LdltInits, SvInits>::type>& inits,
		unsigned int seed
	)
	: BaseMcmc(params, inits, seed),
		own_id(params._own_id), cross_id(params._cross_id),
		coef_minnesota(params._minnesota), grp_mat(params._grp_mat), grp_vec(grp_mat.reshaped()),
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
	virtual ~McmcHierminn() = default;
	void appendRecords(LIST& list) override {}

protected:
	using BaseMcmc::dim;
	using BaseMcmc::num_alpha;
	using BaseMcmc::include_mean;
	using BaseMcmc::rng;
	using BaseMcmc::prior_alpha_mean;
	using BaseMcmc::prior_alpha_prec;
	using BaseMcmc::alpha_penalty;
	using BaseMcmc::prior_chol_mean;
	using BaseMcmc::prior_chol_prec;
	using BaseMcmc::coef_vec;
	using BaseMcmc::contem_coef;
	using BaseMcmc::updateCoefRecords;
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
		for (int i = 0; i < num_alpha; ++i) {
			if (own_id.find(grp_vec[i]) != own_id.end()) {
				prior_alpha_prec[i] /= own_lambda; // divide because it is precision
				alpha_penalty[i] = 0;
			}
			if (cross_id.find(grp_vec[i]) != cross_id.end()) {
				prior_alpha_prec[i] /= cross_lambda; // divide because it is precision
				alpha_penalty[i] = 1;
			}
		}
	}
	void updatePenalty() override {};
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

template <typename BaseMcmc = McmcReg>
class McmcSsvs : public BaseMcmc {
public:
	McmcSsvs(
		const SsvsParams<typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, RegParams, SvParams>::type>& params,
		const SsvsInits<typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, LdltInits, SvInits>::type>& inits,
		unsigned int seed
	)
	: BaseMcmc(params, inits, seed),
		own_id(params._own_id), grp_id(params._grp_id), grp_vec(params._grp_mat.reshaped()), num_grp(grp_id.size()),
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
	virtual ~McmcSsvs() = default;
	void appendRecords(LIST& list) override {
		list["gamma_record"] = ssvs_record.coef_dummy_record;
	}

protected:
	using BaseMcmc::num_iter;
	using BaseMcmc::num_alpha;
	using BaseMcmc::num_lowerchol;
	using BaseMcmc::mcmc_step;
	using BaseMcmc::rng;
	using BaseMcmc::prior_alpha_prec;
	using BaseMcmc::alpha_penalty;
	using BaseMcmc::prior_chol_prec;
	using BaseMcmc::coef_vec;
	using BaseMcmc::contem_coef;
	using BaseMcmc::updateCoefRecords;
	void updateCoefPrec() override {
		ssvs_local_slab(coef_slab, coef_dummy, coef_vec.head(num_alpha), ig_shape, ig_scl, spike_scl, rng);
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
		prior_alpha_prec.head(num_alpha).array() = 1 / (spike_scl * (1 - coef_dummy.array()) * coef_slab.array() + coef_dummy.array() * coef_slab.array());
	}
	void updatePenalty() override {
		for (int i = 0; i < num_alpha; ++i) {
			if (own_id.find(grp_vec[i]) != own_id.end()) {
				alpha_penalty[i] = 0;
			} else {
				alpha_penalty[i] = 1;
			}
		}
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
	std::set<int> own_id;
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

template <typename BaseMcmc = McmcReg, bool isGroup = true>
class McmcHorseshoe : public BaseMcmc {
public:
	McmcHorseshoe(
		const HorseshoeParams<typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, RegParams, SvParams>::type>& params,
		const HsInits<typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, LdltInits, SvInits>::type>& inits,
		unsigned int seed
	)
	: BaseMcmc(params, inits, seed),
		own_id(params._own_id), grp_id(params._grp_id), grp_vec(params._grp_mat.reshaped()), num_grp(grp_id.size()),
		hs_record(num_iter, num_alpha, num_grp),
		local_lev(inits._init_local), group_lev(inits._init_group), global_lev(isGroup ? inits._init_global : 1.0),
		shrink_fac(Eigen::VectorXd::Zero(num_alpha)),
		latent_local(Eigen::VectorXd::Zero(num_alpha)), latent_group(Eigen::VectorXd::Zero(num_grp)), latent_global(0.0),
		coef_var(Eigen::VectorXd::Zero(num_alpha)),
		contem_local_lev(inits._init_contem_local), contem_global_lev(inits._init_conetm_global),
		contem_var(Eigen::VectorXd::Zero(num_lowerchol)),
		latent_contem_local(Eigen::VectorXd::Zero(num_lowerchol)), latent_contem_global(Eigen::VectorXd::Zero(1)) {
		hs_record.assignRecords(0, shrink_fac, local_lev, group_lev, global_lev);
	}
	virtual ~McmcHorseshoe() = default;
	void appendRecords(LIST& list) override {
		list["lambda_record"] = hs_record.local_record;
		list["eta_record"] = hs_record.group_record;
		list["tau_record"] = hs_record.global_record;
		list["kappa_record"] = hs_record.shrink_record;
	}

protected:
	using BaseMcmc::num_iter;
	using BaseMcmc::num_alpha;
	using BaseMcmc::num_lowerchol;
	using BaseMcmc::mcmc_step;
	using BaseMcmc::rng;
	using BaseMcmc::prior_alpha_prec;
	using BaseMcmc::alpha_penalty;
	using BaseMcmc::prior_chol_prec;
	using BaseMcmc::coef_vec;
	using BaseMcmc::contem_coef;
	using BaseMcmc::updateCoefRecords;
	void updateCoefPrec() override {
		horseshoe_latent(latent_group, group_lev, rng);
		horseshoe_mn_sparsity(group_lev, grp_vec, grp_id, latent_group, global_lev, local_lev, coef_vec.head(num_alpha), 1, rng);
		for (int j = 0; j < num_grp; j++) {
			coef_var = (grp_vec.array() == grp_id[j]).select(
				group_lev[j],
				coef_var
			);
		}
		horseshoe_latent(latent_local, local_lev, rng);
		if constexpr (isGroup) {
			horseshoe_latent(latent_global, global_lev, rng);
			global_lev = horseshoe_global_sparsity(latent_global, coef_var.array() * local_lev.array(), coef_vec.head(num_alpha), 1, rng);
		}
		horseshoe_local_sparsity(local_lev, latent_local, coef_var, coef_vec.head(num_alpha), global_lev * global_lev, rng);
		prior_alpha_prec.head(num_alpha) = 1 / (global_lev * coef_var.array() * local_lev.array()).square();
		shrink_fac = 1 / (1 + prior_alpha_prec.head(num_alpha).array());
	}
	void updatePenalty() override {
		for (int i = 0; i < num_alpha; ++i) {
			if (own_id.find(grp_vec[i]) != own_id.end()) {
				alpha_penalty[i] = 0;
			} else {
				alpha_penalty[i] = 1;
			}
		}
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
	std::set<int> own_id;
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

template <typename BaseMcmc = McmcReg, bool isGroup = true>
class McmcNg : public BaseMcmc {
public:
	McmcNg(
		const NgParams<typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, RegParams, SvParams>::type>& params,
		const NgInits<typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, LdltInits, SvInits>::type>& inits,
		unsigned int seed
	)
	: BaseMcmc(params, inits, seed),
		own_id(params._own_id), grp_id(params._grp_id), grp_vec(params._grp_mat.reshaped()), num_grp(grp_id.size()),
		ng_record(num_iter, num_alpha, num_grp),
		mh_sd(params._mh_sd),
		local_shape(inits._init_local_shape), local_shape_fac(Eigen::VectorXd::Ones(num_alpha)),
		contem_shape(inits._init_contem_shape),
		group_shape(params._group_shape), group_scl(params._global_scl),
		global_shape(params._global_shape), global_scl(params._global_scl),
		contem_global_shape(params._contem_global_shape), contem_global_scl(params._contem_global_scl),
		local_lev(inits._init_local), group_lev(inits._init_group), global_lev(isGroup ? inits._init_global : 1.0),
		coef_var(Eigen::VectorXd::Zero(num_alpha)),
		contem_global_lev(inits._init_conetm_global),
		contem_fac(contem_global_lev[0] * inits._init_contem_local) {
		ng_record.assignRecords(0, local_lev, group_lev, global_lev);
	}
	virtual ~McmcNg() = default;
	void appendRecords(LIST& list) override {
		list["lambda_record"] = ng_record.local_record;
		list["eta_record"] = ng_record.group_record;
		list["tau_record"] = ng_record.global_record;
	}

protected:
	using BaseMcmc::num_iter;
	using BaseMcmc::num_alpha;
	using BaseMcmc::num_lowerchol;
	using BaseMcmc::mcmc_step;
	using BaseMcmc::rng;
	using BaseMcmc::prior_alpha_prec;
	using BaseMcmc::alpha_penalty;
	using BaseMcmc::prior_chol_prec;
	using BaseMcmc::coef_vec;
	using BaseMcmc::contem_coef;
	using BaseMcmc::updateCoefRecords;
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
		ng_local_sparsity(local_lev, local_shape_fac, coef_vec.head(num_alpha), global_lev * coef_var, rng);
		if constexpr (isGroup) {
			global_lev = ng_global_sparsity(local_lev.array() / coef_var.array(), local_shape_fac, global_shape, global_scl, rng);
		}
		ng_mn_sparsity(group_lev, grp_vec, grp_id, local_shape, global_lev, local_lev, group_shape, group_scl, rng);
		prior_alpha_prec.head(num_alpha) = 1 / local_lev.array().square();
	}
	void updatePenalty() override {
		for (int i = 0; i < num_alpha; ++i) {
			if (own_id.find(grp_vec[i]) != own_id.end()) {
				alpha_penalty[i] = 0;
			} else {
				alpha_penalty[i] = 1;
			}
		}
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
	std::set<int> own_id;
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

template <typename BaseMcmc = McmcReg, bool isGroup = true>
class McmcDl : public BaseMcmc {
public:
	McmcDl(
		const DlParams<typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, RegParams, SvParams>::type>& params,
		const GlInits<typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, LdltInits, SvInits>::type>& inits,
		unsigned int seed
	)
	: BaseMcmc(params, inits, seed),
		own_id(params._own_id), grp_id(params._grp_id), grp_vec(params._grp_mat.reshaped()), num_grp(grp_id.size()),
		dl_record(num_iter, num_alpha),
		dir_concen(0.0), contem_dir_concen(0.0),
		shape(params._shape), rate(params._rate),
		grid_size(params._grid_size),
		local_lev(inits._init_local), group_lev(Eigen::VectorXd::Zero(num_grp)), global_lev(isGroup ? inits._init_global : 1.0),
		latent_local(Eigen::VectorXd::Zero(num_alpha)),
		coef_var(Eigen::VectorXd::Zero(num_alpha)),
		contem_local_lev(inits._init_contem_local), contem_global_lev(inits._init_conetm_global),
		latent_contem_local(Eigen::VectorXd::Zero(num_lowerchol)) {
		dl_record.assignRecords(0, local_lev, global_lev);
	}
	virtual ~McmcDl() = default;
	void appendRecords(LIST& list) override {
		list["lambda_record"] = dl_record.local_record;
		list["tau_record"] = dl_record.global_record;
	}

protected:
	using BaseMcmc::num_iter;
	using BaseMcmc::num_alpha;
	using BaseMcmc::num_lowerchol;
	using BaseMcmc::mcmc_step;
	using BaseMcmc::rng;
	using BaseMcmc::prior_alpha_prec;
	using BaseMcmc::alpha_penalty;
	using BaseMcmc::prior_chol_prec;
	using BaseMcmc::coef_vec;
	using BaseMcmc::contem_coef;
	using BaseMcmc::updateCoefRecords;
	void updateCoefPrec() override {
		dl_mn_sparsity(group_lev, grp_vec, grp_id, global_lev, local_lev, shape, rate, coef_vec.head(num_alpha), rng);
		for (int j = 0; j < num_grp; j++) {
			coef_var = (grp_vec.array() == grp_id[j]).select(
				group_lev[j],
				coef_var
			);
		}
		dl_latent(latent_local, global_lev * local_lev.array() * coef_var.array(), coef_vec.head(num_alpha), rng);
		dl_dir_griddy(dir_concen, grid_size, local_lev, global_lev, rng);
		dl_local_sparsity(local_lev, dir_concen, coef_vec.head(num_alpha).array() / coef_var.array(), rng);
		if constexpr (isGroup) {
			global_lev = dl_global_sparsity(local_lev.array() * coef_var.array(), dir_concen, coef_vec.head(num_alpha), rng);
		}
		prior_alpha_prec.head(num_alpha) = 1 / ((global_lev * local_lev.array() * coef_var.array()).square() * latent_local.array());
	}
	void updatePenalty() override {
		for (int i = 0; i < num_alpha; ++i) {
			if (own_id.find(grp_vec[i]) != own_id.end()) {
				alpha_penalty[i] = 0;
			} else {
				alpha_penalty[i] = 1;
			}
		}
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
	std::set<int> own_id;
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

template <typename BaseMcmc = McmcReg, bool isGroup = true>
inline std::vector<std::unique_ptr<BaseMcmc>> initialize_mcmc(
	int num_chains, int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
	LIST& param_reg, LIST& param_prior, LIST& param_intercept, LIST_OF_LIST& param_init, int prior_type,
  const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
  bool include_mean, Eigen::Ref<const Eigen::VectorXi> seed_chain, Optional<int> num_design = NULLOPT
) {
	using PARAMS = typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, RegParams, SvParams>::type;
	using INITS = typename std::conditional<std::is_same<BaseMcmc, McmcReg>::value, LdltInits, SvInits>::type;
	std::vector<std::unique_ptr<BaseMcmc>> mcmc_ptr(num_chains);
	switch (prior_type) {
		case 1: {
			MinnParams<PARAMS> minn_params(
				num_iter, x, y,
				param_reg,
				own_id, cross_id,
				param_prior,
				param_intercept, include_mean
			);
			for (int i = 0; i < num_chains; ++i) {
				LIST init_spec = param_init[i];
				// INITS ldlt_inits(init_spec);
				INITS ldlt_inits = num_design ? INITS(init_spec, *num_design) : INITS(init_spec);
				mcmc_ptr[i] = std::make_unique<McmcMinn<BaseMcmc>>(minn_params, ldlt_inits, static_cast<unsigned int>(seed_chain[i]));
			}
			return mcmc_ptr;
		}
		case 2: {
			SsvsParams<PARAMS> ssvs_params(
				num_iter, x, y,
				param_reg,
				own_id, cross_id,
				grp_id, grp_mat,
				param_prior,
				param_intercept,
				include_mean
			);
			for (int i = 0; i < num_chains; ++i) {
				LIST init_spec = param_init[i];
				// SsvsInits<INITS> ssvs_inits(init_spec);
				SsvsInits<INITS> ssvs_inits = num_design ? SsvsInits<INITS>(init_spec, *num_design) : SsvsInits<INITS>(init_spec);
				mcmc_ptr[i] = std::make_unique<McmcSsvs<BaseMcmc>>(ssvs_params, ssvs_inits, static_cast<unsigned int>(seed_chain[i]));
			}
			return mcmc_ptr;
		}
		case 3: {
			HorseshoeParams<PARAMS> hs_params(
				num_iter, x, y,
				param_reg,
				own_id, cross_id,
				grp_id, grp_mat,
				param_intercept, include_mean
			);
			for (int i = 0; i < num_chains; ++i) {
				LIST init_spec = param_init[i];
				// HsInits<INITS> hs_inits(init_spec);
				HsInits<INITS> hs_inits = num_design ? HsInits<INITS>(init_spec, *num_design) : HsInits<INITS>(init_spec);
				mcmc_ptr[i] = std::make_unique<McmcHorseshoe<BaseMcmc, isGroup>>(hs_params, hs_inits, static_cast<unsigned int>(seed_chain[i]));
			}
			return mcmc_ptr;
		}
		case 4: {
			HierminnParams<PARAMS> minn_params(
				num_iter, x, y,
				param_reg,
				own_id, cross_id, grp_mat,
				param_prior,
				param_intercept, include_mean
			);
			for (int i = 0; i < num_chains; ++i) {
				LIST init_spec = param_init[i];
				// HierminnInits<INITS> minn_inits(init_spec);
				HierminnInits<INITS> minn_inits = num_design ? HierminnInits<INITS>(init_spec, *num_design) : HierminnInits<INITS>(init_spec);
				mcmc_ptr[i] = std::make_unique<McmcHierminn<BaseMcmc>>(minn_params, minn_inits, static_cast<unsigned int>(seed_chain[i]));
			}
			return mcmc_ptr;
		}
		case 5: {
			NgParams<PARAMS> ng_params(
				num_iter, x, y,
				param_reg,
				own_id, cross_id,
				grp_id, grp_mat,
				param_prior,
				param_intercept,
				include_mean
			);
			for (int i = 0; i < num_chains; ++i) {
				LIST init_spec = param_init[i];
				// NgInits<INITS> ng_inits(init_spec);
				NgInits<INITS> ng_inits = num_design ? NgInits<INITS>(init_spec, *num_design) : NgInits<INITS>(init_spec);
				mcmc_ptr[i] = std::make_unique<McmcNg<BaseMcmc, isGroup>>(ng_params, ng_inits, static_cast<unsigned int>(seed_chain[i]));
			}
			return mcmc_ptr;
		}
		case 6: {
			DlParams<PARAMS> dl_params(
				num_iter, x, y,
				param_reg,
				own_id, cross_id,
				grp_id, grp_mat,
				param_prior,
				param_intercept,
				include_mean
			);
			for (int i = 0; i < num_chains; ++i) {
				LIST init_spec = param_init[i];
				// GlInits<INITS> dl_inits(init_spec);
				GlInits<INITS> dl_inits = num_design ? GlInits<INITS>(init_spec, *num_design) : GlInits<INITS>(init_spec);
				mcmc_ptr[i] = std::make_unique<McmcDl<BaseMcmc, isGroup>>(dl_params, dl_inits, static_cast<unsigned int>(seed_chain[i]));
			}
			return mcmc_ptr;
		}
	}
	return mcmc_ptr;
}

class McmcInterface {
public:
	virtual ~McmcInterface() = default;
	virtual LIST_OF_LIST returnRecords() = 0;
};

template <typename BaseMcmc = McmcReg, bool isGroup = true>
class McmcRun : public McmcInterface {
public:
	McmcRun(
		int num_chains, int num_iter, int num_burn, int thin,
    const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		LIST& param_cov, LIST& param_prior, LIST& param_intercept,
		LIST_OF_LIST& param_init, int prior_type,
    const Eigen::VectorXi& grp_id, const Eigen::VectorXi& own_id, const Eigen::VectorXi& cross_id, const Eigen::MatrixXi& grp_mat,
    bool include_mean, const Eigen::VectorXi& seed_chain, bool display_progress, int nthreads
	)
	: num_chains(num_chains), num_iter(num_iter), num_burn(num_burn), thin(thin), nthreads(nthreads),
		display_progress(display_progress), mcmc_ptr(num_chains), res(num_chains) {
		mcmc_ptr = initialize_mcmc<BaseMcmc, isGroup>(
			num_chains, num_iter - num_burn, x, y,
			param_cov, param_prior, param_intercept, param_init, prior_type,
			grp_id, own_id, cross_id, grp_mat,
			include_mean, seed_chain
		);
	}
	virtual ~McmcRun() = default;
	void fit() {
		if (num_chains == 1) {
			runGibbs(0);
		} else {
		#ifdef _OPENMP
			#pragma omp parallel for num_threads(nthreads)
		#endif
			for (int chain = 0; chain < num_chains; chain++) {
				runGibbs(chain);
			}
		}
	}
	LIST_OF_LIST returnRecords() override {
		fit();
		return WRAP(res);
	}

protected:
	void runGibbs(int chain) {
		bvharprogress bar(num_iter, display_progress);
		bvharinterrupt();
		for (int i = 0; i < num_burn; ++i) {
			bar.increment();
			mcmc_ptr[chain]->doWarmUp();
			bar.update();
		}
		for (int i = num_burn; i < num_iter; ++i) {
			if (bvharinterrupt::is_interrupted()) {
			#ifdef _OPENMP
				#pragma omp critical
			#endif
				{
					res[chain] = mcmc_ptr[chain]->returnRecords(0, 1);
				}
				break;
			}
			bar.increment();
			mcmc_ptr[chain]->doPosteriorDraws();
			bar.update();
		}
	#ifdef _OPENMP
		#pragma omp critical
	#endif
		{
			res[chain] = mcmc_ptr[chain]->returnRecords(0, thin);
		}
	}

private:
	std::set<int> own_id;
	int num_chains;
	int num_iter;
	int num_burn;
	int thin;
	int nthreads;
	bool display_progress;
	std::vector<std::unique_ptr<BaseMcmc>> mcmc_ptr;
	std::vector<LIST> res;
};

} // namespace bvhar

#endif // BVHARMCMC_H