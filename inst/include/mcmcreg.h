#ifndef MCMCREG_H
#define MCMCREG_H

#include "ldltconfig.h"
#include "bvharprogress.h"

namespace bvhar {

class McmcReg;
class MinnReg;
class HierminnReg;
class SsvsReg;
class HorseshoeReg;
class NgReg;
class DlReg;

class McmcReg {
public:
	McmcReg(const RegParams& params, const LdltInits& inits, unsigned int seed)
	: include_mean(params._mean),
		x(params._x), y(params._y),
		num_iter(params._iter), dim(params._dim), dim_design(params._dim_design), num_design(params._num_design),
		num_lowerchol(params._num_lowerchol), num_coef(params._num_coef),
		num_alpha(params._num_alpha),
		reg_record(num_iter, dim, num_design, num_coef, num_lowerchol),
		sparse_record(num_iter, dim, num_design, num_alpha, num_lowerchol),
		mcmc_step(0), rng(seed),
		coef_vec(Eigen::VectorXd::Zero(num_coef)),
		contem_coef(inits._contem), diag_vec(inits._diag),
		prior_alpha_mean(Eigen::VectorXd::Zero(num_coef)),
		prior_alpha_prec(Eigen::VectorXd::Zero(num_coef)),
		prior_chol_mean(Eigen::VectorXd::Zero(num_lowerchol)),
		prior_chol_prec(Eigen::VectorXd::Ones(num_lowerchol)),
		coef_mat(inits._coef),
		contem_id(0),
		chol_lower(build_inv_lower(dim, contem_coef)),
		latent_innov(y - x * coef_mat),
		// ortho_latent(Eigen::MatrixXd::Zero(num_design, dim)),
		response_contem(Eigen::VectorXd::Zero(num_design)),
		// response_contem(Eigen::MatrixXd::Zero(num_design, dim)),
		sqrt_sv(Eigen::MatrixXd::Zero(num_design, dim)),
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
		reg_record.assignRecords(0, coef_vec, contem_coef, diag_vec);
		sparse_record.assignRecords(0, sparse_coef, sparse_contem);
	}
	virtual ~McmcReg() = default;
	virtual void doPosteriorDraws() = 0;
// #ifdef USE_RCPP
// 	virtual Rcpp::List returnRecords(int num_burn, int thin) const = 0;
// #else
// 	virtual py::dict returnRecords(int num_burn, int thin) const = 0;
// #endif
	virtual LIST returnRecords(int num_burn, int thin) const = 0;
	LdltRecords returnLdltRecords(int num_burn, int thin, bool sparse = false) const {
		if (sparse) {
			Eigen::MatrixXd coef_record(num_iter + 1, num_coef);
			if (include_mean) {
				coef_record << sparse_record.coef_record, reg_record.coef_record.rightCols(dim);
			} else {
				coef_record = sparse_record.coef_record;
			}
			return LdltRecords(
				thin_record(coef_record, num_iter, num_burn, thin).derived(),
				thin_record(sparse_record.contem_coef_record, num_iter, num_burn, thin).derived(),
				thin_record(reg_record.fac_record, num_iter, num_burn, thin).derived()
			);
		}
		LdltRecords res_record(
			thin_record(reg_record.coef_record, num_iter, num_burn, thin).derived(),
			thin_record(reg_record.contem_coef_record, num_iter, num_burn, thin).derived(),
			thin_record(reg_record.fac_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
	// virtual SsvsRecords returnSsvsRecords(int num_burn, int thin) const = 0;
	// virtual HorseshoeRecords returnHsRecords(int num_burn, int thin) const = 0;
	// virtual NgRecords returnNgRecords(int num_burn, int thin) const = 0;
	// virtual GlobalLocalRecords returnGlRecords(int num_burn, int thin) const = 0;

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
	LdltRecords reg_record;
	SparseRecords sparse_record;
	std::atomic<int> mcmc_step; // MCMC step
	boost::random::mt19937 rng; // RNG instance for multi-chain
	Eigen::VectorXd coef_vec;
	Eigen::VectorXd contem_coef;
	Eigen::VectorXd diag_vec; // inverse of d_i
	Eigen::VectorXd prior_alpha_mean; // prior mean vector of alpha
	Eigen::VectorXd prior_alpha_prec; // Diagonal of alpha prior precision
	Eigen::VectorXd prior_chol_mean; // prior mean vector of a = 0
	Eigen::VectorXd prior_chol_prec; // Diagonal of prior precision of a = I
	Eigen::MatrixXd coef_mat;
	int contem_id;
	Eigen::MatrixXd chol_lower; // L in Sig_t^(-1) = L D_t^(-1) LT
	Eigen::MatrixXd latent_innov; // Z0 = Y0 - X0 A = (eps_p+1, eps_p+2, ..., eps_n+p)^T
  // Eigen::MatrixXd ortho_latent; // orthogonalized Z0
	Eigen::VectorXd response_contem; // j-th column of Z0 = Y0 - X0 * A: n-dim
	// Eigen::MatrixXd response_contem; // j-th column of Z0 = Y0 - X0 * A: n-dim
	Eigen::MatrixXd sqrt_sv; // stack sqrt of exp(h_t) = (exp(-h_1t / 2), ..., exp(-h_kt / 2)), t = 1, ..., n => n x k
	Eigen::MatrixXd sparse_coef;
	Eigen::VectorXd sparse_contem;
	void updateCoef() {
		for (int j = 0; j < dim; j++) {
			coef_mat.col(j).setZero(); // j-th column of A = 0: A(-j) = (alpha_1, ..., alpha_(j-1), 0, alpha_(j), ..., alpha_k)
			Eigen::MatrixXd chol_lower_j = chol_lower.bottomRows(dim - j); // L_(j:k) = a_jt to a_kt for t = 1, ..., j - 1
			Eigen::MatrixXd sqrt_sv_j = sqrt_sv.rightCols(dim - j); // use h_jt to h_kt for t = 1, .. n => (k - j + 1) x k
			Eigen::MatrixXd design_coef = kronecker_eigen(chol_lower_j.col(j), x).array().colwise() / sqrt_sv_j.reshaped().array(); // L_(j:k, j) otimes X0 scaled by D_(1:n, j:k): n(k - j + 1) x kp
			// Eigen::VectorXd response_j = (((y - x * coef_mat) * chol_lower_j.transpose()).array() / sqrt_sv_j.array()).reshaped(); // Hadamard product between: (Y - X0 A(-j))L_(j:k)^T and D_(1:n, j:k)
			if (include_mean) {
				Eigen::VectorXd prior_mean_j(dim_design);
				Eigen::VectorXd prior_prec_j(dim_design);
				prior_mean_j << prior_alpha_mean.segment(j * num_alpha / dim, num_alpha / dim), prior_alpha_mean.tail(dim)[j];
				prior_prec_j << prior_alpha_prec.segment(j * num_alpha / dim, num_alpha / dim), prior_alpha_prec.tail(dim)[j];
				draw_coef(
					coef_mat.col(j), design_coef,
					(((y - x * coef_mat) * chol_lower_j.transpose()).array() / sqrt_sv_j.array()).reshaped(),
					prior_mean_j, prior_prec_j, rng
				);
				coef_vec.head(num_alpha) = coef_mat.topRows(num_alpha / dim).reshaped();
				coef_vec.tail(dim) = coef_mat.bottomRows(1).transpose();
			} else {
				draw_coef(
					coef_mat.col(j),
					design_coef,
					(((y - x * coef_mat) * chol_lower_j.transpose()).array() / sqrt_sv_j.array()).reshaped(),
					prior_alpha_mean.segment(dim_design * j, dim_design), // Prior mean vector of j-th column of A
					prior_alpha_prec.segment(dim_design * j, dim_design), // Prior precision of j-th column of A
					rng
				);
				coef_vec.head(num_alpha) = coef_mat.topRows(num_alpha / dim).reshaped();
			}
			draw_savs(sparse_coef.col(j), coef_mat.col(j).head(num_alpha / dim), design_coef);
		}
	}
	void updateDiag() {
		// ortho_latent = latent_innov * chol_lower.transpose(); // L eps_t <=> Z0 U
		// reg_ldlt_diag(diag_vec, prior_sig_shp, prior_sig_scl, ortho_latent, rng);
		reg_ldlt_diag(diag_vec, prior_sig_shp, prior_sig_scl, latent_innov * chol_lower.transpose(), rng);
	}
	void updateImpact() {
		// response_contem.array() = latent_innov.array() / sqrt_sv.array();
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
	virtual void updateCoefPrec() = 0;
	virtual void updateCoefShrink() = 0;
	virtual void updateImpactPrec() = 0;
	virtual void updateRecords() = 0;
	void updateCoefRecords() {
		reg_record.assignRecords(mcmc_step, coef_vec, contem_coef, diag_vec);
		sparse_record.assignRecords(mcmc_step, sparse_coef, sparse_contem);
	}

private:
	Eigen::VectorXd prior_sig_shp;
	Eigen::VectorXd prior_sig_scl;
};

class MinnReg : public McmcReg {
public:
	MinnReg(const MinnParams& params, const LdltInits& inits, unsigned int seed) : McmcReg(params, inits, seed) {
		prior_alpha_mean.head(num_alpha) = params._prior_mean.reshaped();
		prior_alpha_prec.head(num_alpha) = kronecker_eigen(params._prec_diag, params._prior_prec).diagonal();
		if (include_mean) {
			prior_alpha_mean.tail(dim) = params._mean_non;
		}
	}
	virtual ~MinnReg() = default;
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		sqrt_sv = diag_vec.cwiseSqrt().transpose().replicate(num_design, 1);
		updateCoef();
		latent_innov = y - x * coef_mat; // E_t before a
		updateImpact();
		chol_lower = build_inv_lower(dim, contem_coef); // L before h_t
		updateDiag();
		updateRecords();
	}
	LIST returnRecords(int num_burn, int thin) const override {
		LIST res = CREATE_LIST(
			NAMED("alpha_record") = reg_record.coef_record.leftCols(num_alpha),
			NAMED("a_record") = reg_record.contem_coef_record,
			NAMED("d_record") = reg_record.fac_record,
			NAMED("alpha_sparse_record") = sparse_record.coef_record,
			NAMED("a_sparse_record") = sparse_record.contem_coef_record
		);
		if (include_mean) {
			// res["c_record"] = reg_record.coef_record.rightCols(dim);
			res["c_record"] = CAST_MATRIX(reg_record.coef_record.rightCols(dim));
		}
		for (auto& record : res) {
			ACCESS_LIST(record, res) = thin_record(CAST<Eigen::MatrixXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
		}
		return res;
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
	// GlobalLocalRecords returnGlRecords(int num_burn, int thin) const override {
	// 	return GlobalLocalRecords();
	// }

protected:
	void updateCoefPrec() override {};
	void updateCoefShrink() override {};
	void updateImpactPrec() override {};
	void updateRecords() override { updateCoefRecords(); }
};

class HierminnReg : public McmcReg {
public:
	HierminnReg(const HierminnParams& params, const HierminnInits& inits, unsigned int seed)
	: McmcReg(params, inits, seed),
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
	virtual ~HierminnReg() = default;
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		// updateCoefShrink();
		updateCoefPrec();
		sqrt_sv = diag_vec.cwiseSqrt().transpose().replicate(num_design, 1);
		updateCoef();
		updateImpactPrec();
		latent_innov = y - x * coef_mat; // E_t before a
		updateImpact();
		chol_lower = build_inv_lower(dim, contem_coef); // L before h_t
		updateDiag();
		updateRecords();
	}
	LIST returnRecords(int num_burn, int thin) const override {
		LIST res = CREATE_LIST(
			NAMED("alpha_record") = reg_record.coef_record.leftCols(num_alpha),
			NAMED("a_record") = reg_record.contem_coef_record,
			NAMED("d_record") = reg_record.fac_record,
			NAMED("alpha_sparse_record") = sparse_record.coef_record,
			NAMED("a_sparse_record") = sparse_record.contem_coef_record
		);
		if (include_mean) {
			// res["c_record"] = reg_record.coef_record.rightCols(dim);
			res["c_record"] = CAST_MATRIX(reg_record.coef_record.rightCols(dim));
		}
		for (auto& record : res) {
			ACCESS_LIST(record, res) = thin_record(CAST<Eigen::MatrixXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
		}
		return res;
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
	// GlobalLocalRecords returnGlRecords(int num_burn, int thin) const override {
	// 	return GlobalLocalRecords();
	// }

protected:
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

class SsvsReg : public McmcReg {
public:
	SsvsReg(const SsvsParams& params, const SsvsInits& inits, unsigned int seed)
	: McmcReg(params, inits, seed),
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
	virtual ~SsvsReg() = default;
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefPrec();
		sqrt_sv = diag_vec.cwiseSqrt().transpose().replicate(num_design, 1);
		updateCoef();
		updateCoefShrink();
		updateImpactPrec();
		latent_innov = y - x * coef_mat; // E_t before a
		updateImpact();
		chol_lower = build_inv_lower(dim, contem_coef); // L before d_i
		updateDiag();
		updateRecords();
	}
	LIST returnRecords(int num_burn, int thin) const override {
		LIST res = CREATE_LIST(
			NAMED("alpha_record") = reg_record.coef_record.leftCols(num_alpha),
			NAMED("a_record") = reg_record.contem_coef_record,
			NAMED("d_record") = reg_record.fac_record,
			NAMED("gamma_record") = ssvs_record.coef_dummy_record,
			NAMED("alpha_sparse_record") = sparse_record.coef_record,
			NAMED("a_sparse_record") = sparse_record.contem_coef_record
		);
		if (include_mean) {
			// res["c_record"] = reg_record.coef_record.rightCols(dim);
			res["c_record"] = CAST_MATRIX(reg_record.coef_record.rightCols(dim));
		}
		for (auto& record : res) {
			ACCESS_LIST(record, res) = thin_record(CAST<Eigen::MatrixXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
		}
		return res;
	}

protected:
	void updateCoefPrec() override {
		// coef_mixture_mat = build_ssvs_sd(coef_spike, coef_slab, coef_dummy);
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
	// Eigen::VectorXd coef_spike; // remove later? coef_spike = spike_scl * coef_slab
	Eigen::VectorXd coef_slab;
	double spike_scl, contem_spike_scl; // scaling factor between 0 and 1: spike_sd = c * slab_sd
	double ig_shape, ig_scl, contem_ig_shape, contem_ig_scl; // IG hyperparameter for spike sd
	// Eigen::VectorXd contem_spike; // remove later? contem_spike = contem_spike_scl * contem_slab
	Eigen::VectorXd contem_slab;
	Eigen::VectorXd coef_s1, coef_s2;
	double contem_s1, contem_s2;
	Eigen::VectorXd slab_weight; // pij vector
};

class HorseshoeReg : public McmcReg {
public:
	HorseshoeReg(const HorseshoeParams& params, const HsInits& inits, unsigned int seed)
	: McmcReg(params, inits, seed),
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
	virtual ~HorseshoeReg() = default;
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefPrec();
		sqrt_sv = diag_vec.cwiseSqrt().transpose().replicate(num_design, 1);
		updateCoef();
		updateCoefShrink();
		updateImpactPrec();
		latent_innov = y - x * coef_mat; // E_t before a
		updateImpact();
		chol_lower = build_inv_lower(dim, contem_coef); // L before d_i
		updateDiag();
		updateRecords();
	}
	LIST returnRecords(int num_burn, int thin) const override {
		LIST res = CREATE_LIST(
			NAMED("alpha_record") = reg_record.coef_record.leftCols(num_alpha),
			NAMED("a_record") = reg_record.contem_coef_record,
			NAMED("d_record") = reg_record.fac_record,
			NAMED("lambda_record") = hs_record.local_record,
			NAMED("eta_record") = hs_record.group_record,
			NAMED("tau_record") = hs_record.global_record,
			NAMED("kappa_record") = hs_record.shrink_record,
			NAMED("alpha_sparse_record") = sparse_record.coef_record,
			NAMED("a_sparse_record") = sparse_record.contem_coef_record
		);
		if (include_mean) {
			// res["c_record"] = reg_record.coef_record.rightCols(dim);
			res["c_record"] = CAST_MATRIX(reg_record.coef_record.rightCols(dim));
		}
		for (auto& record : res) {
			if (IS_MATRIX(ACCESS_LIST(record, res))) {
				ACCESS_LIST(record, res) = thin_record(CAST<Eigen::MatrixXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			} else {
				ACCESS_LIST(record, res) = thin_record(CAST<Eigen::VectorXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			}
		}
		return res;
	}

protected:
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

class NgReg : public McmcReg {
public:
	NgReg(const NgParams& params, const NgInits& inits, unsigned int seed)
	: McmcReg(params, inits, seed),
		grp_id(params._grp_id), grp_vec(params._grp_mat.reshaped()), num_grp(grp_id.size()),
		ng_record(num_iter, num_alpha, num_grp),
		mh_sd(params._mh_sd),
		local_shape(inits._init_local_shape), local_shape_fac(Eigen::VectorXd::Ones(num_alpha)),
		contem_shape(inits._init_contem_shape),
		group_shape(params._group_shape), group_scl(params._global_scl),
		global_shape(params._global_shape), global_scl(params._global_scl),
		contem_global_shape(params._contem_global_shape), contem_global_scl(params._contem_global_scl),
		local_lev(inits._init_local), group_lev(inits._init_group), global_lev(inits._init_global),
		// local_fac(Eigen::VectorXd::Zero(num_alpha)),
		coef_var(Eigen::VectorXd::Zero(num_alpha)),
		// contem_local_lev(inits._init_contem_local),
		contem_global_lev(inits._init_conetm_global),
		// contem_var(Eigen::VectorXd::Zero(num_lowerchol)),
		contem_fac(contem_global_lev[0] * inits._init_contem_local) {
		ng_record.assignRecords(0, local_lev, group_lev, global_lev);
	}
	virtual ~NgReg() = default;
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefPrec();
		sqrt_sv = diag_vec.cwiseSqrt().transpose().replicate(num_design, 1);
		updateCoef();
		// updateCoefShrink();
		updateImpactPrec();
		latent_innov = y - x * coef_mat; // E_t before a
		updateImpact();
		chol_lower = build_inv_lower(dim, contem_coef); // L before d_i
		updateDiag();
		updateRecords();
	}
	LIST returnRecords(int num_burn, int thin) const override {
		LIST res = CREATE_LIST(
			NAMED("alpha_record") = reg_record.coef_record.leftCols(num_alpha),
			NAMED("a_record") = reg_record.contem_coef_record,
			NAMED("d_record") = reg_record.fac_record,
			NAMED("lambda_record") = ng_record.local_record,
			NAMED("eta_record") = ng_record.group_record,
			NAMED("tau_record") = ng_record.global_record,
			NAMED("alpha_sparse_record") = sparse_record.coef_record,
			NAMED("a_sparse_record") = sparse_record.contem_coef_record
		);
		if (include_mean) {
			// res["c_record"] = reg_record.coef_record.rightCols(dim);
			res["c_record"] = CAST_MATRIX(reg_record.coef_record.rightCols(dim));
		}
		for (auto& record : res) {
			if (IS_MATRIX(ACCESS_LIST(record, res))) {
				ACCESS_LIST(record, res) = thin_record(CAST<Eigen::MatrixXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			} else {
				ACCESS_LIST(record, res) = thin_record(CAST<Eigen::VectorXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			}
		}
		return res;
	}

protected:
	void updateCoefPrec() override {
		ng_mn_shape_jump(local_shape, local_lev, group_lev, grp_vec, grp_id, global_lev, mh_sd, rng);
		// ng_mn_shape_jump(local_shape, local_fac, group_lev, grp_vec, grp_id, global_lev, mh_sd, rng);
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
		// local_fac.array() = global_lev * coef_var.array() * local_lev.array(); // tilde_lambda
		// prior_alpha_prec.topLeftCorner(num_alpha, num_alpha).diagonal() = 1 / local_fac.array().square();
		updateCoefShrink();
		prior_alpha_prec.head(num_alpha) = 1 / local_lev.array().square();
	}
	void updateCoefShrink() override {
		ng_local_sparsity(local_lev, local_shape_fac, coef_vec.head(num_alpha), global_lev * coef_var, rng);
		global_lev = ng_global_sparsity(local_lev.array() / coef_var.array(), local_shape_fac, global_shape, global_scl, rng);
		ng_mn_sparsity(group_lev, grp_vec, grp_id, local_shape, global_lev, local_lev, group_shape, group_scl, rng);
	}
	void updateImpactPrec() override {
		// contem_var = contem_global_lev.replicate(1, num_lowerchol).reshaped();
		// contem_fac = contem_global_lev[0] * contem_local_lev;
		contem_shape = ng_shape_jump(contem_shape, contem_fac, contem_global_lev[0], mh_sd, rng);
		// ng_local_sparsity(contem_fac, contem_shape, contem_coef, contem_var, rng);
		ng_local_sparsity(contem_fac, contem_shape, contem_coef, contem_global_lev.replicate(1, num_lowerchol).reshaped(), rng);
		// contem_local_lev = contem_fac / contem_global_lev[0];
		contem_global_lev[0] = ng_global_sparsity(contem_fac, contem_shape, contem_global_shape, contem_global_scl, rng);
		// contem_fac = contem_global_lev[0] * contem_local_lev;
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
	// Eigen::VectorXd local_fac;
	Eigen::VectorXd coef_var;
	// Eigen::VectorXd contem_local_lev;
	Eigen::VectorXd contem_global_lev;
	// Eigen::VectorXd contem_var;
	Eigen::VectorXd contem_fac;
};

class DlReg : public McmcReg {
public:
	DlReg(const DlParams& params, const GlInits& inits, unsigned int seed)
	: McmcReg(params, inits, seed),
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
	virtual ~DlReg() = default;
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefPrec();
		sqrt_sv = diag_vec.cwiseSqrt().transpose().replicate(num_design, 1);
		updateCoef();
		// updateCoefShrink();
		updateImpactPrec();
		latent_innov = y - x * coef_mat; // E_t before a
		updateImpact();
		chol_lower = build_inv_lower(dim, contem_coef); // L before d_i
		updateDiag();
		updateRecords();
	}
	LIST returnRecords(int num_burn, int thin) const override {
		LIST res = CREATE_LIST(
			NAMED("alpha_record") = reg_record.coef_record.leftCols(num_alpha),
			NAMED("a_record") = reg_record.contem_coef_record,
			NAMED("d_record") = reg_record.fac_record,
			NAMED("lambda_record") = dl_record.local_record,
			NAMED("tau_record") = dl_record.global_record,
			NAMED("alpha_sparse_record") = sparse_record.coef_record,
			NAMED("a_sparse_record") = sparse_record.contem_coef_record
		);
		if (include_mean) {
			// res["c_record"] = reg_record.coef_record.rightCols(dim);
			res["c_record"] = CAST_MATRIX(reg_record.coef_record.rightCols(dim));
		}
		for (auto& record : res) {
			if (IS_MATRIX(ACCESS_LIST(record, res))) {
				ACCESS_LIST(record, res) = thin_record(CAST<Eigen::MatrixXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			} else {
				ACCESS_LIST(record, res) = thin_record(CAST<Eigen::VectorXd>(ACCESS_LIST(record, res)), num_iter, num_burn, thin);
			}
		}
		return res;
	}

protected:
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

#endif // MCMCREG_H