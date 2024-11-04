#ifndef BVHARMCMC_H
#define BVHARMCMC_H

#include "bvharconfig.h"
#include "bvharprogress.h"

namespace bvhar {

class McmcTriangular;
class McmcReg;
class McmcSv;

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
	void doPosteriorDraws() {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefPrec();
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
	virtual void updateImpactPrec() = 0;
	virtual void updateRecords() = 0;
	void updateCoef() {
		for (int j = 0; j < dim; j++) {
			coef_mat.col(j).setZero(); // j-th column of A = 0: A(-j) = (alpha_1, ..., alpha_(j-1), 0, alpha_(j), ..., alpha_k)
			Eigen::MatrixXd chol_lower_j = chol_lower.bottomRows(dim - j); // L_(j:k) = a_jt to a_kt for t = 1, ..., j - 1
			Eigen::MatrixXd sqrt_sv_j = sqrt_sv.rightCols(dim - j); // use h_jt to h_kt for t = 1, .. n => (k - j + 1) x k
			Eigen::MatrixXd design_coef = kronecker_eigen(chol_lower_j.col(j), x).array().colwise() / sqrt_sv_j.reshaped().array(); // L_(j:k, j) otimes X0 scaled by D_(1:n, j:k): n(k - j + 1) x kp
			Eigen::VectorXd prior_mean_j(dim_design);
			Eigen::VectorXd prior_prec_j(dim_design);
			if (include_mean) {
				prior_mean_j << prior_alpha_mean.segment(j * nrow_coef, nrow_coef), prior_alpha_mean.tail(dim)[j];
				prior_prec_j << prior_alpha_prec.segment(j * nrow_coef, nrow_coef), prior_alpha_prec.tail(dim)[j];
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
				draw_coef(
					coef_mat.col(j),
					design_coef,
					(((y - x * coef_mat) * chol_lower_j.transpose()).array() / sqrt_sv_j.array()).reshaped(),
					prior_mean_j, prior_prec_j, rng
				);
				coef_vec = coef_mat.reshaped();
			}
			draw_mn_savs(sparse_coef.col(j), coef_mat.col(j), design_coef, prior_prec_j);
		}
	}
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
		sparse_record.assignRecords(mcmc_step, sparse_coef, sparse_contem);
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
	void updateSv() override { sqrt_sv = (-lvol_draw / 2).array().exp(); }
	void updateCoefRecords() override {
		reg_record->assignRecords(mcmc_step, coef_vec, contem_coef, lvol_draw, lvol_sig, lvol_init);
		sparse_record.assignRecords(mcmc_step, sparse_coef, sparse_contem);
	}

private:
	Eigen::MatrixXd ortho_latent; // orthogonalized Z0
	Eigen::MatrixXd lvol_draw; // h_j = (h_j1, ..., h_jn)
	Eigen::VectorXd lvol_init;
	Eigen::VectorXd lvol_sig;
	Eigen::VectorXd prior_init_mean;
	Eigen::MatrixXd prior_init_prec;
};

} // namespace bvhar

#endif // BVHARMCMC_H