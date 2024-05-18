#ifndef MCMCREG_H
#define MCMCREG_H

#include "bvhardraw.h"
#include "bvharprogress.h"

namespace bvhar {

struct HsParams : public RegParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXi _grp_mat;

	HsParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& hs_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
		Rcpp::List& intercept, bool include_mean
	)
	: RegParams(num_iter, x, y, hs_spec, intercept, include_mean), _grp_id(grp_id), _grp_mat(grp_mat) {}
};

struct LdltInits : public RegInits {
	Eigen::VectorXd _diag;
	
	LdltInits(Rcpp::List& init)
	: RegInits(init),
		_diag(Rcpp::as<Eigen::VectorXd>(init["init_diag"])) {}
};

struct HsInits : public LdltInits {
	Eigen::VectorXd _init_local;
	Eigen::VectorXd _init_group;
	double _init_global;
	Eigen::VectorXd _init_contem_local;
	Eigen::VectorXd _init_conetm_global;
	
	HsInits(Rcpp::List& init)
	: LdltInits(init),
		_init_local(Rcpp::as<Eigen::VectorXd>(init["local_sparsity"])),
		_init_group(Rcpp::as<Eigen::VectorXd>(init["group_sparsity"])),
		_init_global(init["global_sparsity"]),
		_init_contem_local(Rcpp::as<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_conetm_global(Rcpp::as<Eigen::VectorXd>(init["contem_global_sparsity"])) {}
};

struct LdltRecords : public RegRecords {
	Eigen::MatrixXd fac_record; // d_1, ..., d_m in D of LDLT

	LdltRecords(int num_iter, int dim, int num_design, int num_coef, int num_lowerchol)
	: RegRecords(num_iter, dim, num_design, num_coef, num_lowerchol),
		fac_record(Eigen::MatrixXd::Zero(num_iter + 1, dim)) {}
	
	LdltRecords(const Eigen::MatrixXd& alpha_record, const Eigen::MatrixXd& a_record, const Eigen::MatrixXd& d_record)
	: RegRecords(alpha_record, a_record), fac_record(d_record) {}

	void assignRecords(
		int id,
		const Eigen::VectorXd& coef_vec, const Eigen::VectorXd& contem_coef, const Eigen::VectorXd& diag_vec
	) {
		coef_record.row(id) = coef_vec;
		contem_coef_record.row(id) = contem_coef;
		fac_record.row(id) = diag_vec;
	}
};

struct HsInits : public LdltInits {
	Eigen::VectorXd _init_local;
	Eigen::VectorXd _init_group;
	double _init_global;
	Eigen::VectorXd _init_contem_local;
	Eigen::VectorXd _init_conetm_global;

	HsInits(Rcpp::List& init)
	: LdltInits(init),
		_init_local(Rcpp::as<Eigen::VectorXd>(init["local_sparsity"])),
		_init_group(Rcpp::as<Eigen::VectorXd>(init["group_sparsity"])),
		_init_global(init["global_sparsity"]),
		_init_contem_local(Rcpp::as<Eigen::VectorXd>(init["contem_local_sparsity"])),
		_init_conetm_global(Rcpp::as<Eigen::VectorXd>(init["contem_global_sparsity"])) {}
};

class McmcReg {
public:
	McmcReg(const RegParams& params, const LdltInits& inits, unsigned int seed)
	: include_mean(params._mean),
		x(params._x), y(params._y),
		num_iter(params._iter), dim(y.cols()), dim_design(x.cols()), num_design(y.rows()),
		num_lowerchol(dim * (dim - 1) / 2), num_coef(dim * dim_design),
		num_alpha(include_mean ? num_coef - dim : num_coef),
		reg_record(num_iter, dim, num_design, num_coef, num_lowerchol),
		mcmc_step(0), rng(seed),
		prior_mean_non(params._mean_non),
		prior_sd_non(params._sd_non * Eigen::VectorXd::Ones(dim)),
		coef_vec(Eigen::VectorXd::Zero(num_coef)),
		contem_coef(inits._contem), diag_vec(inits._diag),
		prior_alpha_mean(Eigen::VectorXd::Zero(num_coef)),
		prior_alpha_prec(Eigen::MatrixXd::Zero(num_coef, num_coef)),
		prior_chol_mean(Eigen::VectorXd::Zero(num_lowerchol)),
		prior_chol_prec(Eigen::MatrixXd::Identity(num_lowerchol, num_lowerchol)),
		coef_mat(inits._coef),
		contem_id(0),
		chol_lower(build_inv_lower(dim, contem_coef)),
		latent_innov(y - x * coef_mat),
		ortho_latent(Eigen::MatrixXd::Zero(num_design, dim)),
		prior_mean_j(Eigen::VectorXd::Zero(dim_design)),
		prior_prec_j(Eigen::MatrixXd::Identity(dim_design, dim_design)),
		coef_j(coef_mat),
		response_contem(Eigen::VectorXd::Zero(num_design)),
		sqrt_sv(Eigen::MatrixXd::Zero(num_design, dim)),
		prior_sig_shp(params._sig_shp), prior_sig_scl(params._sig_scl) {
		if (include_mean) {
			prior_alpha_mean.tail(dim) = prior_mean_non;
			prior_alpha_prec.bottomRightCorner(dim, dim).diagonal() = prior_sd_non.array().square();
		}
		coef_vec.head(num_alpha) = coef_mat.topRows(num_alpha / dim).reshaped();
		if (include_mean) {
			coef_vec.tail(dim) = coef_mat.bottomRows(1).transpose();
		}
		reg_record.assignRecords(0, coef_vec, contem_coef, diag_vec);
	}
	virtual ~McmcReg() = default;
	void updateCoef() {
		for (int j = 0; j < dim; j++) {
			prior_mean_j = prior_alpha_mean.segment(dim_design * j, dim_design);
			prior_prec_j = prior_alpha_prec.block(dim_design * j, dim_design * j, dim_design, dim_design);
			coef_j = coef_mat;
			coef_j.col(j).setZero();
			Eigen::MatrixXd chol_lower_j = chol_lower.bottomRows(dim - j); // L_(j:k) = a_jt to a_kt for t = 1, ..., j - 1
			Eigen::MatrixXd sqrt_sv_j = sqrt_sv.rightCols(dim - j); // use h_jt to h_kt for t = 1, .. n => (k - j + 1) x k
			Eigen::MatrixXd design_coef = kronecker_eigen(chol_lower_j.col(j), x).array().colwise() * sqrt_sv_j.reshaped().array(); // L_(j:k, j) otimes X0 scaled by D_(1:n, j:k): n(k - j + 1) x kp
			Eigen::VectorXd response_j = (((y - x * coef_j) * chol_lower_j.transpose()).array() * sqrt_sv_j.array()).reshaped(); // Hadamard product between: (Y - X0 A(-j))L_(j:k)^T and D_(1:n, j:k)
			varsv_regression(
				coef_mat.col(j),
				design_coef, response_j,
				prior_mean_j, prior_prec_j,
				rng
			);
		}
		coef_vec.head(num_alpha) = coef_mat.topRows(num_alpha / dim).reshaped();
		if (include_mean) {
			coef_vec.tail(dim) = coef_mat.bottomRows(1).transpose();
		}
	}
	void updateDiag() {
		ortho_latent = latent_innov * chol_lower.transpose(); // L eps_t <=> Z0 U
		// ortho_latent = (ortho_latent.array().square() + .0001).array().log(); // adjustment log(e^2 + c) for some c = 10^(-4) against numerical problems
		reg_ldlt_diag(diag_vec, prior_sig_shp, prior_sig_scl, ortho_latent, rng);
	}
	void updateImpact() {
		for (int j = 2; j < dim + 1; j++) {
			response_contem = latent_innov.col(j - 2).array() * sqrt_sv.col(j - 2).array(); // n-dim
			Eigen::MatrixXd design_contem = latent_innov.leftCols(j - 1).array().colwise() * sqrt_sv.col(j - 2).reshaped().array(); // n x (j - 1)
			contem_id = (j - 1) * (j - 2) / 2;
			varsv_regression(
				contem_coef.segment(contem_id, j - 1),
				design_contem, response_contem,
				prior_chol_mean.segment(contem_id, j - 1),
				prior_chol_prec.block(contem_id, contem_id, j - 1, j - 1),
				rng
			);
		}
	}
	void addStep() { mcmc_step++; }
	virtual void updateCoefPrec() = 0;
	virtual void updateCoefShrink() = 0;
	virtual void updateImpactPrec() = 0;
	virtual void updateRecords() = 0;
	virtual void doPosteriorDraws() = 0;
	virtual Rcpp::List returnRecords(int num_burn, int thin) const = 0;
	LdltRecords returnLdltRecords(int num_burn, int thin) const {
		LdltRecords res_record(
			thin_record(reg_record.coef_record, num_iter, num_burn, thin).derived(),
			thin_record(reg_record.contem_coef_record, num_iter, num_burn, thin).derived(),
			thin_record(reg_record.fac_record, num_iter, num_burn, thin).derived()
		);
		return res_record;
	}
	virtual SsvsRecords returnSsvsRecords(int num_burn, int thin) const = 0;
	virtual HorseshoeRecords returnHsRecords(int num_burn, int thin) const = 0;

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
	// SvRecords sv_record;
	LdltRecords reg_record;
	std::atomic<int> mcmc_step; // MCMC step
	boost::random::mt19937 rng; // RNG instance for multi-chain
	Eigen::VectorXd prior_mean_non; // prior mean of intercept term
	Eigen::VectorXd prior_sd_non; // prior sd of intercept term: c^2 I
	Eigen::VectorXd coef_vec;
	Eigen::VectorXd contem_coef;
	Eigen::VectorXd diag_vec;
	Eigen::VectorXd prior_alpha_mean; // prior mean vector of alpha
	Eigen::MatrixXd prior_alpha_prec; // prior precision of alpha
	Eigen::VectorXd prior_chol_mean; // prior mean vector of a = 0
	Eigen::MatrixXd prior_chol_prec; // prior precision of a = I
	Eigen::MatrixXd coef_mat;
	int contem_id;
	Eigen::MatrixXd chol_lower; // L in Sig_t^(-1) = L D_t^(-1) LT
	Eigen::MatrixXd latent_innov; // Z0 = Y0 - X0 A = (eps_p+1, eps_p+2, ..., eps_n+p)^T
  Eigen::MatrixXd ortho_latent; // orthogonalized Z0
	Eigen::VectorXd prior_mean_j; // Prior mean vector of j-th column of A
  Eigen::MatrixXd prior_prec_j; // Prior precision of j-th column of A
  Eigen::MatrixXd coef_j; // j-th column of A = 0: A(-j) = (alpha_1, ..., alpha_(j-1), 0, alpha_(j), ..., alpha_k)
	Eigen::VectorXd response_contem; // j-th column of Z0 = Y0 - X0 * A: n-dim
	Eigen::MatrixXd sqrt_sv; // stack sqrt of exp(h_t) = (exp(-h_1t / 2), ..., exp(-h_kt / 2)), t = 1, ..., n => n x k

private:
	Eigen::VectorXd prior_sig_shp;
	Eigen::VectorXd prior_sig_scl;
};

class HorseshoeReg : public McmcReg {
public:
	HorseshoeReg(const HsParams& params, const HsInits& inits, unsigned int seed)
	: McmcReg(params, inits, seed),
		grp_id(params._grp_id), grp_mat(params._grp_mat), grp_vec(grp_mat.reshaped()), num_grp(grp_id.size()),
		hs_record(num_iter, num_alpha, num_grp, num_lowerchol),
		local_lev(inits._init_local), group_lev(inits._init_group), global_lev(inits._init_global),
		local_fac(Eigen::VectorXd::Zero(num_alpha)),
		shrink_fac(Eigen::VectorXd::Zero(num_alpha)),
		latent_local(Eigen::VectorXd::Zero(num_alpha)), latent_group(Eigen::VectorXd::Zero(num_grp)), latent_global(0.0),
		lambda_mat(Eigen::MatrixXd::Zero(num_alpha, num_alpha)),
		coef_var(Eigen::VectorXd::Zero(num_alpha)),
		coef_var_loc(Eigen::MatrixXd::Zero(num_alpha / dim, dim)),
		contem_local_lev(inits._init_contem_local), contem_global_lev(inits._init_conetm_global),
		contem_var(Eigen::VectorXd::Zero(num_lowerchol)),
		latent_contem_local(Eigen::VectorXd::Zero(num_lowerchol)), latent_contem_global(Eigen::VectorXd::Zero(1)) {
		hs_record.assignRecords(0, shrink_fac, local_lev, group_lev, global_lev);
	}
	virtual ~HorseshoeReg() = default;
	void updateCoefPrec() override {
		for (int j = 0; j < num_grp; j++) {
			coef_var_loc = (grp_mat.array() == grp_id[j]).select(
				group_lev[j],
				coef_var_loc
			);
		}
		coef_var = coef_var_loc.reshaped();
		local_fac.array() = coef_var.array() * local_lev.array();
		lambda_mat.setZero();
		lambda_mat.diagonal() = 1 / (global_lev * local_fac.array());
		prior_alpha_prec.topLeftCorner(num_alpha, num_alpha) = lambda_mat;
		shrink_fac = 1 / (1 + lambda_mat.diagonal().array());
	}
	void updateCoefShrink() override {
		horseshoe_latent(latent_local, local_lev, rng);
		horseshoe_latent(latent_group, group_lev, rng);
		horseshoe_latent(latent_global, global_lev, rng);
		global_lev = horseshoe_global_sparsity(latent_global, local_fac, coef_vec.head(num_alpha), 1, rng);
		horseshoe_mn_sparsity(group_lev, grp_vec, grp_id, latent_group, global_lev, local_lev, coef_vec.head(num_alpha), 1, rng);
		horseshoe_local_sparsity(local_lev, latent_local, coef_var, coef_vec.head(num_alpha), 1, rng);
	}
	void updateImpactPrec() override {
		horseshoe_latent(latent_contem_local, contem_local_lev, rng);
		horseshoe_latent(latent_contem_global, contem_global_lev, rng);
		contem_var = contem_global_lev.replicate(1, num_lowerchol).reshaped();
		horseshoe_local_sparsity(contem_local_lev, latent_contem_local, contem_var, contem_coef, 1, rng);
		contem_global_lev[0] = horseshoe_global_sparsity(latent_contem_global[0], latent_contem_local, contem_coef, 1, rng);
		build_shrink_mat(prior_chol_prec, contem_var, contem_local_lev);
	}
	void updateRecords() override {
		reg_record.assignRecords(mcmc_step, coef_vec, contem_coef, diag_vec);
		hs_record.assignRecords(mcmc_step, shrink_fac, local_lev.cwiseSqrt(), group_lev.cwiseSqrt(), sqrt(global_lev));
	}
	void doPosteriorDraws() override {
		std::lock_guard<std::mutex> lock(mtx);
		addStep();
		updateCoefPrec();
		// sqrt_sv = (-lvol_draw / 2).array().exp(); // D_t before coef
		// Use replicate to define sqrt_sv from diag_vec
		updateCoef();
		updateCoefShrink();
		updateImpactPrec();
		latent_innov = y - x * coef_mat; // E_t before a
		updateImpact();
		chol_lower = build_inv_lower(dim, contem_coef); // L before h_t
		updateDiag();
		updateRecords();
	}
private:
	Eigen::VectorXi grp_id;
	Eigen::MatrixXi grp_mat;
	Eigen::VectorXi grp_vec;
	int num_grp;
	HorseshoeRecords hs_record;
	Eigen::VectorXd local_lev;
	Eigen::VectorXd group_lev;
	double global_lev;
	Eigen::VectorXd local_fac;
	Eigen::VectorXd shrink_fac;
	Eigen::VectorXd latent_local;
	Eigen::VectorXd latent_group;
	double latent_global;
	Eigen::MatrixXd lambda_mat;
	Eigen::VectorXd coef_var;
	Eigen::MatrixXd coef_var_loc;
	Eigen::VectorXd contem_local_lev;
	Eigen::VectorXd contem_global_lev; // -> double
	Eigen::VectorXd contem_var;
	Eigen::VectorXd latent_contem_local;
	Eigen::VectorXd latent_contem_global; // -> double
};

}; // namespace bvhar

#endif // MCMCREG_H