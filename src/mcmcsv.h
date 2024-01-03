#ifndef MCMCSV_H
#define MCMCSV_H

#include <RcppEigen.h>
#include "bvhardraw.h"

class McmcSv {
public:
	McmcSv(
		const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		const Eigen::VectorXd& prior_sig_shp, const Eigen::VectorXd& prior_sig_scl,
		const Eigen::VectorXd& prior_init_mean, const Eigen::MatrixXd& prior_init_prec
	);
	virtual ~McmcSv() = default;
	void UpdateCoef();
	void UpdateState();
	void UpdateImpact();
	void UpdateStateVar();
	void UpdateInitState();
	Eigen::VectorXd coef_vec; // alpha in VAR
	Eigen::VectorXd contem_coef; // a = a21, a31, a32, ..., ak1, ..., ak(k-1)
	Eigen::MatrixXd lvol_draw; // h_j = (h_j1, ..., h_jn)
	Eigen::VectorXd lvol_init; // h0 = h10, ..., hk0
	Eigen::VectorXd lvol_sig; // sigma_h^2 = (sigma_(h1i)^2, ..., sigma_(hki)^2)

protected:
	int dim; // k
  int dim_design; // kp(+1)
  int num_design; // n = T - p
  int num_lowerchol;
  int num_coef;
	Eigen::MatrixXd x;
	Eigen::MatrixXd y;
	Eigen::MatrixXd chol_lower; // L in Sig_t^(-1) = L D_t^(-1) LT
	Eigen::VectorXd prior_alpha_mean; // prior mean vector of alpha
	Eigen::MatrixXd prior_alpha_prec; // prior precision of alpha
	Eigen::VectorXd prior_chol_mean; // prior mean vector of a = 0
	Eigen::MatrixXd prior_chol_prec; // prior precision of a = I
  Eigen::MatrixXd latent_innov; // Z0 = Y0 - X0 A = (eps_p+1, eps_p+2, ..., eps_n+p)^T
  Eigen::MatrixXd ortho_latent; // orthogonalized Z0
	Eigen::VectorXd prior_mean_j; // Prior mean vector of j-th column of A
  Eigen::MatrixXd prior_prec_j; // Prior precision of j-th column of A
  Eigen::MatrixXd coef_j; // j-th column of A = 0: A(-j) = (alpha_1, ..., alpha_(j-1), 0, alpha_(j), ..., alpha_k)
	Eigen::VectorXd response_contem; // j-th column of Z0 = Y0 - X0 * A: n-dim
	Eigen::MatrixXd sqrt_sv; // stack sqrt of exp(h_t) = (exp(-h_1t / 2), ..., exp(-h_kt / 2)), t = 1, ..., n => n x k
	Eigen::MatrixXd coef_mat;
	int contem_id;

private:
	Eigen::VectorXd prior_sig_shp;
	Eigen::VectorXd prior_sig_scl;
	Eigen::VectorXd prior_init_mean;
	Eigen::MatrixXd prior_init_prec;
};

class MinnSv : McmcSv {
	public:
		MinnSv(
			const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
			const Eigen::VectorXd& prior_sig_shp, const Eigen::VectorXd& prior_sig_scl,
			const Eigen::VectorXd& prior_init_mean, const Eigen::MatrixXd& prior_init_prec,
			const Eigen::MatrixXd& prior_coef_mean, const Eigen::MatrixXd& prior_coef_prec, const Eigen::MatrixXd& prec_diag
		);
		virtual ~MinnSv() = default;
};

class SsvsSv : McmcSv {
public:
	SsvsSv(
		const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		const Eigen::VectorXd& prior_sig_shp, const Eigen::VectorXd& prior_sig_scl,
		const Eigen::VectorXd& prior_init_mean, const Eigen::MatrixXd& prior_init_prec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXd& grp_mat,
		const Eigen::VectorXd& coef_spike, const Eigen::VectorXd& coef_slab,
    const Eigen::VectorXd& coef_slab_weight, const Eigen::VectorXd& chol_spike,
    const Eigen::VectorXd& chol_slab, const Eigen::VectorXd& chol_slab_weight,
    const double& coef_s1, const double& coef_s2,
    const double& chol_s1, const double& chol_s2,
    const Eigen::VectorXd& mean_non, const double& sd_non, const bool& include_mean
	);
	virtual ~SsvsSv() = default;
	void UpdateCoefPrec();
	void UpdateCoefShrink();
	void UpdateImpactPrec();
	Eigen::VectorXd coef_dummy;
	Eigen::VectorXd coef_weight;
	Eigen::VectorXd contem_dummy;
	Eigen::VectorXd contem_weight;
private:
	bool include_mean;
	int num_alpha;
	int num_grp;
	Eigen::VectorXi grp_id;
	Eigen::MatrixXd grp_mat;
	Eigen::VectorXd grp_vec;
	Eigen::VectorXd coef_spike;
	Eigen::VectorXd coef_slab;
	Eigen::VectorXd contem_spike;
	Eigen::VectorXd contem_slab;
	double coef_s1, coef_s2;
	double contem_s1, contem_s2;
	double prior_sd_non;
	Eigen::VectorXd prior_sd;
	Eigen::VectorXd slab_weight; // pij vector
	Eigen::MatrixXd slab_weight_mat; // pij matrix: (dim*p) x dim
	Eigen::VectorXd coef_mixture_mat;
};

class HorseshoeSv : McmcSv {
public:
	HorseshoeSv(
		const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		const Eigen::VectorXd& prior_sig_shp, const Eigen::VectorXd& prior_sig_scl,
		const Eigen::VectorXd& prior_init_mean, const Eigen::MatrixXd& prior_init_prec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXd& grp_mat,
		const Eigen::VectorXd& init_local, const Eigen::VectorXd& init_global,
		const Eigen::VectorXd& init_contem_local, const Eigen::VectorXd& init_contem_global
	);
	virtual ~HorseshoeSv() = default;
	void UpdateCoefPrec();
	void UpdateCoefShrink();
	void UpdateImpactPrec();
	Eigen::VectorXd local_lev;
	Eigen::VectorXd global_lev;
	Eigen::VectorXd shrink_fac;

private:
	int num_grp;
	Eigen::VectorXi grp_id;
	Eigen::MatrixXd grp_mat;
	Eigen::VectorXd grp_vec;
	Eigen::VectorXd latent_local;
	Eigen::VectorXd latent_global;
	Eigen::VectorXd coef_var;
	Eigen::MatrixXd coef_var_loc;
	Eigen::VectorXd contem_local_lev;
	Eigen::VectorXd contem_global_lev;
	Eigen::VectorXd contem_var;
	Eigen::VectorXd latent_contem_local;
	Eigen::VectorXd latent_contem_global;
};

#endif