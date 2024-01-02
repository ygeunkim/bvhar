#ifndef MCMCSV_H
#define MCMCSV_H

#include <RcppEigen.h>
#include "bvhardraw.h"

class McmcSv {
public:
	McmcSv(Eigen::MatrixXd x, Eigen::MatrixXd y, Eigen::VectorXd prior_sig_shp, Eigen::VectorXd prior_sig_scl, Eigen::VectorXd prior_init_mean, Eigen::MatrixXd prior_init_prec);
	virtual ~McmcSv() = default;
	void UpdateCoef(Eigen::MatrixXd prior_alpha_mean, Eigen::MatrixXd prior_alpha_prec);
	void UpdateState();
	void UpdateImpact(Eigen::MatrixXd prior_chol_mean, Eigen::MatrixXd prior_chol_prec);
	void UpdateStateVar();
	void UpdateInitState();
	Eigen::VectorXd contem_coef; // a = a21, a31, a32, ..., ak1, ..., ak(k-1)
	Eigen::MatrixXd coef_mat; // alpha in VAR
	Eigen::MatrixXd lvol_draw; // h_j = (h_j1, ..., h_jn)
	Eigen::VectorXd lvol_init; // h0 = h10, ..., hk0
	Eigen::VectorXd lvol_sig; // sigma_h^2 = (sigma_(h1i)^2, ..., sigma_(hki)^2)

private:
	int dim; // k
  int dim_design; // kp(+1)
  int num_design; // n = T - p
  int num_lowerchol;
  int num_coef;
	Eigen::MatrixXd x;
	Eigen::MatrixXd y;
	Eigen::VectorXd prior_sig_shp;
	Eigen::VectorXd prior_sig_scl;
	Eigen::VectorXd prior_init_mean;
	Eigen::MatrixXd prior_init_prec;
	Eigen::MatrixXd chol_lower; // L in Sig_t^(-1) = L D_t^(-1) LT
  Eigen::MatrixXd latent_innov; // Z0 = Y0 - X0 A = (eps_p+1, eps_p+2, ..., eps_n+p)^T
  Eigen::MatrixXd ortho_latent; // orthogonalized Z0
	Eigen::VectorXd prior_mean_j; // Prior mean vector of j-th column of A
  Eigen::MatrixXd prior_prec_j; // Prior precision of j-th column of A
  Eigen::MatrixXd coef_j; // j-th column of A = 0: A(-j) = (alpha_1, ..., alpha_(j-1), 0, alpha_(j), ..., alpha_k)
	Eigen::VectorXd response_contem; // j-th column of Z0 = Y0 - X0 * A: n-dim
	Eigen::MatrixXd sqrt_sv; // stack sqrt of exp(h_t) = (exp(-h_1t / 2), ..., exp(-h_kt / 2)), t = 1, ..., n => n x k
	int contem_id;
};

#endif