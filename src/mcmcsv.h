#ifndef MCMCSV_H
#define MCMCSV_H

#include <RcppEigen.h>
#include "bvhardraw.h"
#include <atomic>
#include <mutex>
#include <vector> // std::vector
#include <memory> // std::unique_ptr

struct SvParams {
	int _iter;
	Eigen::MatrixXd _x;
	Eigen::MatrixXd _y;
	Eigen::VectorXd _sig_shp;
	Eigen::VectorXd _sig_scl;
	Eigen::VectorXd _init_mean;
	Eigen::MatrixXd _init_prec;

	SvParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& spec
	);
};

struct MinnParams : public SvParams {
	Eigen::MatrixXd _prior_mean;
	Eigen::MatrixXd _prior_prec;
	Eigen::MatrixXd _prec_diag;

	MinnParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& sv_spec, Rcpp::List& priors
	);
};

struct SsvsParams : public SvParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXd _grp_mat;
	Eigen::VectorXd _coef_spike;
	Eigen::VectorXd _coef_slab;
	Eigen::VectorXd _coef_weight;
	Eigen::VectorXd _contem_spike;
	Eigen::VectorXd _contem_slab;
	Eigen::VectorXd _contem_weight;
	double _coef_s1;
	double _coef_s2;
	double _contem_s1;
	double _contem_s2;
	Eigen::VectorXd _mean_non;
	double _sd_non;
	bool _mean;

	SsvsParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& sv_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXd& grp_mat,
		Rcpp::List& ssvs_spec,
		bool include_mean
	);
};

struct HorseshoeParams : public SvParams {
	Eigen::VectorXi _grp_id;
	Eigen::MatrixXd _grp_mat;
	// Eigen::VectorXd _init_local;
	// Eigen::VectorXd _init_global;
	// Eigen::VectorXd _init_contem_local;
	// Eigen::VectorXd _init_conetm_global;

	HorseshoeParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		Rcpp::List& sv_spec,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXd& grp_mat
	);
	// ,Rcpp::List& hs_spec
};

struct SvInits {
	Eigen::MatrixXd _coef;
	Eigen::VectorXd _contem;
	Eigen::VectorXd _lvol_init;
	Eigen::MatrixXd _lvol;
	Eigen::VectorXd _lvol_sig;

	SvInits(const SvParams& params);
	SvInits(Rcpp::List& init);
};

struct SsvsInits : public SvInits {
	Eigen::VectorXd _coef_dummy;
	Eigen::VectorXd _coef_weight; // in SsvsParams: move coef_mixture and chol_mixture in set_ssvs()?
	Eigen::VectorXd _contem_weight; // in SsvsParams
	
	SsvsInits(Rcpp::List& init);
};

struct HorseshoeInits : public SvInits {
	Eigen::VectorXd _init_local;
	Eigen::VectorXd _init_global;
	Eigen::VectorXd _init_contem_local;
	Eigen::VectorXd _init_conetm_global;
	
	HorseshoeInits(Rcpp::List& init);
};

class McmcSv {
public:
	McmcSv(const SvParams& params, const SvInits& inits, unsigned int seed);
	virtual ~McmcSv() = default;
	virtual void updateCoefPrec() = 0;
	virtual void updateCoefShrink() = 0;
	void updateCoef();
	void updateState();
	virtual void updateImpactPrec() = 0;
	void updateImpact();
	void updateStateVar();
	void updateInitState();
	virtual void updateRecords() = 0;
	void addStep();
	virtual void doPosteriorDraws() = 0;
	virtual Rcpp::List returnRecords(int num_burn, int thin) const = 0;

protected:
	Eigen::MatrixXd x;
	Eigen::MatrixXd y;
	std::mutex mtx;
	Eigen::MatrixXd coef_record; // alpha in VAR
	Eigen::MatrixXd contem_coef_record; // a = a21, a31, a32, ..., ak1, ..., ak(k-1)
	Eigen::MatrixXd lvol_sig_record; // sigma_h^2 = (sigma_(h1i)^2, ..., sigma_(hki)^2)
	Eigen::MatrixXd lvol_init_record; // h0 = h10, ..., hk0
	Eigen::MatrixXd lvol_record; // time-varying h = (h_1, ..., h_k) with h_j = (h_j1, ..., h_jn), row-binded
	int num_iter;
	int dim; // k
  int dim_design; // kp(+1)
  int num_design; // n = T - p
  int num_lowerchol;
  int num_coef;
	std::atomic<int> mcmc_step; // MCMC step
	boost::random::mt19937 rng; // RNG instance for multi-chain
	Eigen::VectorXd coef_vec;
	Eigen::VectorXd contem_coef;
	Eigen::MatrixXd lvol_draw; // h_j = (h_j1, ..., h_jn)
	Eigen::VectorXd lvol_init;
	Eigen::VectorXd lvol_sig;
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
	Eigen::VectorXd prior_init_mean;
	Eigen::MatrixXd prior_init_prec;
};

class MinnSv : public McmcSv {
public:
	MinnSv(const MinnParams& params, const SvInits& inits, unsigned int seed);
	virtual ~MinnSv() = default;
	void updateCoefPrec() override {};
	void updateCoefShrink() override {};
	void updateImpactPrec() override {};
	void updateRecords() override;
	void doPosteriorDraws() override;
	Rcpp::List returnRecords(int num_burn, int thin) const override;
};

class SsvsSv : public McmcSv {
public:
	SsvsSv(const SsvsParams& params, const SsvsInits& inits, unsigned int seed);
	virtual ~SsvsSv() = default;
	void updateCoefPrec() override;
	void updateCoefShrink() override;
	void updateImpactPrec() override;
	void updateRecords() override;
	void doPosteriorDraws() override;
	Rcpp::List returnRecords(int num_burn, int thin) const override;
private:
	bool include_mean;
	int num_alpha;
	Eigen::VectorXi grp_id;
	Eigen::MatrixXd grp_mat;
	Eigen::VectorXd grp_vec;
	int num_grp;
	Eigen::MatrixXd coef_dummy_record;
	Eigen::MatrixXd coef_weight_record;
	Eigen::MatrixXd contem_dummy_record;
	Eigen::MatrixXd contem_weight_record;
	Eigen::VectorXd coef_dummy;
	Eigen::VectorXd coef_weight;
	Eigen::VectorXd contem_dummy;
	Eigen::VectorXd contem_weight;
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

class HorseshoeSv : public McmcSv {
public:
	HorseshoeSv(const HorseshoeParams& params, const HorseshoeInits& inits, unsigned int seed);
	virtual ~HorseshoeSv() = default;
	void updateCoefPrec() override;
	void updateCoefShrink() override;
	void updateImpactPrec() override;
	void updateRecords() override;
	void doPosteriorDraws() override;
	Rcpp::List returnRecords(int num_burn, int thin) const override;

private:
	Eigen::VectorXi grp_id;
	Eigen::MatrixXd grp_mat;
	Eigen::VectorXd grp_vec;
	int num_grp;
	Eigen::MatrixXd local_record;
	Eigen::MatrixXd global_record;
	Eigen::MatrixXd shrink_record;
	Eigen::VectorXd local_lev;
	Eigen::VectorXd global_lev;
	Eigen::VectorXd shrink_fac;
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