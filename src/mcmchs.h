#ifndef MCMCHS_H
#define MCMCHS_H

#include <RcppEigen.h>
#include "bvhardraw.h"

struct HsParams {
	int _iter;
	Eigen::MatrixXd _x;
	Eigen::MatrixXd _y;
  Eigen::VectorXi _grp_id;
	Eigen::MatrixXd _grp_mat;
	Eigen::VectorXd _init_local;
	Eigen::VectorXd _init_global;
	double _init_sigma;
	
	HsParams(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
    const Eigen::VectorXd& init_local, const Eigen::VectorXd& init_global, const double& init_sigma,
		const Eigen::VectorXi& grp_id, const Eigen::MatrixXd& grp_mat
	);
};

class McmcHs {
public:
	McmcHs(const HsParams& params);
	virtual ~McmcHs() = default;
	void addStep();
	void updateCoefCov();
	virtual void updateCoef();
	void updateCov();
	void doPosteriorDraws();
	Rcpp::List returnRecords(int num_burn) const;
protected:
	int num_iter;
	int dim; // k
	int dim_design; // kp(+1)
	int num_design; // n = T - p
	int num_coef;
	int mcmc_step; // MCMC step
	Eigen::MatrixXd design_mat;
	Eigen::VectorXd response_vec;
	Eigen::MatrixXd lambda_mat; // covariance
	Eigen::VectorXi grp_id;
	Eigen::MatrixXd grp_mat;
	Eigen::VectorXd grp_vec;
	int num_grp;
	Eigen::VectorXd coef_draw;
	double sig_draw;
	Eigen::VectorXd local_lev;
	Eigen::VectorXd global_lev;
	int glob_len;
	Eigen::VectorXd shrink_fac;
	Eigen::VectorXd latent_local;
	Eigen::VectorXd latent_global;
	Eigen::VectorXd coef_var;
	Eigen::MatrixXd coef_var_loc;
	Eigen::MatrixXd coef_record;
  Eigen::MatrixXd local_record;
  Eigen::MatrixXd global_record; // tau1: own-lag, tau2: cross-lag, ...
  Eigen::VectorXd sig_record;
  Eigen::MatrixXd shrink_record;
};

class BlockHs : public McmcHs {
public:
	BlockHs(const HsParams& params);
	virtual ~BlockHs() = default;
	void updateCoef() override;
private:
	Eigen::VectorXd block_coef;
};

class FastHs : public McmcHs {
public:
	FastHs(const HsParams& params);
	virtual ~FastHs() = default;
	void updateCoef() override;
};

#endif