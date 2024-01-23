#ifndef MCMCSSVS_H
#define MCMCSSVS_H

#include "bvhardraw.h"
#include "bvharprogress.h"

namespace bvhar {

class McmcSsvs {
public:
	McmcSsvs(
		int num_iter, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y,
		const Eigen::VectorXd& init_coef, const Eigen::VectorXd& init_chol_diag, const Eigen::VectorXd& init_chol_upper,
  	const Eigen::VectorXd& init_coef_dummy, const Eigen::VectorXd& init_chol_dummy,
  	const Eigen::VectorXd& coef_spike, const Eigen::VectorXd& coef_slab, const Eigen::VectorXd& coef_slab_weight,
  	const Eigen::VectorXd& shape, const Eigen::VectorXd& rate,
  	const double& coef_s1, const double& coef_s2,
  	const Eigen::VectorXd& chol_spike, const Eigen::VectorXd& chol_slab, const Eigen::VectorXd& chol_slab_weight,
  	const double& chol_s1, const double& chol_s2,
  	const Eigen::VectorXi& grp_id, const Eigen::MatrixXi& grp_mat,
  	const Eigen::VectorXd& mean_non, const double& sd_non, bool include_mean, bool init_gibbs,
		unsigned int seed
	);
	virtual ~McmcSsvs() = default;
	void addStep();
	void updateChol();
	void updateCholDummy();
	void updateCoef();
	void updateCoefDummy();
	void updateRecords();
	void doPosteriorDraws();
	Rcpp::List returnRecords(int num_burn, int thin) const;

private:
	int num_iter;
	Eigen::MatrixXd x;
	Eigen::MatrixXd y;
	std::mutex mtx;
	int dim; // k
	int dim_design; // kp(+1)
	int num_design; // n = T - p
	int num_coef;
	int num_upperchol;
	std::atomic<int> mcmc_step; // MCMC step
	boost::random::mt19937 rng; // RNG instance for multi-chain
	Eigen::VectorXd coef_spike;
	Eigen::VectorXd coef_slab;
	Eigen::VectorXd chol_spike;
	Eigen::VectorXd chol_slab;
	Eigen::VectorXd shape;
	Eigen::VectorXd rate;
	double coef_s1, coef_s2;
	double chol_s1, chol_s2;
	Eigen::VectorXd prior_mean_non;
	double prior_sd_non;
	Eigen::VectorXd prior_sd;
	Eigen::VectorXi grp_id;
	Eigen::MatrixXi grp_mat;
	Eigen::VectorXi grp_vec;
	int num_grp;
	bool include_mean;
	int num_restrict;
	Eigen::VectorXd coef_mean;
	Eigen::VectorXd prior_mean;
	Eigen::VectorXd coef_mixture_mat;
	Eigen::VectorXd chol_mixture_mat;
	Eigen::VectorXd slab_weight; // pij vector
	Eigen::MatrixXd slab_weight_mat; // pij matrix: (dim*p) x dim
	Eigen::MatrixXd gram;
	Eigen::MatrixXd coef_ols;
	Eigen::VectorXd coef_vec;
	Eigen::MatrixXd chol_ols;
	Eigen::MatrixXd coef_record;
	Eigen::MatrixXd coef_dummy_record;
	Eigen::MatrixXd coef_weight_record;
	Eigen::MatrixXd chol_diag_record;
	Eigen::MatrixXd chol_upper_record;
	Eigen::MatrixXd chol_dummy_record;
	Eigen::MatrixXd chol_weight_record;
	Eigen::MatrixXd chol_factor_record; // 3d matrix alternative
	Eigen::VectorXd coef_weight;
	Eigen::VectorXd chol_weight;
	Eigen::VectorXd coef_draw;
	Eigen::VectorXd coef_dummy;
	Eigen::VectorXd chol_diag;
	Eigen::VectorXd chol_coef;
	Eigen::VectorXd chol_dummy;
	Eigen::MatrixXd chol_factor;
	Eigen::MatrixXd coef_mat;
	Eigen::MatrixXd sse_mat;
};

} // namespace bvhar

#endif // MCMCSSVS_H