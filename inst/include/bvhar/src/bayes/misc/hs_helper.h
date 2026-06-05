#ifndef BVHAR_BAYES_MISC_HS_HELPER_H
#define BVHAR_BAYES_MISC_HS_HELPER_H

#include "./helper.h"

namespace baecon {
namespace bvhar {

// Generating the Squared Grouped Global Sparsity Hyperparameter in Horseshoe Gibbs Sampler
// 
// In MCMC process of Horseshoe prior, this function generates the grouped global sparsity hyperparameter.
// 
// @param global_latent Latent global vector
// @param local_hyperparam Squared local sparsity hyperparameters vector
// @param coef_vec Coefficients vector
// @param prior_var Variance constant of the likelihood
inline double horseshoe_global_sparsity(double global_latent, Eigen::Ref<const Eigen::VectorXd> local_hyperparam,
                                 				Eigen::Ref<Eigen::VectorXd> coef_vec, const double& prior_var, BVHAR_BHRNG& rng) {
  int dim = coef_vec.size();
	// double invgam_scl = 1 / global_latent + (coef_vec.array().square() / (2 * prior_var * local_hyperparam.array().square())).sum();
	return 1 / gamma_rand(
		(dim + 1) / 2,
		1 / (1 / global_latent + (coef_vec.array().square() / (2 * prior_var * local_hyperparam.array())).sum()),
		rng
	);
}

// Generating the Squared Grouped Global Sparsity Hyperparameter in Horseshoe Gibbs Sampler
// 
// In MCMC process of Horseshoe prior, this function generates the grouped global sparsity hyperparameter.
// 
// @param glob_lev Squared global sparsity hyperparameters
// @param grp_vec Group vector
// @param grp_id Unique group id
// @param global_latent Latent global vector
// @param local_hyperparam Squared local sparsity hyperparameters
// @param coef_vec Coefficients vector
// @param prior_var Variance constant of the likelihood
inline void horseshoe_mn_global_sparsity(Eigen::VectorXd& global_lev, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
                                  			 Eigen::VectorXd& global_latent, Eigen::VectorXd& local_hyperparam,
																				 Eigen::Ref<Eigen::VectorXd> coef_vec, const double& prior_var, BVHAR_BHRNG& rng) {
  int num_grp = grp_id.size();
  int num_coef = coef_vec.size();
	Eigen::Array<bool, Eigen::Dynamic, 1> global_id;
  int mn_size = 0;
  for (int i = 0; i < num_grp; i++) {
		global_id = grp_vec.array() == grp_id[i];
		mn_size = global_id.count();
    Eigen::VectorXd mn_coef(mn_size);
    Eigen::VectorXd mn_local(mn_size);
		for (int j = 0, k = 0; j < num_coef; ++j) {
			if (global_id[j]) {
				mn_coef[k] = coef_vec[j];
				mn_local[k++] = local_hyperparam[j];
			}
		}
    global_lev[i] = horseshoe_global_sparsity(global_latent[i], mn_local, mn_coef, prior_var, rng); 
  }
}

// For group shrinkage
inline void horseshoe_mn_sparsity(Eigen::VectorXd& group_lev, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
                                  Eigen::VectorXd& group_latent, double& global_lev, Eigen::VectorXd& local_hyperparam,
																	Eigen::Ref<Eigen::VectorXd> coef_vec, const double& prior_var, BVHAR_BHRNG& rng) {
  int num_grp = grp_id.size();
  int num_coef = coef_vec.size();
	Eigen::Array<bool, Eigen::Dynamic, 1> group_id;
  int mn_size = 0;
  for (int i = 0; i < num_grp; i++) {
		group_id = grp_vec.array() == grp_id[i];
		mn_size = group_id.count();
    Eigen::VectorXd mn_coef(mn_size);
    Eigen::VectorXd mn_local(mn_size);
		for (int j = 0, k = 0; j < num_coef; ++j) {
			if (group_id[j]) {
				mn_coef[k] = coef_vec[j];
				mn_local[k++] = global_lev * local_hyperparam[j];
			}
		}
    group_lev[i] = horseshoe_global_sparsity(group_latent[i], mn_local, mn_coef, prior_var, rng); 
  }
}

// Generating the Latent Vector for Sparsity Hyperparameters in Horseshoe Gibbs Sampler
// 
// In MCMC process of Horseshoe prior, this function generates the latent vector for local sparsity hyperparameters.
// 
// @param hyperparam sparsity hyperparameters vector
inline void horseshoe_latent(Eigen::VectorXd& latent, Eigen::VectorXd& hyperparam, BVHAR_BHRNG& rng) {
  int dim = hyperparam.size();
  for (int i = 0; i < dim; i++) {
		latent[i] = 1 / gamma_rand(1.0, 1 / (1 + 1 / hyperparam[i]), rng);
  }
}
// overloading
inline void horseshoe_latent(double& latent, double& hyperparam, BVHAR_BHRNG& rng) {
  latent = 1 / gamma_rand(1.0, 1 / (1 + 1 / hyperparam), rng);
}

// Generating the Squared Grouped Local Sparsity Hyperparameters Vector in Horseshoe Gibbs Sampler
// 
// In MCMC process of Horseshoe prior, this function generates the local sparsity hyperparameters vector.
// 
// @param local_latent Latent vectors defined for local sparsity vector
// @param global_hyperparam Squared global sparsity hyperparameter vector
// @param coef_vec Coefficients vector
// @param prior_var Variance constant of the likelihood
inline void horseshoe_local_sparsity(Eigen::VectorXd& local_lev, Eigen::VectorXd& local_latent, Eigen::VectorXd& global_hyperparam,
                            				 Eigen::Ref<Eigen::VectorXd> coef_vec, const double& prior_var, BVHAR_BHRNG& rng) {
  int dim = coef_vec.size();
	Eigen::VectorXd invgam_scl = (1 / local_latent.array() + coef_vec.array().square() / (2 * prior_var * global_hyperparam.array())).cwiseInverse();
  for (int i = 0; i < dim; i++) {
		local_lev[i] = 1 / gamma_rand(1.0, invgam_scl[i], rng);
  }
}

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_MISC_HS_HELPER_H