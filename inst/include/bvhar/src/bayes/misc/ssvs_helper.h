#ifndef BVHAR_BAYES_MISC_SSVS_HELPER_H
#define BVHAR_BAYES_MISC_SSVS_HELPER_H

#include "./helper.h"

namespace baecon {
namespace bvhar {

// Building Spike-and-slab SD Diagonal Matrix
// 
// In MCMC process of SSVS, this function computes diagonal matrix \eqn{D} or \eqn{D_j} defined by spike-and-slab sd.
// 
// @param spike_sd Standard deviance for Spike normal distribution
// @param slab_sd Standard deviance for Slab normal distribution
// @param mixture_dummy Indicator vector (0-1) corresponding to each element
inline Eigen::VectorXd build_ssvs_sd(Eigen::VectorXd spike_sd, Eigen::VectorXd slab_sd, Eigen::VectorXd mixture_dummy) {
  Eigen::VectorXd res(spike_sd.size());
  res.array() = (1 - mixture_dummy.array()) * spike_sd.array() + mixture_dummy.array() * slab_sd.array(); // diagonal term = spike_sd if mixture_dummy = 0 while slab_sd if mixture_dummy = 1
  return res;
}

// Generating Dummy Vector for Parameters in SSVS Gibbs Sampler
// 
// In MCMC process of SSVS, this function generates latent \eqn{\gamma_j} or \eqn{\omega_{ij}} conditional posterior.
// 
// @param param_obs Realized parameters vector
// @param sd_numer Standard deviance for Slab normal distribution, which will be used for numerator.
// @param sd_denom Standard deviance for Spike normal distribution, which will be used for denominator.
// @param slab_weight Proportion of nonzero coefficients
inline void ssvs_dummy(Eigen::VectorXd& dummy, Eigen::VectorXd param_obs,
											 Eigen::VectorXd& sd_numer, Eigen::Ref<const Eigen::VectorXd> sd_denom,
											 Eigen::VectorXd& slab_weight, BVHAR_BHRNG& rng) {
  int num_latent = slab_weight.size();
	Eigen::VectorXd exp_u1 = -param_obs.array().square() / (2 * sd_numer.array());
	Eigen::VectorXd exp_u2 = -param_obs.array().square() / (2 * sd_denom.array());
	Eigen::VectorXd max_exp = exp_u1.cwiseMax(exp_u2); // use log-sum-exp against overflow
	exp_u1 = slab_weight.array() * (exp_u1 - max_exp).array().exp() / sd_numer.cwiseSqrt().array();
	exp_u2 = (1 - slab_weight.array()) * (exp_u2 - max_exp).array().exp() / sd_denom.cwiseSqrt().array();
  for (int i = 0; i < num_latent; i++) {
		dummy[i] = ber_rand(exp_u1[i] / (exp_u1[i] + exp_u2[i]), rng);
  }
}

// Generating Slab Weight Vector in SSVS Gibbs Sampler
// 
// In MCMC process of SSVS, this function generates \eqn{p_j}.
// 
// @param param_obs Indicator variables
// @param prior_s1 First prior shape of Beta distribution
// @param prior_s2 Second prior shape of Beta distribution
inline void ssvs_weight(Eigen::VectorXd& weight, Eigen::VectorXd param_obs, double prior_s1, double prior_s2, BVHAR_BHRNG& rng) {
  int num_latent = param_obs.size();
  double post_s1 = prior_s1 + param_obs.sum(); // s1 + number of ones
  double post_s2 = prior_s2 + num_latent - param_obs.sum(); // s2 + number of zeros
  for (int i = 0; i < num_latent; i++) {
		weight[i] = beta_rand(post_s1, post_s2, rng);
  }
}

// Generating Slab Weight Vector in MN-SSVS Gibbs Sampler
// 
// In MCMC process of SSVS, this function generates \eqn{p_j}.
// 
// @param grp_vec Group vector
// @param grp_id Unique group id
// @param param_obs Indicator variables
// @param prior_s1 First prior shape of Beta distribution
// @param prior_s2 Second prior shape of Beta distribution
inline void ssvs_mn_weight(Eigen::VectorXd& weight, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
  												 Eigen::VectorXd& param_obs, Eigen::VectorXd& prior_s1, Eigen::VectorXd& prior_s2, BVHAR_BHRNG& rng) {
  int num_grp = grp_id.size();
  int num_latent = param_obs.size();
	Eigen::Array<bool, Eigen::Dynamic, 1> global_id;
  int mn_size = 0;
  for (int i = 0; i < num_grp; i++) {
		global_id = grp_vec.array() == grp_id[i];
		mn_size = global_id.count();
    Eigen::VectorXd mn_param(mn_size);
		for (int j = 0, k = 0; j < num_latent; ++j) {
			if (global_id[j]) {
				mn_param[k++] = param_obs[j];
			}
		}
    weight[i] = beta_rand(prior_s1[i] + mn_param.sum(), prior_s2[i] + mn_size - mn_param.sum(), rng);
  }
}

// Generating SSVS Local Slab Parameter
// 
// @param slab_param Slab parameter
// @param dummy_param Bernoulli parameter
// @param coef_vec Coefficient
// @param shp IG shape for slab parameter
// @param scl IG scale for slab parameter
// @param spike_scl scaling factor to make spike sd smaller than slab sd (spike_sd = spike_scl * slab_sd)
// @param rng boost rng
inline void ssvs_local_slab(Eigen::VectorXd& slab_param, Eigen::VectorXd& dummy_param, Eigen::Ref<Eigen::VectorXd> coef_vec,
														double& shp, double& scl, double& spike_scl, BVHAR_BHRNG& rng) {
	for (int i = 0; i < coef_vec.size(); ++i) {
		slab_param[i] = 1 / gamma_rand(
			shp + .5,
			1 / (scl + coef_vec[i] * coef_vec[i] / (dummy_param[i] + (1 - dummy_param[i]) * spike_scl)),
			rng
		);
	}
}

inline double ssvs_logdens_scl(double& cand, Eigen::Ref<Eigen::VectorXd> coef_vec, Eigen::Ref<Eigen::VectorXd> slab_sd) {
	// return -(coef_vec.array() / slab_sd.array()).square().sum() / (2 * cand * cand) - coef_vec.size() * log(cand);
	return -(coef_vec.array().square() / slab_sd.array()).sum() / (2 * cand) - coef_vec.size() * log(cand) / 2;
}

inline void ssvs_scl_griddy(double& spike_scl, int grid_size,
														Eigen::Ref<Eigen::VectorXd> coef_vec, Eigen::Ref<Eigen::VectorXd> slab_param, BVHAR_BHRNG& rng) {
	Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(grid_size + 2, 0.0, 1.0).segment(1, grid_size);
	Eigen::VectorXd log_wt(grid_size);
	for (int i = 0; i < grid_size; ++i) {
		log_wt[i] = ssvs_logdens_scl(grid[i], coef_vec, slab_param);
	}
	Eigen::VectorXd weight = (log_wt.array() - log_wt.maxCoeff()).exp();
	weight /= weight.sum();
	spike_scl = grid[cat_rand(weight, rng)];
}

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_MISC_SSVS_HELPER_H