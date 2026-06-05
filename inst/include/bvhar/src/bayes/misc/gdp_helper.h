#ifndef BVHAR_BAYES_MISC_GDP_HELPER_H
#define BVHAR_BAYES_MISC_GDP_HELPER_H

#include "./helper.h"

namespace baecon {
namespace bvhar {

// Draw Local sparsity in GDP prior
// 
// @param local_param
// @param local_shape
// @param coef
// @param rng
inline void gdp_local_sparsity(Eigen::Ref<Eigen::VectorXd> local_param, Eigen::Ref<const Eigen::VectorXd> local_shape,
															 Eigen::Ref<Eigen::VectorXd> coef, BVHAR_BHRNG& rng) {
	for (int i = 0; i < local_param.size(); ++i) {
		local_param[i] = sim_invgauss(abs(local_shape[i] / coef[i]), local_shape[i] * local_shape[i], rng);
		// local_param[i] = 1 / (local_shape[i] * sim_invgauss(1 / abs(coef[i]), local_shape[i], rng));
	}
}

// Draw Rate in GDP prior
// @param prior_shape
// @param prior_rate
// @param coef
// @param rng
inline void gdp_exp_rate(Eigen::Ref<Eigen::VectorXd> rate_hyper, double prior_shape, double prior_rate,
											 		 Eigen::Ref<Eigen::VectorXd> coef, BVHAR_BHRNG& rng) {
	for (int i = 0; i < rate_hyper.size(); ++i) {
		rate_hyper[i] = gamma_rand(prior_shape + 1, 1 / (prior_rate + abs(coef[i])), rng);
	}
}

// Draw Group Rate in GDP prior
// @param group_rate
// @param coef
// @param grp_vec Group vector
// @param grp_id Unique group id
// @param rng
inline void gdp_exp_rate(Eigen::Ref<Eigen::VectorXd> group_rate, double prior_shape, double prior_rate,
												 Eigen::Ref<Eigen::VectorXd> coef,
												 Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id, BVHAR_BHRNG& rng) {
	Eigen::Array<bool, Eigen::Dynamic, 1> group_id;
  int mn_size = 0;
	int num_coef = coef.size();
  for (int i = 0; i < grp_id.size(); ++i) {
		group_id = grp_vec.array() == grp_id[i];
		mn_size = group_id.count();
    Eigen::VectorXd mn_local(mn_size);
		for (int j = 0, k = 0; j < num_coef; ++j) {
			if (group_id[j]) {
				mn_local[k++] = abs(coef[j]);
			}
		}
		group_rate[i] = gamma_rand(prior_shape + mn_size, 1 / (prior_rate + mn_local.lpNorm<1>()), rng);
  }
}

// Log-density for Gamma shape hyperparameter in GDP
// 
// @param cand
// @param coef
// @param rate
inline double gdp_logdens_shape(double cand, Eigen::Ref<Eigen::VectorXd> coef, double rate) {
	int num_coef = coef.size();
	return num_coef * (log(1 - cand) - log(cand)) - (coef.cwiseAbs().array() / rate).log1p().sum() / cand;
}

inline double gdp_logdens_rate(double cand, Eigen::Ref<Eigen::VectorXd> coef, double shape) {
	int num_coef = coef.size();
	return num_coef * (log(cand) - log(1 - cand)) - (shape + 1) * (coef.cwiseAbs().array() * cand / (1 - cand)).log1p().sum();
}

// Griddy Gibbs for Hyperparameter of Gamma Prior in GDP
// 
// @param gamma_shape
// @param grid_size
// @param coef
// @param rng
inline void gdp_shape_griddy(double& shape_hyper, double rate_hyper, int grid_size, Eigen::Ref<Eigen::VectorXd> coef, BVHAR_BHRNG& rng) {
	Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(grid_size + 2, 0.0, 1.0).segment(1, grid_size);
	Eigen::VectorXd log_wt(grid_size);
	for (int i = 0; i < grid_size; ++i) {
		log_wt[i] = gdp_logdens_shape(grid[i], coef, rate_hyper);
	}
	Eigen::VectorXd weight = (log_wt.array() - log_wt.maxCoeff()).exp();
	weight /= weight.sum();
	int id = cat_rand(weight, rng);
	shape_hyper = (1 - grid[id]) / grid[id];
}

inline void gdp_rate_griddy(double& rate_hyper, double shape_hyper, int grid_size, Eigen::Ref<Eigen::VectorXd> coef, BVHAR_BHRNG& rng) {
	Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(grid_size + 2, 0.0, 1.0).segment(1, grid_size);
	Eigen::VectorXd log_wt(grid_size);
	for (int i = 0; i < grid_size; ++i) {
		log_wt[i] = gdp_logdens_rate(grid[i], coef, shape_hyper);
	}
	Eigen::VectorXd weight = (log_wt.array() - log_wt.maxCoeff()).exp();
	weight /= weight.sum();
	int id = cat_rand(weight, rng);
	rate_hyper = (1 - grid[id]) / grid[id];
}

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_MISC_GDP_HELPER_H