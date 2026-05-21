#ifndef BVHAR_BAYES_MISC_DL_HELPER_H
#define BVHAR_BAYES_MISC_DL_HELPER_H

#include "./helper.h"

namespace baecon {
namespace bvhar {

// Generating Latent Scaling Factor of Dirichlet-Laplace Prior
// 
// @param latent_param Scaling factor psi
// @param local_param Local sparsity level
// @param glob_param Global sparsity level
// @param coef_vec Coefficients vector
// @param rng boost rng
inline void dl_latent(Eigen::VectorXd& latent_param, Eigen::Ref<const Eigen::VectorXd> local_param,
									 		Eigen::Ref<Eigen::VectorXd> coef_vec, BVHAR_BHRNG& rng) {
	// int num_alpha = latent_param.size();
	for (int i = 0; i < latent_param.size(); ++i) {
		// latent_param[i] = sim_gig(
		// 	1, .5,
		// 	1, coef_vec[i] * coef_vec[i] / (local_param[i] * local_param[i]), rng
		// )[0];
		latent_param[i] = sim_invgauss(local_param[i] / abs(coef_vec[i]), 1, rng);
		// cut_positive_param(latent_param[i]);
		// latent_param[i] = sim_invgauss(local_param[i] / abs(coef_vec[i]), 1, rng);
		// latent_param[i] = abs(coef_vec[i]) / sim_invgauss(local_param[i], abs(coef_vec[i]), rng);
	}
}

// Generating Local Parameter of Dirichlet-Laplace Prior
// 
// @param local_param Local sparsity level
// @param dir_concent Hyperparameter of Dirichlet prior
// @param coef Coefficients vector
// @param rng boost rng
inline void dl_local_sparsity(Eigen::VectorXd& local_param, double& dir_concen,
															Eigen::Ref<const Eigen::VectorXd> coef, BVHAR_BHRNG& rng) {
	for (int i = 0; i < coef.size(); ++i) {
		local_param[i] = sim_gig(dir_concen - 1, 1, 2 * abs(coef[i]), rng);
	}
	local_param /= local_param.sum();
}

// Generating Global Parameter of Dirichlet-Laplace Prior
// 
// @param local_param Local sparsity level
// @param dir_concent Hyperparameter of Dirichlet prior
// @param coef Coefficients vector
// @param rng boost rng
inline double dl_global_sparsity(Eigen::Ref<const Eigen::VectorXd> local_param, double& dir_concen,
										 						 Eigen::Ref<Eigen::VectorXd> coef, BVHAR_BHRNG& rng) {
	// return sim_gig(1, coef.size() * (dir_concen - 1), 1, 2 * (coef.cwiseAbs().array() / local_param.array()).sum(), rng)[0];
	double tau = sim_gig(coef.size() * (dir_concen - 1), 1, 2 * (coef.cwiseAbs().array() / local_param.array()).sum(), rng);
	// cut_positive_param(tau);
	return tau;
}

// Generating Group Parameter of Dirichlet-Laplace Prior
// 
// @param group_param Group shrinkage
// @param grp_vec Group vector
// @param grp_id Unique group id
// 
// @param rng boost rng
inline void dl_mn_sparsity(Eigen::VectorXd& group_param, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
													 double& global_param, Eigen::VectorXd& local_param, double& shape, double& rate,
													 Eigen::Ref<Eigen::VectorXd> coef_vec, BVHAR_BHRNG& rng) {
	Eigen::Array<bool, Eigen::Dynamic, 1> group_id;
  int mn_size = 0;
  for (int i = 0; i < grp_id.size(); i++) {
		group_id = grp_vec.array() == grp_id[i];
		mn_size = group_id.count();
    // Eigen::VectorXd mn_coef(mn_size);
    // Eigen::VectorXd mn_local(mn_size);
		Eigen::VectorXd mn_scl(mn_size);
		for (int j = 0, k = 0; j < coef_vec.size(); ++j) {
			if (group_id[j]) {
				// mn_coef[k] = coef_vec[j];
				// mn_local[k++] = global_param * local_param[j];
				mn_scl[k++] = abs(coef_vec[j]) / (global_param * local_param[j]);
			}
		}
		// group_param[i] = sim_gig(1, shape - mn_size, 2 * rate, 2 * mn_scl.sum(), rng)[0];
		group_param[i] = gamma_rand(
			shape + mn_size,
			1 / (rate + mn_scl.sum()),
			rng
		);
		// cut_positive_param(group_param[i]);
  }
}

// inline void dl_mn_sparsity(Eigen::VectorXd& group_param, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
// 													 double& global_param, Eigen::Ref<Eigen::VectorXd> local_param, Eigen::Ref<Eigen::VectorXd> latent_param,
// 													 double& shape, double& rate,
// 													 Eigen::Ref<Eigen::VectorXd> coef_vec, BVHAR_BHRNG& rng) {
// 	Eigen::Array<bool, Eigen::Dynamic, 1> group_id;
//   int mn_size = 0;
//   for (int i = 0; i < grp_id.size(); i++) {
// 		group_id = grp_vec.array() == grp_id[i];
// 		mn_size = group_id.count();
// 		Eigen::VectorXd mn_scl(mn_size);
// 		for (int j = 0, k = 0; j < coef_vec.size(); ++j) {
// 			if (group_id[j]) {
// 				mn_scl[k++] = coef_vec[j] * coef_vec[j] / (global_param * global_param * local_param[j] * local_param[j] * latent_param[j]);
// 			}
// 		}
// 		group_param[i] = sqrt(1 / gamma_rand(
// 			shape + mn_size / 2,
// 			1 / (rate + mn_scl.sum() / 2),
// 			rng
// 		));
//   }
// }

// Log-density for Dirichlet Hyperparameter in DL
// 
// Log density of Dirichlet hyperparameter ignoring constant term
// 
// @param cand Dirichlet hyperparameter
// @param local_param Local shrinkage
// @param global_param Global shrinkage
inline double dl_logdens_dir(double cand, Eigen::Ref<Eigen::VectorXd> local_param, double& global_param) {
	int num_coef = local_param.size();
	// return cand * (num_coef * log(global_param) - num_coef * log(2.0) + local_param.sum()) - lgammafn(num_coef * cand);
	// return (cand * num_coef - 1) * log(global_param) - num_coef * (cand * log(2.0) - lgammafn(cand)) + (cand - 1) * local_param.array().log().sum();
	return cand * (num_coef * (log(global_param) - log(2.0)) + local_param.array().log().sum()) - num_coef * lgammafn(cand);
	// return (cand * num_coef - 1) * log(global_param) - num_coef * lgammafn(cand) - lgammafn(num_coef * cand) + (cand - 1) * local_param.array().log().sum();
}

// Griddy Gibbs for Hyperparameter of Dirichlet Prior in DL
// 
// @param dir_concen Dirichlet hyperparameter
// @param grid_size Grid size
// @param local_param Local shrinkage
// @param global_param Global shrinkage
inline void dl_dir_griddy(double& dir_concen, int grid_size, Eigen::Ref<Eigen::VectorXd> local_param, double global_param, BVHAR_BHRNG& rng) {
	Eigen::VectorXd grid = 1 / local_param.size() < .5 ? Eigen::VectorXd::LinSpaced(grid_size, 1 / local_param.size(), .5) : Eigen::VectorXd::LinSpaced(grid_size, .5, 1 / local_param.size());
	// Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(grid_size, 1 / local_param.size(), 1);
	Eigen::VectorXd log_wt(grid_size);
	for (int i = 0; i < grid_size; ++i) {
		log_wt[i] = dl_logdens_dir(grid[i], local_param, global_param);
	}
	Eigen::VectorXd weight = (log_wt.array() - log_wt.maxCoeff()).exp(); // use log-sum-exp against overflow
	weight /= weight.sum();
	dir_concen = grid[cat_rand(weight, rng)];
}

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_MISC_DL_HELPER_H