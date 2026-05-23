#ifndef BVHAR_BAYES_MISC_NG_HELPER_H
#define BVHAR_BAYES_MISC_NG_HELPER_H

#include "./helper.h"

namespace baecon {
namespace bvhar {

// Generating local shrinkage of Normal-Gamma prior
// 
// @param local_param local shrinkage
// @param shape Gamma prior shape
// @param coef Coefficients vector
// @param global_param Global shrinkage
// @param rng boost rng
inline void ng_local_sparsity(Eigen::VectorXd& local_param, double& shape,
										 					Eigen::Ref<Eigen::VectorXd> coef, Eigen::Ref<const Eigen::VectorXd> global_param,
										 					BVHAR_BHRNG& rng) {
	for (int i = 0; i < coef.size(); ++i) {
		local_param[i] = sim_gig(
			shape - .5,
			2 * shape / global_param[i],
			coef[i] * coef[i], rng
		);
		// cut_positive_param(local_param[i]);
	}
}
// overloading
inline void ng_local_sparsity(Eigen::VectorXd& local_param, Eigen::VectorXd& shape,
										 					Eigen::Ref<Eigen::VectorXd> coef, Eigen::Ref<const Eigen::VectorXd> global_param,
										 					BVHAR_BHRNG& rng) {
	for (int i = 0; i < coef.size(); ++i) {
		local_param[i] = sim_gig(
			shape[i] - .5,
			2 * shape[i] / global_param[i],
			coef[i] * coef[i], rng
		);
		// cut_positive_param(local_param[i]);
	}
}

// Generating global shrinkage of Normal-Gamma prior
// 
// @param local_param local shrinkage
// @param shape Inverse Gamma prior shape
// @param rate Inverse Gamma prior scale
// @param coef Coefficients vector
// @param rng boost rng
inline double ng_global_sparsity(Eigen::Ref<const Eigen::VectorXd> local_param, double& hyper_gamma,
																 double& shape, double& scl, BVHAR_BHRNG& rng) {
	// return sqrt(1 / gamma_rand(
	// 	shape + local_param.size() * hyper_gamma,
	// 	1 / (hyper_gamma * local_param.squaredNorm() + scl),
	// 	rng
	// ));
	double tau = 1 / gamma_rand(
		shape + local_param.size() * hyper_gamma,
		1 / (hyper_gamma * local_param.lpNorm<1>() + scl),
		rng
	);
	// cut_positive_param(tau);
	return tau;
}
// overloading
inline double ng_global_sparsity(Eigen::Ref<const Eigen::VectorXd> local_param, Eigen::VectorXd& hyper_gamma,
																 double& shape, double& scl, BVHAR_BHRNG& rng) {
	// return sqrt(1 / gamma_rand(
	// 	shape + hyper_gamma.sum(),
	// 	1 / ((hyper_gamma.array() * local_param.array().square()).sum() + scl),
	// 	rng
	// ));
	double tau = 1 / gamma_rand(
		shape + hyper_gamma.sum(),
		1 / ((hyper_gamma.array() * local_param.array()).sum() + scl),
		rng
	);
	// cut_positive_param(tau);
	return tau;
}

// For MN structure
// @param group_param Group shrinkage
// @param grp_vec Group vector
// @param grp_id Unique group id
inline void ng_mn_sparsity(Eigen::VectorXd& group_param, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
													 Eigen::VectorXd& hyper_gamma, double& global_param, Eigen::VectorXd& local_param, double& shape, double& scl,
													 BVHAR_BHRNG& rng) {
  int num_grp = grp_id.size();
  int num_coef = local_param.size();
	Eigen::Array<bool, Eigen::Dynamic, 1> group_id;
  int mn_size = 0;
  for (int i = 0; i < num_grp; i++) {
		group_id = grp_vec.array() == grp_id[i];
		mn_size = group_id.count();
    Eigen::VectorXd mn_local(mn_size);
		for (int j = 0, k = 0; j < num_coef; ++j) {
			if (group_id[j]) {
				mn_local[k++] = local_param[j] / global_param;
			}
		}
		group_param[i] = ng_global_sparsity(mn_local, hyper_gamma[i], shape, scl, rng);
  }
}

// MH for shape parameter of Normal-Gamma Prior
inline double ng_shape_jump(double& gamma_hyper, Eigen::VectorXd& local_param,
														double global_param, double lognormal_sd, BVHAR_BHRNG& rng) {
  int num_coef = local_param.size();
	double cand = exp(log(gamma_hyper) + normal_rand(rng) * lognormal_sd);
	double log_ratio = log(cand) - log(gamma_hyper) + num_coef * (lgammafn(gamma_hyper) - lgammafn(cand));
	log_ratio += num_coef * (cand * log(cand) - gamma_hyper * log(gamma_hyper));
	log_ratio -= num_coef * (cand - gamma_hyper) * log(global_param);
	log_ratio += (cand - gamma_hyper) * local_param.array().log().sum() / 2;
	// log_ratio += (gamma_hyper - cand) * local_param.array().sum() / (2 * global_param);
	log_ratio += (gamma_hyper - cand) * local_param.array().sum() / global_param;
	if (log(1 - unif_rand(rng)) < std::min(log_ratio, 0.0)) { // unif_rand has [0, 1) -> change to (0, 1]
		return cand;
	}
	// double acc_ratio = (cand / gamma_hyper) * pow(gammafn(gamma_hyper) / gammafn(cand), num_coef);
	// acc_ratio *= pow(cand / (global_param * global_param), num_coef * cand);
	// acc_ratio *= pow(global_param * global_param / gamma_hyper, num_coef * gamma_hyper);
	// acc_ratio *= pow(local_param.prod(), cand - gamma_hyper);
	// acc_ratio *= exp((gamma_hyper - cand) * local_param.array().square().sum() / (global_param * global_param));
	// if (unif_rand(0, 1, rng) < std::min(acc_ratio, 1.0)) {
	// 	return cand;
	// }
	return gamma_hyper;
}
// 
inline void ng_mn_shape_jump(Eigen::VectorXd& gamma_hyper, Eigen::VectorXd& local_param,
														 Eigen::VectorXd& group_param, Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id,
														 double& global_param, double lognormal_sd, BVHAR_BHRNG& rng) {
  int num_coef = local_param.size();
	Eigen::Array<bool, Eigen::Dynamic, 1> group_id;
  int mn_size = 0;
  for (int i = 0; i < grp_id.size(); i++) {
		group_id = grp_vec.array() == grp_id[i];
		mn_size = group_id.count();
    Eigen::VectorXd mn_local(mn_size);
		for (int j = 0, k = 0; j < num_coef; ++j) {
			if (group_id[j]) {
				mn_local[k++] = local_param[j];
				// mn_local[k++] = local_param[j] * group_param[i] * global_param;
			}
		}
		gamma_hyper[i] = ng_shape_jump(gamma_hyper[i], mn_local, global_param * group_param[i], lognormal_sd, rng);
  }
}

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_MISC_NG_HELPER_H