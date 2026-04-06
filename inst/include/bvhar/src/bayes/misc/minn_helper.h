#ifndef BVHAR_BAYES_MISC_MINN_HELPER_H_H
#define BVHAR_BAYES_MISC_MINN_HELPER_H_H

#include "./helper.h"

namespace baecon {
namespace bvhar {

// Numerically Stable Log Marginal Likelihood Excluding Constant Term
// 
// This function computes log of ML stable,
// excluding the constant term.
// 
// @param dim Dimension of the time series
// @param num_design The number of the data matrix, \eqn{n = T - p}
// @param prior_prec Prior precision of Matrix Normal distribution
// @param prior_scale Prior scale of Inverse-Wishart distribution
// @param mn_prec Posterior precision of Matrix Normal distribution
// @param iw_scale Posterior scale of Inverse-Wishart distribution
// @param posterior_shape Posterior shape of Inverse-Wishart distribution
inline double compute_logml(int dim, int num_design, Eigen::MatrixXd prior_prec, Eigen::MatrixXd prior_scale,
														Eigen::MatrixXd mn_prec, Eigen::MatrixXd iw_scale, int posterior_shape) {
  Eigen::LLT<Eigen::MatrixXd> lltOfmn(prior_prec.inverse());
  Eigen::MatrixXd chol_mn = lltOfmn.matrixL();
  Eigen::MatrixXd stable_mat_a = chol_mn.transpose() * (mn_prec - prior_prec) * chol_mn;
  Eigen::LLT<Eigen::MatrixXd> lltOfiw(prior_scale.inverse());
  Eigen::MatrixXd chol_iw = lltOfiw.matrixL();
  Eigen::MatrixXd stable_mat_b = chol_iw.transpose() * (iw_scale - prior_scale) * chol_iw;
  // eigenvalues
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_a(stable_mat_a);
  Eigen::VectorXd a_eigen = es_a.eigenvalues();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_b(stable_mat_b);
  Eigen::VectorXd b_eigen = es_b.eigenvalues();
  // sum of log(1 + eigenvalues)
  double a_term = a_eigen.array().log1p().sum();
  double b_term = b_eigen.array().log1p().sum();
  // result
  return - num_design / 2.0 * log(prior_scale.determinant()) - dim / 2.0 * a_term - posterior_shape / 2.0 * b_term;
}

// Log of Joint Posterior Density of Hyperparameters
// 
// This function computes the log of joint posterior density of hyperparameters.
// 
// @param cand_gamma Candidate value of hyperparameters following Gamma distribution
// @param cand_invgam Candidate value of hyperparameters following Inverse Gamma distribution
// @param dim Dimension of the time series
// @param num_design The number of the data matrix, \eqn{n = T - p}
// @param prior_prec Prior precision of Matrix Normal distribution
// @param prior_scale Prior scale of Inverse-Wishart distribution
// @param mn_prec Posterior precision of Matrix Normal distribution
// @param iw_scale Posterior scale of Inverse-Wishart distribution
// @param posterior_shape Posterior shape of Inverse-Wishart distribution
// @param gamma_shape Shape of hyperprior Gamma distribution
// @param gamma_rate Rate of hyperprior Gamma distribution
// @param invgam_shape Shape of hyperprior Inverse gamma distribution
// @param invgam_scl Scale of hyperprior Inverse gamma distribution
inline double jointdens_hyperparam(double cand_gamma, Eigen::VectorXd cand_invgam, int dim, int num_design,
                            			 Eigen::MatrixXd prior_prec, Eigen::MatrixXd prior_scale, int prior_shape,
                            			 Eigen::MatrixXd mn_prec, Eigen::MatrixXd iw_scale,
                            			 int posterior_shape, double gamma_shp, double gamma_rate, double invgam_shp, double invgam_scl) {
  double res = compute_logml(dim, num_design, prior_prec, prior_scale, mn_prec, iw_scale, posterior_shape);
  res += -dim * num_design / 2.0 * log(M_PI) +
    lmgammafn((prior_shape + num_design) / 2.0, dim) -
    lmgammafn(prior_shape / 2.0, dim); // constant term
  res += gamma_dens(cand_gamma, gamma_shp, 1 / gamma_rate, true); // gamma distribution
  for (int i = 0; i < cand_invgam.size(); i++) {
    res += invgamma_dens(cand_invgam[i], invgam_shp, invgam_scl, true); // inverse gamma distribution
  }
  return res;
}

// Generating contemporaneous lambda of Minnesota-SV
// 
// @param lambda lambda1 or lambda2
// @param shape Gamma prior shape
// @param rate Gamma prior rate
// @param coef_vec Coefficients vector
// @param coef_mean Prior mean
// @param coef_prec Prior precision matrix
// @param rng boost rng
inline void minnesota_lambda(double& lambda, double& shape, double& rate, Eigen::Ref<Eigen::VectorXd> coef,
														 Eigen::Ref<Eigen::VectorXd> coef_mean, Eigen::Ref<Eigen::VectorXd> coef_prec,
														 BVHAR_BHRNG& rng) {
	coef_prec.array() *= lambda;
	// double gig_chi = (coef - coef_mean).squaredNorm();
	double gig_chi = ((coef - coef_mean).array().square() * coef_prec.array()).sum();
	lambda = sim_gig(shape - coef.size() / 2, 2 * rate, gig_chi, rng);
	cut_param(lambda);
	coef_prec.array() /= lambda;
}

inline double minnesota_logdens_scl(double& cand, Eigen::Ref<Eigen::VectorXd> coef,
														 		 		Eigen::Ref<Eigen::VectorXd> coef_mean, Eigen::Ref<Eigen::VectorXd> coef_prec,
														 		 		Eigen::VectorXi& grp_vec, std::set<int>& grp_id) {
	double gaussian_kernel = 0;
	int mn_size = 0;
	for (int i = 0; i < coef.size(); ++i) {
		if (grp_id.find(grp_vec[i]) != grp_id.end()) {
			mn_size++;
			gaussian_kernel += (coef[i] - coef_mean[i]) * (coef[i] - coef_mean[i]) * coef_prec[i];
		}
	}
	return -(mn_size * log(cand) + gaussian_kernel / cand) / 2;
}

inline double minnesota_logdens_scl(double& cand, Eigen::Ref<Eigen::VectorXd> coef,
														 		 		Eigen::Ref<Eigen::VectorXd> coef_mean, Eigen::Ref<Eigen::VectorXd> coef_prec,
														 		 		Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id) {
	double gaussian_kernel = 0;
	int mn_size = 0;
	int num_coef = coef.size();
	Eigen::Array<bool, Eigen::Dynamic, 1> global_id;
	for (int i = 0; i < grp_id.size(); ++i) {
		global_id = grp_vec.array() == grp_id[i];
		mn_size += global_id.count();
		for (int j = 0; j < num_coef; ++j) {
			if (global_id[j]) {
				gaussian_kernel += (coef[j] - coef_mean[j]) * (coef[j] - coef_mean[j]) * coef_prec[j];
			}
		}
	}
	return -(mn_size * log(cand) + gaussian_kernel / cand) / 2;
}

inline void minnesota_nu_griddy(double& nu, int grid_size, Eigen::Ref<Eigen::VectorXd> coef,
																Eigen::Ref<Eigen::VectorXd> coef_mean, Eigen::Ref<Eigen::VectorXd> coef_prec,
																Eigen::VectorXi& grp_vec, std::set<int>& grp_id, BVHAR_BHRNG& rng) {
	Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(grid_size + 2, 0.0, 1.0).segment(1, grid_size);
	Eigen::VectorXd log_wt(grid_size);
	double old_nu = nu;
	for (int i = 0; i < grid_size; ++i) {
		log_wt[i] = minnesota_logdens_scl(grid[i], coef, coef_mean, coef_prec, grp_vec, grp_id);
	}
	Eigen::VectorXd weight = (log_wt.array() - log_wt.maxCoeff()).exp();
	weight /= weight.sum();
	nu = grid[cat_rand(weight, rng)];
	for (int i = 0; i < coef.size(); ++i) {
		if (grp_id.find(grp_vec[i]) != grp_id.end()) {
			coef_prec[i] *= old_nu / nu;
		}
	}
}

inline void minnesota_nu_griddy(double& nu, int grid_size, Eigen::Ref<Eigen::VectorXd> coef,
																Eigen::Ref<Eigen::VectorXd> coef_mean, Eigen::Ref<Eigen::VectorXd> coef_prec,
																Eigen::VectorXi& grp_vec, Eigen::VectorXi& grp_id, BVHAR_BHRNG& rng) {
	Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(grid_size + 2, 0.0, 1.0).segment(1, grid_size);
	Eigen::VectorXd log_wt(grid_size);
	double old_nu = nu;
	for (int i = 0; i < grid_size; ++i) {
		log_wt[i] = minnesota_logdens_scl(grid[i], coef, coef_mean, coef_prec, grp_vec, grp_id);
	}
	Eigen::VectorXd weight = (log_wt.array() - log_wt.maxCoeff()).exp();
	weight /= weight.sum();
	nu = grid[cat_rand(weight, rng)];
	int num_coef = coef.size();
	Eigen::Array<bool, Eigen::Dynamic, 1> global_id;
	for (int i = 0; i < grp_id.size(); ++i) {
		global_id = grp_vec.array() == grp_id[i];
		for (int j = 0; j < num_coef; ++j) {
			if (global_id[j]) {
				coef_prec[j] *= old_nu / nu;
			}
		}
	}
}

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_MISC_MINN_HELPER_H_H