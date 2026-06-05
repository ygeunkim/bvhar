#ifndef BVHAR_BAYES_MISC_COEF_HELPER_H_H_H
#define BVHAR_BAYES_MISC_COEF_HELPER_H_H_H

#include "./helper.h"

namespace baecon {
namespace bvhar {

// Generating the Equation-wise Coefficients Vector and Contemporaneous Coefficients
// 
// This function generates j-th column of coefficients matrix and j-th row of impact matrix using precision sampler.
//
// @param x Design matrix of the system
// @param y Response vector of the system
// @param prior_mean Prior mean vector
// @param prior_prec Prior precision matrix
// @param innov_prec Stacked precision matrix of innovation
inline void draw_coef(Eigen::Ref<Eigen::VectorXd> coef, Eigen::Ref<const Eigen::MatrixXd> x, Eigen::Ref<const Eigen::VectorXd> y,
											Eigen::Ref<Eigen::VectorXd> prior_mean, Eigen::Ref<Eigen::VectorXd> prior_prec, BVHAR_BHRNG& rng) {
  int dim = prior_mean.size();
  Eigen::VectorXd res(dim);
  for (int i = 0; i < dim; i++) {
		res[i] = normal_rand(rng);
  }
	Eigen::LLT<Eigen::MatrixXd> llt_of_prec(
		(prior_prec.asDiagonal().toDenseMatrix() + x.transpose() * x).selfadjointView<Eigen::Lower>()
	);
	double temp_penalty = .0001;
	while (llt_of_prec.info() != Eigen::Success && temp_penalty < .1) {
		llt_of_prec.compute(
			(prior_prec.asDiagonal().toDenseMatrix() + x.transpose() * x + temp_penalty * Eigen::MatrixXd::Identity(dim, dim)).selfadjointView<Eigen::Lower>()
		);
		temp_penalty *= 2;
	}
	// do {
	// 	llt_of_prec.compute(
	// 		(prior_prec.asDiagonal().toDenseMatrix() + x.transpose() * x + temp_penalty * Eigen::MatrixXd::Identity(dim, dim)).selfadjointView<Eigen::Lower>()
	// 	);
	// 	temp_penalty *= 2;
	// } while (llt_of_prec.info() != Eigen::Success && temp_penalty < .1);
	// if (llt_of_prec.info() != Eigen::Success) {
	// 	eigen_assert("LLT failed in precision sampler.");
	// }
  Eigen::VectorXd post_mean = llt_of_prec.solve(prior_prec.cwiseProduct(prior_mean) + x.transpose() * y);
	coef = post_mean + llt_of_prec.matrixU().solve(res);
	// Eigen::MatrixXd pos_def = (prior_prec.asDiagonal().toDenseMatrix() + x.transpose() * x).selfadjointView<Eigen::Lower>();
	// double scl = pos_def.maxCoeff();
	// Eigen::LLT<Eigen::MatrixXd> llt_of_prec(pos_def / scl); // (L / sqrt(scl)) * (L^T / sqrt(scl))
  // Eigen::VectorXd post_mean = llt_of_prec.adjoint().solve((prior_prec.cwiseProduct(prior_mean) + x.transpose() * y) / scl); // (pos_def / scl) * mean = () / scl
	// coef = post_mean + llt_of_prec.matrixU().solve(res) / sqrt(scl); // (L^T / sqrt(scl))^(-1) * res = sqrt(scl) * (L^T)^(-1) * res
}

// SAVS Algorithm for shirnkage prior
// 
// Conduct SAVS for each draw.
// Use after varsv_regression() in the same loop.
// 
// @param coef non-zero coef
// @param x design matrix
inline void draw_savs(Eigen::Ref<Eigen::VectorXd> sparse_coef, Eigen::Ref<Eigen::VectorXd> coef, Eigen::Ref<Eigen::MatrixXd> x) {
	sparse_coef.setZero();
	// for (int i = 0; i < coef.size(); ++i) {
	// 	double mu_i = 1 / (coef[i] * coef[i]);
	// 	double abs_fit = abs(coef[i]) * x.col(i).squaredNorm();
	// 	if (abs_fit > mu_i) {
	// 		int alpha_sign = coef[i] >= 0 ? 1 : -1;
	// 		sparse_coef[i] = alpha_sign * (abs_fit - mu_i) / x.col(i).squaredNorm();
	// 	}
	// }
	Eigen::ArrayXd penalty_vec = 1 / coef.array().square();
	Eigen::ArrayXd coef_abs = coef.cwiseAbs().array();
	Eigen::ArrayXd x_norm = x.colwise().squaredNorm().transpose().array();
	Eigen::ArrayXd abs_fit = coef_abs * x_norm;
	// Eigen::ArrayXd sign_coef = coef.array() / coef_abs;
	// sparse_coef.array() = sign_coef * (abs_fit - penalty_vec).cwiseMax(0) / x_norm;
	sparse_coef.array() = coef.array() * (abs_fit - penalty_vec).cwiseMax(0) / abs_fit;
}

inline void draw_mn_savs(Eigen::Ref<Eigen::VectorXd> sparse_coef, Eigen::Ref<Eigen::VectorXd> coef, Eigen::Ref<Eigen::MatrixXd> x,
												 Eigen::Ref<const Eigen::VectorXd> prior_prec) {
	// sparse_coef.setZero();
	// Eigen::ArrayXd penalty_vec = 1 / (prior_prec.array() * coef.array().square());
	// Eigen::ArrayXd penalty_vec = prior_prec.array();
	Eigen::ArrayXd penalty_vec = prior_prec.array() / (coef.array().square());
	Eigen::ArrayXd coef_abs = coef.cwiseAbs().array();
	// Eigen::ArrayXd penalty_vec = prior_prec.array() / coef_abs;
	Eigen::ArrayXd x_norm = x.colwise().squaredNorm().array();
	Eigen::ArrayXd abs_fit = coef_abs * x_norm;
	// Eigen::ArrayXd sign_coef = coef.array() / coef_abs;
	// sparse_coef.array() = sign_coef * (abs_fit - penalty_vec).cwiseMax(0) / x_norm;
	sparse_coef.array() = coef.array() * (abs_fit - penalty_vec).cwiseMax(0) / abs_fit;
}

// Draw d_i in D from cholesky decomposition of precision matrix
// 
// @param diag_vec d_i vector
// @param shape IG shape of d_i prior
// @param scl IG scale of d_i prior
// @param ortho_latent Residual matrix of triangular equation
// @param rng boost rng
inline void reg_ldlt_diag(Eigen::Ref<Eigen::VectorXd> diag_vec, Eigen::VectorXd& shape, Eigen::VectorXd& scl,
													Eigen::Ref<const Eigen::MatrixXd> ortho_latent, BVHAR_BHRNG& rng) {
	int num_design = ortho_latent.rows();
	for (int i = 0; i < diag_vec.size(); ++i) {
		// diag_vec[i] = 1 / gamma_rand(
    //   shape[i] + num_design / 2,
		// 	1 / (scl[i] + ortho_latent.col(i).squaredNorm() / 2),
		// 	rng
    // );
		diag_vec[i] = 1 / gamma_rand(
      shape[i] + num_design / 2,
			1 / (scl[i] + ortho_latent.col(i).squaredNorm() / 2),
			rng
    );
	}
}

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_BAYES_MISC_COEF_HELPER_H_H_H