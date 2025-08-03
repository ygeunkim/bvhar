#ifndef BVHAR_BAYES_MISC_DEPRECATED_DRAW_H
#define BVHAR_BAYES_MISC_DEPRECATED_DRAW_H

#include "../../../math/random.h"
#include "../../../math/structural.h"

namespace bvhar {

#ifdef BVHAR_USE_RCPP
// Generating the Diagonal Component of Cholesky Factor in SSVS Gibbs Sampler
// 
// In MCMC process of SSVS, this function generates the diagonal component \eqn{\Psi} from variance matrix
// 
// @param sse_mat The result of \eqn{Z_0^T Z_0 = (Y_0 - X_0 \hat{A})^T (Y_0 - X_0 \hat{A})}
// @param DRD Inverse of matrix product between \eqn{D_j} and correlation matrix \eqn{R_j}
// @param shape Gamma shape parameters for precision matrix
// @param rate Gamma rate parameters for precision matrix
// @param num_design The number of sample used, \eqn{n = T - p}
inline void ssvs_chol_diag(Eigen::VectorXd& chol_diag, Eigen::MatrixXd& sse_mat, Eigen::VectorXd& DRD,
													 Eigen::VectorXd& shape, Eigen::VectorXd& rate, int num_design, BVHAR_BHRNG& rng) {
  int dim = sse_mat.cols();
  int num_param = DRD.size();
  Eigen::MatrixXd inv_DRD = Eigen::MatrixXd::Zero(num_param, num_param);
  inv_DRD.diagonal() = 1 / DRD.array().square();
  Eigen::VectorXd sse_colvec(dim - 1); // sj = (s1j, ..., s(j-1, j)) from SSE
  shape.array() += (double)num_design / 2;
  rate[0] += sse_mat(0, 0) / 2;
  chol_diag[0] = sqrt(gamma_rand(shape[0], 1 / rate[0], rng)); // psi[11]^2 ~ Gamma(shape, rate)
  int block_id = 0;
  for (int j = 1; j < dim; j++) {
    sse_colvec.segment(0, j) = sse_mat.block(0, j, j, 1); // (s1j, ..., sj-1,j)
    rate[j] += (
      sse_mat(j, j) - 
        sse_colvec.segment(0, j).transpose() * 
        (sse_mat.topLeftCorner(j, j) + inv_DRD.block(block_id, block_id, j, j)).llt().solve(Eigen::MatrixXd::Identity(j, j)) * 
        sse_colvec.segment(0, j)
    ) / 2;
    chol_diag[j] = sqrt(gamma_rand(shape[j], 1 / rate[j], rng)); // psi[jj]^2 ~ Gamma(shape, rate)
    block_id += j;
  }
}

// Generating the Off-Diagonal Component of Cholesky Factor in SSVS Gibbs Sampler
// 
// In MCMC process of SSVS, this function generates the off-diagonal component \eqn{\Psi} of variance matrix
// 
// @param sse_mat The result of \eqn{Z_0^T Z_0 = (Y_0 - X_0 \hat{A})^T (Y_0 - X_0 \hat{A})}
// @param chol_diag Diagonal element of the cholesky factor
// @param DRD Inverse of matrix product between \eqn{D_j} and correlation matrix \eqn{R_j}
inline void ssvs_chol_off(Eigen::VectorXd& chol_off, Eigen::MatrixXd& sse_mat,
													Eigen::VectorXd& chol_diag, Eigen::VectorXd& DRD, BVHAR_BHRNG& rng) {
	int dim = sse_mat.cols();
  int num_param = DRD.size();
  Eigen::MatrixXd normal_variance(dim - 1, dim - 1);
  Eigen::VectorXd sse_colvec(dim - 1); // sj = (s1j, ..., s(j-1, j)) from SSE
  Eigen::VectorXd normal_mean(dim - 1);
  // Eigen::VectorXd res(num_param);
  Eigen::MatrixXd inv_DRD = Eigen::MatrixXd::Zero(num_param, num_param);
  inv_DRD.diagonal() = 1 / DRD.array().square();
  int block_id = 0;
  for (int j = 1; j < dim; j++) {
    sse_colvec.segment(0, j) = sse_mat.block(0, j, j, 1);
    normal_variance.topLeftCorner(j, j) = (sse_mat.topLeftCorner(j, j) + inv_DRD.block(block_id, block_id, j, j)).llt().solve(Eigen::MatrixXd::Identity(j, j));
    normal_mean.segment(0, j) = -chol_diag[j] * normal_variance.topLeftCorner(j, j) * sse_colvec.segment(0, j);
    chol_off.segment(block_id, j) = vectorize_eigen(sim_mgaussian_chol(1, normal_mean.segment(0, j), normal_variance.topLeftCorner(j, j), rng));
    block_id += j;
  }
}

// Filling Cholesky Factor Upper Triangular Matrix
// 
// This function builds a cholesky factor matrix \eqn{\Psi} (upper triangular) using diagonal component vector and off-diagonal component vector.
// 
// @param diag_vec Diagonal components
// @param off_diagvec Off-diagonal components
inline Eigen::MatrixXd build_chol(Eigen::VectorXd diag_vec, Eigen::VectorXd off_diagvec) {
  int dim = diag_vec.size();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim, dim);
  res.diagonal() = diag_vec; // psi
  int id = 0; // length of eta = m(m-1)/2
  // assign eta_j = (psi_1j, ..., psi_j-1,j)
  for (int j = 1; j < dim; j++) {
    for (int i = 0; i < j; i++) {
      res(i, j) = off_diagvec[id + i]; // assign i-th row = psi_ij
    }
    id += j;
  }
  return res;
}

inline Eigen::MatrixXd build_cov(Eigen::VectorXd diag_vec, Eigen::VectorXd off_diagvec) {
  int dim = diag_vec.size();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim, dim);
  res.diagonal() = diag_vec;
  int id = 0;
  for (int j = 1; j < dim; j++) {
    for (int i = 0; i < j; i++) {
      res(i, j) = off_diagvec[id + i]; // assign i-th row = psi_ij
      res(j, i) = res(i, j);
    }
    id += j;
  }
  return res;
}

// Generating Coefficient Vector in SSVS Gibbs Sampler
// 
// In MCMC process of SSVS, this function generates \eqn{\alpha_j} conditional posterior.
// 
// @param prior_mean The prior mean vector of the VAR coefficient vector
// @param prior_sd Diagonal prior sd matrix of the VAR coefficient vector
// @param XtX The result of design matrix arithmetic \eqn{X_0^T X_0}
// @param coef_ols OLS (MLE) estimator of the VAR coefficient
// @param chol_factor Cholesky factor of variance matrix
inline void ssvs_coef(Eigen::VectorXd& coef, Eigen::VectorXd& prior_mean, Eigen::VectorXd& prior_sd,
											Eigen::MatrixXd& XtX, Eigen::VectorXd& coef_ols,
											Eigen::MatrixXd& chol_factor, BVHAR_BHRNG& rng) {
	int num_coef = prior_sd.size();
  Eigen::MatrixXd scaled_xtx = kronecker_eigen((chol_factor * chol_factor.transpose()).eval(), XtX); // Sigma^(-1) = chol * chol^T
  Eigen::MatrixXd prior_prec = Eigen::MatrixXd::Zero(num_coef, num_coef);
  prior_prec.diagonal() = 1 / prior_sd.array().square();
  // Eigen::MatrixXd normal_variance = (scaled_xtx + prior_prec).llt().solve(Eigen::MatrixXd::Identity(num_coef, num_coef)); // Delta
  // Eigen::VectorXd normal_mean = normal_variance * (scaled_xtx * coef_ols + prior_prec * prior_mean); // mu
  // coef = vectorize_eigen(sim_mgaussian_chol(1, normal_mean, normal_variance, rng));
	Eigen::VectorXd standard_normal(num_coef);
	for (int i = 0; i < num_coef; i++) {
		standard_normal[i] = normal_rand(rng);
	}
	Eigen::MatrixXd normal_variance = scaled_xtx + prior_prec; // Delta^(-1)
	Eigen::LLT<Eigen::MatrixXd> llt_sig(normal_variance);
	Eigen::MatrixXd normal_mean = llt_sig.solve(scaled_xtx * coef_ols + prior_prec * prior_mean);
	coef = normal_mean + llt_sig.matrixU().solve(standard_normal);
}

// Building a Inverse Diagonal Matrix by Global and Local Hyperparameters
// 
// In MCMC process of Horseshoe, this function computes diagonal matrix \eqn{\Lambda_\ast^{-1}} defined by
// global and local sparsity levels.
// 
// @param global_hyperparam Global sparsity hyperparameters
// @param local_hyperparam Local sparsity hyperparameters
inline void build_shrink_mat(Eigen::MatrixXd& cov, Eigen::VectorXd& global_hyperparam, Eigen::Ref<Eigen::VectorXd> local_hyperparam) {
	cov.setZero();
  cov.diagonal() = 1 / (local_hyperparam.array() * global_hyperparam.array()).square();
	// cov.diagonal() = 1 / (local_hyperparam.array() * global_hyperparam.array());
	// cov.diagonal() = -2 * (local_hyperparam.array() * global_hyperparam.array()).log();
	// cov.diagonal() = (-2 * (local_hyperparam.array() * global_hyperparam.array()).log()).exp();
	// cov.diagonal() = cov.diagonal().array().exp();
}

// Generating the Coefficient Vector in Horseshoe Gibbs Sampler
// 
// In MCMC process of Horseshoe prior, this function generates the coefficients vector.
// 
// @param response_vec Response vector for vectorized formulation
// @param design_mat Design matrix for vectorized formulation
// @param shrink_mat Diagonal matrix made by global and local sparsity hyperparameters
inline void horseshoe_coef(Eigen::VectorXd& coef, Eigen::VectorXd& response_vec, Eigen::MatrixXd& design_mat,
                    			 double var, Eigen::MatrixXd& shrink_mat, BVHAR_BHRNG& rng) {
	int dim = coef.size();
	Eigen::VectorXd res(dim);
  for (int i = 0; i < dim; i++) {
		res[i] = normal_rand(rng);
  }
	Eigen::MatrixXd post_sig = shrink_mat / var + design_mat.transpose() * design_mat;
	Eigen::LLT<Eigen::MatrixXd> llt_sig(post_sig);
	Eigen::VectorXd post_mean = llt_sig.solve(design_mat.transpose() * response_vec);
	// coef = post_mean + llt_sig.matrixU().solve(design_mat.transpose() * response_vec);
	coef = post_mean + llt_sig.matrixU().solve(res);
}

// Generating the Coefficient Vector using Fast Sampling
// 
// In MCMC process of Horseshoe prior, this function generates the coefficients vector.
// 
// @param response_vec Response vector for vectorized formulation
// @param design_mat Design matrix for vectorized formulation
// @param shrink_mat Diagonal matrix made by global and local sparsity hyperparameters
inline void horseshoe_fast_coef(Eigen::VectorXd& coef, Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat,
												 				Eigen::MatrixXd shrink_mat, BVHAR_BHRNG& rng) {
  int num_coef = design_mat.cols(); // k^2 kp(+1)
  int num_sur = response_vec.size(); // nk-dim
  Eigen::MatrixXd sur_identity = Eigen::MatrixXd::Identity(num_sur, num_sur);
  Eigen::VectorXd u_vec = vectorize_eigen(sim_mgaussian_chol(1, Eigen::VectorXd::Zero(num_coef), shrink_mat, rng));
  Eigen::VectorXd delta_vec = vectorize_eigen(sim_mgaussian_chol(1, Eigen::VectorXd::Zero(num_sur), sur_identity, rng));
  Eigen::VectorXd nu = design_mat * u_vec + delta_vec;
  Eigen::VectorXd lin_solve = (design_mat * shrink_mat * design_mat.transpose() + sur_identity).llt().solve(
    response_vec - nu
  );
  coef = u_vec + shrink_mat * design_mat.transpose() * lin_solve;
}

// Generating the Coefficient Vector in Horseshoe Gibbs Sampler
// 
// In MCMC process of Horseshoe prior, this function generates the coefficients vector.
// 
// @param response_vec Response vector for vectorized formulation
// @param design_mat Design matrix for vectorized formulation
// @param shrink_mat Diagonal matrix made by global and local sparsity hyperparameters
inline void horseshoe_coef_var(Eigen::VectorXd& coef_var, Eigen::VectorXd& response_vec, Eigen::MatrixXd& design_mat,
															 Eigen::MatrixXd& shrink_mat, BVHAR_BHRNG& rng) {
  int dim = design_mat.cols();
  int sample_size = response_vec.size();
  Eigen::MatrixXd prec_mat = (design_mat.transpose() * design_mat + shrink_mat).llt().solve(
    Eigen::MatrixXd::Identity(dim, dim)
  );
  double scl = response_vec.transpose() * (Eigen::MatrixXd::Identity(sample_size, sample_size) - design_mat * prec_mat * design_mat.transpose()) * response_vec;
  coef_var[0] = 1 / gamma_rand(sample_size / 2, scl / 2, rng);
  coef_var.tail(dim) = vectorize_eigen(
    sim_mgaussian_chol(1, prec_mat * design_mat.transpose() * response_vec, coef_var[0] * prec_mat, rng)
  );
}

// Generating the Prior Variance Constant in Horseshoe Gibbs Sampler
// 
// In MCMC process of Horseshoe prior, this function generates the prior variance.
// 
// @param response_vec Response vector for vectorized formulation
// @param design_mat Design matrix for vectorized formulation
// @param coef_vec Coefficients vector
// @param shrink_mat Diagonal matrix made by global and local sparsity hyperparameters
// inline double horseshoe_var(Eigen::VectorXd& response_vec, Eigen::MatrixXd& design_mat, Eigen::MatrixXd& shrink_mat, boost::random::mt19937& rng) {
//   int sample_size = response_vec.size();
//   double scl = response_vec.transpose() * (Eigen::MatrixXd::Identity(sample_size, sample_size) - design_mat * shrink_mat * design_mat.transpose()) * response_vec;
//   return 1 / gamma_rand(sample_size / 2, 2 / scl, rng);
// }

inline double horseshoe_var(Eigen::VectorXd& response_vec, Eigen::MatrixXd& design_mat, Eigen::VectorXd& coef_vec, Eigen::MatrixXd& shrink_mat, BVHAR_BHRNG& rng) {
  return 1 / gamma_rand(
		design_mat.size() / 2,
		2 / ((response_vec - design_mat * coef_vec).squaredNorm() + coef_vec.transpose() * shrink_mat * coef_vec), rng
	);
}
#endif

} // namespace bvhar

#endif // BVHAR_BAYES_MISC_DEPRECATED_DRAW_H