#ifndef BVHAR_MATH_RANDOM_H
#define BVHAR_MATH_RANDOM_H

#include "../core/common.h"
#include <vector>

namespace bvhar {

inline Eigen::MatrixXd sim_mgaussian_chol(int num_sim, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sig, BHRNG& rng) {
  int dim = sig.cols();
  Eigen::MatrixXd standard_normal(num_sim, dim);
  Eigen::MatrixXd res(num_sim, dim);
  for (int i = 0; i < num_sim; i++) {
    for (int j = 0; j < standard_normal.cols(); j++) {
      standard_normal(i, j) = normal_rand(rng);
    }
  }
  res = standard_normal * sig.llt().matrixU(); // use upper because now dealing with row vectors
  res.rowwise() += mu.transpose();
  return res;
}

#ifdef USE_RCPP

inline Eigen::MatrixXd sim_mgaussian_chol(int num_sim, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sig) {
  int dim = sig.cols();
  Eigen::MatrixXd standard_normal(num_sim, dim);
  Eigen::MatrixXd res(num_sim, dim); // result: each column indicates variable
  for (int i = 0; i < num_sim; i++) {
    for (int j = 0; j < standard_normal.cols(); j++) {
      standard_normal(i, j) = norm_rand();
    }
  }
  // Eigen::LLT<Eigen::MatrixXd> lltOfscale(sig);
  // Eigen::MatrixXd sig_sqrt = lltOfscale.matrixU(); // use upper because now dealing with row vectors
  res = standard_normal * sig.llt().matrixU(); // use upper because now dealing with row vectors
  res.rowwise() += mu.transpose();
  return res;
}

// Generate MN(M, U, V)
// @param mat_mean Mean matrix M
// @param mat_scale_u First scale matrix U
// @param mat_scale_v Second scale matrix V
// @param prec If true, use mat_scale_u as inverse of U.
inline Eigen::MatrixXd sim_mn(const Eigen::MatrixXd& mat_mean, const Eigen::MatrixXd& mat_scale_u, const Eigen::MatrixXd& mat_scale_v,
															bool prec) {
  int num_rows = mat_mean.rows();
  int num_cols = mat_mean.cols();
  Eigen::MatrixXd chol_scale_v = mat_scale_v.llt().matrixU(); // V = U_vTU_v
  Eigen::MatrixXd mat_norm(num_rows, num_cols); // standard normal
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      mat_norm(i, j) = norm_rand();
    }
  }
	if (prec) {
		// U^(-1) = LLT => U = LT^(-1) L^(-1)
		return mat_mean + mat_scale_u.llt().matrixU().solve(mat_norm * chol_scale_v); // M + LT^(-1) X U_v ~ MN(M, LT^(-1) L^(-1) = U, U_vT U_v = V)
	}
	Eigen::MatrixXd chol_scale_u = mat_scale_u.llt().matrixL(); // U = LLT
	return mat_mean + chol_scale_u * mat_norm * chol_scale_v; // M + L X U_v ~ MN(M, LLT = U, U_vT U_v = V)
}

// Generate Lower Triangular Matrix of IW
// 
// This function generates \eqn{A = L (Q^{-1})^T}.
// 
// @param mat_scale Scale matrix of IW
// @param shape Shape of IW
// @details
// This function is the internal function for IW sampling and MNIW sampling functions.
inline Eigen::MatrixXd sim_iw_tri(Eigen::MatrixXd mat_scale, double shape) {
  int dim = mat_scale.cols();
	if (shape <= dim - 1) {
    STOP("Wrong 'shape'. shape > dim - 1 must be satisfied.");
  }
  if (mat_scale.rows() != mat_scale.cols()) {
    STOP("Invalid 'mat_scale' dimension.");
  }
  if (dim != mat_scale.rows()) {
    STOP("Invalid 'mat_scale' dimension.");
  }
  Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim); // upper triangular bartlett decomposition
  // generate in row direction
  for (int i = 0; i < dim; i++) {
    mat_bartlett(i, i) = sqrt(bvhar::chisq_rand(shape - (double)i)); // diagonal: qii^2 ~ chi^2(nu - i + 1)
  }
  for (int i = 0; i < dim - 1; i ++) {
    for (int j = i + 1; j < dim; j++) {
      mat_bartlett(i, j) = norm_rand(); // upper triangular (j > i) ~ N(0, 1)
    }
  }
  Eigen::MatrixXd chol_scale = mat_scale.llt().matrixL();
  // return chol_scale * mat_bartlett.inverse().transpose(); // lower triangular
	return chol_scale * mat_bartlett.transpose().triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(dim, dim)); // lower triangular
}

inline Eigen::MatrixXd sim_inv_wishart(const Eigen::MatrixXd& mat_scale, double shape) {
  Eigen::MatrixXd chol_res = sim_iw_tri(mat_scale, shape);
  Eigen::MatrixXd res = chol_res * chol_res.transpose(); // dim x dim
  return res;
}

// Generate MNIW(M, U, Psi, nu)
// 
// @param mat_mean Mean matrix M
// @param mat_scale_u First scale matrix U
// @param mat_scale Inverse wishart scale matrix Psi
// @param shape Inverse wishart shape
// @param prec If true, use mat_scale_u as \eqn{U^{-1}}
inline std::vector<Eigen::MatrixXd> sim_mn_iw(const Eigen::MatrixXd& mat_mean, const Eigen::MatrixXd& mat_scale_u,
																			 				const Eigen::MatrixXd& mat_scale, double shape, bool prec) {
  Eigen::MatrixXd chol_res = sim_iw_tri(mat_scale, shape);
  Eigen::MatrixXd mat_scale_v = chol_res * chol_res.transpose();
	std::vector<Eigen::MatrixXd> res(2);
	res[0] = sim_mn(mat_mean, mat_scale_u, mat_scale_v, prec);
	res[1] = mat_scale_v;
	return res;
}

// Generate Lower Triangular Matrix of Wishart
// 
// This function generates \eqn{A = L (Q^{-1})^T}.
// 
// @param mat_scale Scale matrix of Wishart
// @param shape Shape of Wishart
// @details
// This function generates Wishart random matrix.
inline Eigen::MatrixXd sim_wishart(Eigen::MatrixXd mat_scale, double shape) {
  int dim = mat_scale.cols();
  if (shape <= dim - 1) {
    STOP("Wrong 'shape'. shape > dim - 1 must be satisfied.");
  }
  if (mat_scale.rows() != mat_scale.cols()) {
    STOP("Invalid 'mat_scale' dimension.");
  }
  if (dim != mat_scale.rows()) {
    STOP("Invalid 'mat_scale' dimension.");
  }
  Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim);
  for (int i = 0; i < dim; i++) {
    mat_bartlett(i, i) = sqrt(bvhar::chisq_rand(shape - (double)i));
  }
  for (int i = 1; i < dim; i++) {
    for (int j = 0; j < i; j++) {
      mat_bartlett(i, j) = norm_rand();
    }
  }
  Eigen::LLT<Eigen::MatrixXd> lltOfscale(mat_scale);
  Eigen::MatrixXd chol_scale = lltOfscale.matrixL();
  Eigen::MatrixXd chol_res = chol_scale * mat_bartlett;
  return chol_res * chol_res.transpose();
}
#endif

// Generate MN(M, U, V)
inline Eigen::MatrixXd sim_mn(const Eigen::MatrixXd& mat_mean, const Eigen::MatrixXd& mat_scale_u, const Eigen::MatrixXd& mat_scale_v,
															bool prec, BHRNG& rng) {
  int num_rows = mat_mean.rows();
  int num_cols = mat_mean.cols();
  Eigen::MatrixXd chol_scale_v = mat_scale_v.llt().matrixU(); // V = U_vTU_v
  Eigen::MatrixXd mat_norm(num_rows, num_cols); // standard normal
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      mat_norm(i, j) = normal_rand(rng);
    }
  }
	if (prec) {
		return mat_mean + mat_scale_u.llt().matrixU().solve(mat_norm * chol_scale_v); // M + U_u^(-1) X U_v ~ MN(M, U_u^(-1) U_u^(-1)T = U, U_vT U_v = V)
	}
	Eigen::MatrixXd chol_scale_u = mat_scale_u.llt().matrixL(); // U = LLT
	return mat_mean + chol_scale_u * mat_norm * chol_scale_v; // M + L X U_v ~ MN(M, LLT = U, U_vT U_v = V)
}

// Generate Lower Triangular Matrix of IW
inline Eigen::MatrixXd sim_iw_tri(const Eigen::MatrixXd& mat_scale, double shape, BHRNG& rng) {
  int dim = mat_scale.cols();
	if (shape <= dim - 1) {
    STOP("Wrong 'shape'. shape > dim - 1 must be satisfied.");
  }
  if (mat_scale.rows() != mat_scale.cols()) {
    STOP("Invalid 'mat_scale' dimension.");
  }
  if (dim != mat_scale.rows()) {
    STOP("Invalid 'mat_scale' dimension.");
  }
  Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim); // upper triangular bartlett decomposition
  // generate in row direction
  for (int i = 0; i < dim; i++) {
    mat_bartlett(i, i) = sqrt(bvhar::chisq_rand(shape - (double)i, rng)); // diagonal: qii^2 ~ chi^2(nu - i + 1)
  }
  for (int i = 0; i < dim - 1; i ++) {
    for (int j = i + 1; j < dim; j++) {
      mat_bartlett(i, j) = normal_rand(rng); // upper triangular (j > i) ~ N(0, 1)
    }
  }
  Eigen::MatrixXd chol_scale = mat_scale.llt().matrixL();
	return chol_scale * mat_bartlett.transpose().triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(dim, dim)); // lower triangular
}

// Generate MNIW(M, U, Psi, nu)
inline std::vector<Eigen::MatrixXd> sim_mn_iw(const Eigen::MatrixXd& mat_mean, const Eigen::MatrixXd& mat_scale_u,
																			 				const Eigen::MatrixXd& mat_scale, double shape, bool prec, BHRNG& rng) {
  Eigen::MatrixXd chol_res = sim_iw_tri(mat_scale, shape, rng);
  Eigen::MatrixXd mat_scale_v = chol_res * chol_res.transpose();
	std::vector<Eigen::MatrixXd> res(2);
	res[0] = sim_mn(mat_mean, mat_scale_u, mat_scale_v, prec, rng);
	res[1] = mat_scale_v;
	return res;
}

// Generate Generalized Inverse Gaussian Distribution
// 
// This function samples GIG(lambda, psi, chi) random variate.
// 
// @param lambda Index of modified Bessel function of third kind.
// @param psi Second parameter of GIG
// @param chi Third parameter of GIG
inline double sim_gig(double lambda, double psi, double chi, BHRNG& rng) {
	cut_param(psi);
	cut_param(chi);
	boost::random::generalized_inverse_gaussian_distribution<> rdist(lambda, psi, chi);
	return rdist(rng);
}

// Generate Inverse Gaussian Distribution
// This function generates one Inverse Gaussian random number with mu (mean) and lambda (shape).
inline double sim_invgauss(double mean, double shape, BHRNG& rng) {
	cut_param(mean);
	cut_param(shape);
	boost::random::inverse_gaussian_distribution<> rdist(mean, shape);
	return rdist(rng);
}

} //namespace bvhar

#endif // BVHAR_MATH_RANDOM_H
