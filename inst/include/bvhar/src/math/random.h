#ifndef BVHAR_MATH_RANDOM_H
#define BVHAR_MATH_RANDOM_H

#include "../core/common.h"
#include <vector>

namespace baecon {
namespace bvhar {

inline Eigen::MatrixXd sim_mgaussian_eigen(int num_sim, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sig, BVHAR_BHRNG& rng) {
	int dim = sig.cols();
  Eigen::MatrixXd standard_normal(num_sim, dim);
  Eigen::MatrixXd res(num_sim, dim); // result: each column indicates variable
  for (int i = 0; i < num_sim; ++i) {
    for (int j = 0; j < standard_normal.cols(); ++j) {
      standard_normal(i, j) = normal_rand(rng);
    }
  }
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sig); // Sigma = P Lambda P^T
	res = standard_normal * es.eigenvalues().cwiseSqrt().asDiagonal() * es.eigenvectors().transpose(); // epsilon(t) = Sigma^{1/2} Z(t)
  res.rowwise() += mu.transpose();
  return res;
}

inline Eigen::MatrixXd sim_mgaussian_chol(int num_sim, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sig, BVHAR_BHRNG& rng) {
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

inline Eigen::MatrixXd sim_mstudent_eigen(int num_sim, double df, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sig, BVHAR_BHRNG& rng) {
	int dim = sig.cols();
  Eigen::MatrixXd res = sim_mgaussian_eigen(num_sim, Eigen::VectorXd::Zero(dim), sig, rng);
	for (int i = 0; i < num_sim; ++i) {
    res.row(i) *= sqrt(df / chisq_rand(df, rng));
  }
  res.rowwise() += mu.transpose();
  return res;
}

inline Eigen::MatrixXd sim_mstudent_chol(int num_sim, double df, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sig, BVHAR_BHRNG& rng) {
	int dim = sig.cols();
  Eigen::MatrixXd res = sim_mgaussian_chol(num_sim, Eigen::VectorXd::Zero(dim), sig, rng);
	for (int i = 0; i < num_sim; ++i) {
    res.row(i) *= sqrt(df / chisq_rand(df, rng));
  }
  res.rowwise() += mu.transpose();
  return res;
}

#ifdef BVHAR_USE_RCPP

inline Eigen::MatrixXd sim_mgaussian_eigen(int num_sim, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sig) {
	int dim = sig.cols();
  Eigen::MatrixXd standard_normal(num_sim, dim);
  Eigen::MatrixXd res(num_sim, dim); // result: each column indicates variable
  for (int i = 0; i < num_sim; ++i) {
    for (int j = 0; j < standard_normal.cols(); ++j) {
      standard_normal(i, j) = norm_rand();
    }
  }
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sig); // Sigma = P Lambda P^T
	res = standard_normal * es.eigenvalues().cwiseSqrt().asDiagonal() * es.eigenvectors().transpose(); // epsilon(t) = Sigma^{1/2} Z(t)
  res.rowwise() += mu.transpose();
  return res;
}

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

inline Eigen::MatrixXd sim_mstudent_eigen(int num_sim, double df, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sig) {
	int dim = sig.cols();
  Eigen::MatrixXd res = sim_mgaussian_eigen(num_sim, Eigen::VectorXd::Zero(dim), sig);
	for (int i = 0; i < num_sim; ++i) {
    res.row(i) *= sqrt(df / chisq_rand(df));
  }
  res.rowwise() += mu.transpose();
  return res;
}

inline Eigen::MatrixXd sim_mstudent_chol(int num_sim, double df, const Eigen::VectorXd& mu, const Eigen::MatrixXd& sig) {
	int dim = sig.cols();
  Eigen::MatrixXd res = sim_mgaussian_chol(num_sim, Eigen::VectorXd::Zero(dim), sig);
	for (int i = 0; i < num_sim; ++i) {
    res.row(i) *= sqrt(df / chisq_rand(df));
  }
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

// Generate Lower Triangular Matrix of Wishart
// 
// This function generates \eqn{C = LP}.
// 
// @param mat_scale Scale matrix of Wishart
// @param shape Shape of Wishart
// @details
// This function generates Wishart random matrix.
inline Eigen::MatrixXd sim_wishart_tri(Eigen::Ref<Eigen::MatrixXd> mat_scale, double shape) {
	int dim = mat_scale.cols();
	if (shape <= dim - 1) {
    BVHAR_STOP("Wrong 'shape'. shape > dim - 1 must be satisfied.");
  }
  if (mat_scale.rows() != mat_scale.cols()) {
    BVHAR_STOP("Invalid 'mat_scale' dimension.");
  }
  if (dim != mat_scale.rows()) {
    BVHAR_STOP("Invalid 'mat_scale' dimension.");
  }
	Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim);
	for (int i = 0; i < dim; ++i) {
		mat_bartlett(i, i) = sqrt(chisq_rand(shape - i));
    for (int j = 0; j < i; ++j) {
      mat_bartlett(i, j) = norm_rand();
    }
  }
	Eigen::MatrixXd chol_scale = mat_scale.llt().matrixL();
  return chol_scale * mat_bartlett;
}

inline Eigen::MatrixXd sim_wishart(Eigen::MatrixXd mat_scale, double shape) {
	Eigen::MatrixXd chol_res = sim_wishart_tri(mat_scale, shape);
  return chol_res * chol_res.transpose();
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
    BVHAR_STOP("Wrong 'shape'. shape > dim - 1 must be satisfied.");
  }
  if (mat_scale.rows() != mat_scale.cols()) {
    BVHAR_STOP("Invalid 'mat_scale' dimension.");
  }
  if (dim != mat_scale.rows()) {
    BVHAR_STOP("Invalid 'mat_scale' dimension.");
  }
	Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim);
	// for (int i = 0; i < dim; ++i) {
	// 	mat_bartlett(i, i) = sqrt(chisq_rand(shape - i));
  //   for (int j = 0; j < i; ++j) {
  //     mat_bartlett(i, j) = norm_rand();
  //   }
  // }
	for (int j = 0; j < dim; ++j) {
		for (int i = 0; i < j; ++i) {
			mat_bartlett(i, j) = norm_rand();
		}
		mat_bartlett(j, j) = sqrt(chisq_rand(shape - dim + j + 1));
	}
  Eigen::MatrixXd chol_scale = mat_scale.llt().matrixL();
	return mat_bartlett.transpose().triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(chol_scale);
}

inline Eigen::MatrixXd sim_inv_wishart(const Eigen::MatrixXd& mat_scale, double shape) {
  Eigen::MatrixXd chol_res = sim_iw_tri(mat_scale, shape);
  return chol_res * chol_res.transpose();
	// return sim_wishart(mat_scale.inverse(), shape).inverse();
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
#endif

// Generate MN(M, U, V)
inline Eigen::MatrixXd sim_mn(const Eigen::MatrixXd& mat_mean, const Eigen::MatrixXd& mat_scale_u, const Eigen::MatrixXd& mat_scale_v,
															bool prec, BVHAR_BHRNG& rng) {
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

// Generate lower triangular of Wishart
inline Eigen::MatrixXd sim_wishart_tri(Eigen::Ref<Eigen::MatrixXd> mat_scale, double shape, BVHAR_BHRNG& rng) {
	int dim = mat_scale.cols();
	Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim);
	for (int i = 0; i < dim; ++i) {
		mat_bartlett(i, i) = sqrt(chisq_rand(shape - i, rng));
    for (int j = 0; j < i; ++j) {
      mat_bartlett(i, j) = normal_rand(rng);
    }
  }
	Eigen::MatrixXd chol_scale = mat_scale.llt().matrixL();
  return chol_scale * mat_bartlett;
}

inline Eigen::MatrixXd sim_wishart(Eigen::MatrixXd mat_scale, double shape, BVHAR_BHRNG& rng) {
	Eigen::MatrixXd chol_res = sim_wishart_tri(mat_scale, shape, rng);
  return chol_res * chol_res.transpose();
}

// Generate Lower Triangular Matrix of IW
inline Eigen::MatrixXd sim_iw_tri(const Eigen::MatrixXd& mat_scale, double shape, BVHAR_BHRNG& rng) {
	int dim = mat_scale.cols();
  Eigen::MatrixXd mat_bartlett = Eigen::MatrixXd::Zero(dim, dim); // upper triangular bartlett decomposition
  // // generate in row direction
  // for (int i = 0; i < dim; ++i) {
  //   mat_bartlett(i, i) = sqrt(bvhar::chisq_rand(shape - (double)i, rng)); // diagonal: qii^2 ~ chi^2(nu - i + 1)
  // }
  // for (int i = 0; i < dim - 1; ++i) {
  //   for (int j = i + 1; j < dim; ++j) {
  //     mat_bartlett(i, j) = normal_rand(rng); // upper triangular (j > i) ~ N(0, 1)
  //   }
  // }
	// for (int i = 0; i < dim; ++i) {
	// 	mat_bartlett(i, i) = sqrt(chisq_rand(shape - i, rng));
  //   for (int j = 0; j < i; ++j) {
  //     mat_bartlett(i, j) = normal_rand(rng);
  //   }
  // }
	for (int j = 0; j < dim; ++j) {
		for (int i = 0; i < j; ++i) {
			mat_bartlett(i, j) = normal_rand(rng);
		}
		mat_bartlett(j, j) = sqrt(chisq_rand(shape - dim + j + 1, rng));
	}
  Eigen::MatrixXd chol_scale = mat_scale.llt().matrixL();
	return mat_bartlett.transpose().triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(chol_scale);
}

// Generate MNIW(M, U, Psi, nu)
inline std::vector<Eigen::MatrixXd> sim_mn_iw(const Eigen::MatrixXd& mat_mean, const Eigen::MatrixXd& mat_scale_u,
																			 				const Eigen::MatrixXd& mat_scale, double shape, bool prec, BVHAR_BHRNG& rng) {
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
inline double sim_gig(double lambda, double psi, double chi, BVHAR_BHRNG& rng) {
	cut_param(psi);
	cut_param(chi);
	boost::random::generalized_inverse_gaussian_distribution<> rdist(lambda, psi, chi);
	return rdist(rng);
}

// Generate Inverse Gaussian Distribution
// This function generates one Inverse Gaussian random number with mu (mean) and lambda (shape).
inline double sim_invgauss(double mean, double shape, BVHAR_BHRNG& rng) {
	cut_param(mean);
	cut_param(shape);
	boost::random::inverse_gaussian_distribution<> rdist(mean, shape);
	return rdist(rng);
}

} //namespace bvhar
} // namespace baecon

#endif // BVHAR_MATH_RANDOM_H
