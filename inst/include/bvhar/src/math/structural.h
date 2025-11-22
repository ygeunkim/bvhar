#ifndef BVHAR_MATH_STRUCTURAL_H
#define BVHAR_MATH_STRUCTURAL_H

// #include "../core/eigen.h"
#include "../core/common.h"

namespace baecon {
namespace bvhar {

// Build coefficient in VAR(1) companion form of VAR(p)
// 
// @param coef_mat VAR without constant coefficient matrix form
// 
inline Eigen::MatrixXd build_companion(Eigen::Ref<const Eigen::MatrixXd> coef_mat) {
	int dim = coef_mat.cols();
	int dim_design = coef_mat.rows();
	Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim_design, dim_design);
	res.topRows(dim) = coef_mat.transpose();
	res.bottomLeftCorner(dim_design - dim, dim_design - dim).setIdentity();
	return res;
}

inline Eigen::MatrixXd harx_to_var(Eigen::Ref<Eigen::MatrixXd> coef_mat, Eigen::Ref<Eigen::MatrixXd> har_trans) {
	int dim = coef_mat.cols();
	int dim_design = har_trans.cols();
	int dim_exogen_design = coef_mat.rows() - har_trans.rows();
	if (dim_exogen_design <= 0) {
		return har_trans.transpose() * coef_mat;
	}
	int dim_har = coef_mat.rows() - dim_exogen_design;
	Eigen::MatrixXd var_coef(dim_design + dim_exogen_design, dim); // rbind(C^T Phi, B)
	var_coef.topRows(dim_design) = har_trans.transpose() * coef_mat.topRows(dim_har);
	var_coef.bottomRows(dim_exogen_design) = coef_mat.bottomRows(dim_exogen_design);
	return var_coef;
}

// Characteristic polynomial for stability
// 
// @param var_mat VAR(1) form coefficient matrix
// 
inline Eigen::VectorXd root_unitcircle(Eigen::Ref<Eigen::MatrixXd> var_mat) {
	Eigen::VectorXcd eigenvals = var_mat.eigenvalues();
	return eigenvals.cwiseAbs();
}

// Check if the coefficient is stable
inline bool is_stable(Eigen::Ref<const Eigen::MatrixXd> coef_mat, double threshold) {
	Eigen::MatrixXd companion_mat = build_companion(coef_mat);
	Eigen::VectorXd stableroot = root_unitcircle(companion_mat);
	return stableroot.maxCoeff() < threshold;
}

// Check if the coefficient is stable
inline bool is_stable(Eigen::Ref<const Eigen::MatrixXd> coef_mat, double threshold, Eigen::Ref<const Eigen::MatrixXd> har_trans) {
	Eigen::MatrixXd companion_mat = build_companion(har_trans.transpose() * coef_mat);
	Eigen::VectorXd stableroot = root_unitcircle(companion_mat);
	return stableroot.maxCoeff() < threshold;
}

inline Eigen::MatrixXd convert_var_to_vma(Eigen::Ref<Eigen::MatrixXd> var_coef, int var_lag, int lag_max) {
  int dim = var_coef.cols(); // m
  if (lag_max < 1) {
    BVHAR_STOP("'lag_max' must larger than 0");
  }
  int ma_rows = dim * (lag_max + 1);
  int num_full_arows = ma_rows;
  if (lag_max < var_lag) {
    num_full_arows = dim * var_lag; // for VMA coefficient q < VAR(p)
  }
  Eigen::MatrixXd FullA = Eigen::MatrixXd::Zero(num_full_arows, dim); // same size with VMA coefficient matrix
  FullA.topRows(dim * var_lag) = var_coef.topRows(dim * var_lag); // fill first mp row with VAR coefficient matrix
  // Eigen::MatrixXd Im = Eigen::MatrixXd::Identity(dim, dim); // identity matrix
  Eigen::MatrixXd ma = Eigen::MatrixXd::Zero(ma_rows, dim); // VMA [W1^T, W2^T, ..., W(lag_max)^T]^T, ma_rows = m * lag_max
  ma.topRows(dim) = Eigen::MatrixXd::Identity(dim, dim); // W0 = Im
  ma.middleRows(dim, dim) = FullA.topRows(dim) * ma.topRows(dim); // W1^T = B1^T * W1^T
  if (lag_max == 1) {
    return ma;
  }
  for (int i = 2; i < (lag_max + 1); i++) { // from W2: m-th row
    for (int k = 0; k < i; k++) {
      ma.middleRows(i * dim, dim) += FullA.middleRows(k * dim, dim) * ma.middleRows((i - k - 1) * dim, dim); // Wi = sum(W(i - k)^T * Bk^T)
    }
  }
  return ma;
}

inline Eigen::MatrixXd convert_vma_ortho(Eigen::Ref<Eigen::MatrixXd> var_coef, Eigen::Ref<Eigen::MatrixXd> var_covmat, int var_lag, int lag_max) {
  int dim = var_covmat.cols(); // num_rows = num_cols
  if ((dim != var_covmat.rows()) && (dim != var_coef.cols())) {
    BVHAR_STOP("Wrong covariance matrix format: `var_covmat`.");
  }
  if ((var_coef.rows() != var_lag * dim + 1) && (var_coef.rows() != var_lag * dim)) {
    BVHAR_STOP("Wrong VAR coefficient format: `var_coef`.");
  }
  Eigen::MatrixXd ma = convert_var_to_vma(var_coef, var_lag, lag_max);
  Eigen::MatrixXd res(ma.rows(), dim);
  Eigen::LLT<Eigen::MatrixXd> lltOfcovmat(Eigen::Map<Eigen::MatrixXd>(var_covmat.data(), dim, dim)); // cholesky decomposition for Sigma
  Eigen::MatrixXd chol_covmat = lltOfcovmat.matrixU();
  for (int i = 0; i < lag_max + 1; i++) {
    res.middleRows(i * dim, dim) = chol_covmat * ma.middleRows(i * dim, dim);
  }
  return res;
}

inline Eigen::MatrixXd compute_var_mse(Eigen::Ref<Eigen::MatrixXd> cov_mat, Eigen::Ref<Eigen::MatrixXd> var_coef,
																			 int var_lag, int step) {
	int dim = cov_mat.cols(); // dimension of time series
  Eigen::MatrixXd vma_mat = convert_var_to_vma(var_coef, var_lag, step);
  Eigen::MatrixXd innov_account = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::MatrixXd mse = Eigen::MatrixXd::Zero(dim * step, dim);
  for (int i = 0; i < step; i++) {
    innov_account += vma_mat.middleRows(i * dim, dim).transpose() * cov_mat * vma_mat.middleRows(i * dim, dim);
    mse.block(i * dim, 0, dim, dim) = innov_account;
  }
  return mse;
}

inline Eigen::MatrixXd convert_vhar_to_vma(Eigen::Ref<Eigen::MatrixXd> vhar_coef, Eigen::Ref<Eigen::MatrixXd> HARtrans_mat, int lag_max, int month) {
  int dim = vhar_coef.cols(); // dimension of time series
  // Eigen::MatrixXd coef_mat = HARtrans_mat.transpose() * vhar_coef; // bhat = tilde(T)^T * Phi
	Eigen::MatrixXd coef_mat = harx_to_var(vhar_coef, HARtrans_mat);
  if (lag_max < 1) {
    BVHAR_STOP("'lag_max' must larger than 0");
  }
  int ma_rows = dim * (lag_max + 1);
  int num_full_arows = ma_rows;
  if (lag_max < month) num_full_arows = month * dim; // for VMA coefficient q < VAR(p)
  Eigen::MatrixXd FullA = Eigen::MatrixXd::Zero(num_full_arows, dim); // same size with VMA coefficient matrix
  FullA.block(0, 0, month * dim, dim) = coef_mat.block(0, 0, month * dim, dim); // fill first mp row with VAR coefficient matrix
  // Eigen::MatrixXd Im = Eigen::MatrixXd::Identity(dim, dim); // identity matrix
  Eigen::MatrixXd ma = Eigen::MatrixXd::Zero(ma_rows, dim); // VMA [W1^T, W2^T, ..., W(lag_max)^T]^T, ma_rows = m * lag_max
  ma.topRows(dim) = Eigen::MatrixXd::Identity(dim, dim); // W0 = Im
  ma.middleRows(dim, dim) = FullA.topRows(dim) * ma.topRows(dim); // W1^T = B1^T * W1^T
  if (lag_max == 1) {
		return ma;
	}
  for (int i = 2; i < (lag_max + 1); i++) { // from W2: m-th row
    for (int k = 0; k < i; k++) {
      ma.middleRows(i * dim, dim) += FullA.middleRows(k * dim, dim) * ma.middleRows((i - k - 1) * dim, dim); // Wi = sum(W(i - k)^T * Bk^T)
    }
  }
  return ma;
}

inline Eigen::MatrixXd convert_vhar_vma_ortho(Eigen::Ref<Eigen::MatrixXd> vhar_coef, Eigen::Ref<Eigen::MatrixXd> vhar_covmat, Eigen::Ref<Eigen::MatrixXd> HARtrans_mat, int lag_max, int month) {
  int dim = vhar_covmat.cols(); // num_rows = num_cols
  if ((dim != vhar_covmat.rows()) && (dim != vhar_coef.cols())) {
    BVHAR_STOP("Wrong covariance matrix format: `vhar_covmat`.");
  }
  if ((vhar_coef.rows() != 3 * dim + 1) && (vhar_coef.rows() != 3 * dim)) {
    BVHAR_STOP("Wrong VAR coefficient format: `vhar_coef`.");
  }
  Eigen::MatrixXd ma = convert_vhar_to_vma(vhar_coef, HARtrans_mat, lag_max, month);
  Eigen::MatrixXd res(ma.rows(), dim);
  Eigen::LLT<Eigen::MatrixXd> lltOfcovmat(Eigen::Map<Eigen::MatrixXd>(vhar_covmat.data(), dim, dim)); // cholesky decomposition for Sigma
  Eigen::MatrixXd chol_covmat = lltOfcovmat.matrixU();
  for (int i = 0; i < lag_max + 1; i++) {
    res.middleRows(i * dim, dim) = chol_covmat * ma.middleRows(i * dim, dim);
  }
  return res;
}

inline Eigen::MatrixXd compute_vhar_mse(Eigen::Ref<Eigen::MatrixXd> cov_mat, Eigen::Ref<Eigen::MatrixXd> vhar_coef,
																				Eigen::Ref<Eigen::MatrixXd> har_trans, int month, int step) {
	int dim = cov_mat.cols(); // dimension of time series
  Eigen::MatrixXd vma_mat = convert_vhar_to_vma(vhar_coef, har_trans, month, step);
  Eigen::MatrixXd mse(dim * step, dim);
  mse.topLeftCorner(dim, dim) = cov_mat; // sig(y) = sig
  for (int i = 1; i < step; i++) {
    mse.middleRows(i * dim, dim) = mse.middleRows((i - 1) * dim, dim) + 
      vma_mat.middleRows(i * dim, dim).transpose() * cov_mat * vma_mat.middleRows(i * dim, dim);
  }
  return mse;
}

inline Eigen::MatrixXd compute_vma_fevd(Eigen::Ref<Eigen::MatrixXd> vma_coef, Eigen::Ref<Eigen::MatrixXd> cov_mat, bool normalize) {
  int dim = cov_mat.cols();
  // Eigen::MatrixXd vma_mat = VARcoeftoVMA(var_coef, var_lag, step);
  int step = vma_coef.rows() / dim; // h-step
  Eigen::MatrixXd innov_account = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::MatrixXd ma_prod(dim, dim);
  Eigen::MatrixXd numer = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::MatrixXd denom = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim * step, dim);
  Eigen::MatrixXd cov_diag = Eigen::MatrixXd::Zero(dim, dim);
  cov_diag.diagonal() = 1 / cov_mat.diagonal().cwiseSqrt().array(); // sigma_jj
  for (int i = 0; i < step; i++) {
      ma_prod = vma_coef.block(i * dim, 0, dim, dim).transpose() * cov_mat; // A * Sigma
      innov_account += ma_prod * vma_coef.block(i * dim, 0, dim, dim); // A * Sigma * A^T
      numer.array() += (ma_prod * cov_diag).array().square(); // sum(A * Sigma)_ij / sigma_jj^2
      denom.diagonal() = 1 / innov_account.diagonal().array(); // sigma_jj^(-1) / sum(A * Sigma * A^T)_jj
      res.block(i * dim, 0, dim, dim) = denom * numer; // sigma_jj^(-1) sum(A * Sigma)_ij / sum(A * Sigma * A^T)_jj
  }
  if (normalize) {
      res.array().colwise() /= res.rowwise().sum().array();
  }
  return res;
}

inline Eigen::MatrixXd compute_sp_index(Eigen::Ref<Eigen::MatrixXd> fevd) {
  return fevd.bottomRows(fevd.cols()) * 100;
}

inline Eigen::VectorXd compute_to(Eigen::Ref<Eigen::MatrixXd> spillover) {
  Eigen::MatrixXd diag_mat = spillover.diagonal().asDiagonal();
  return (spillover - diag_mat).colwise().sum();
}

inline Eigen::VectorXd compute_from(Eigen::Ref<Eigen::MatrixXd> spillover) {
  Eigen::MatrixXd diag_mat = spillover.diagonal().asDiagonal();
  return (spillover - diag_mat).rowwise().sum();
}

inline double compute_tot(Eigen::Ref<Eigen::MatrixXd> spillover) {
  Eigen::MatrixXd diag_mat = spillover.diagonal().asDiagonal();
  return (spillover - diag_mat).sum() / spillover.cols();
}

inline Eigen::MatrixXd compute_net(Eigen::Ref<Eigen::MatrixXd> spillover) {
  return (spillover.transpose() - spillover) / spillover.cols();
}

} // namespace bvhar
} // namespace baecon

#endif // BVHAR_MATH_STRUCTURAL_H