#ifndef BVHARSTRUCTURAL_H
#define BVHARSTRUCTURAL_H

#include <RcppEigen.h>

namespace bvhar {

inline Eigen::MatrixXd convert_var_to_vma(Eigen::MatrixXd var_coef, int var_lag, int lag_max) {
  int dim = var_coef.cols(); // m
  if (lag_max < 1) {
    Rcpp::stop("'lag_max' must larger than 0");
  }
  int ma_rows = dim * (lag_max + 1);
  int num_full_arows = ma_rows;
  if (lag_max < var_lag) {
    num_full_arows = dim * var_lag; // for VMA coefficient q < VAR(p)
  }
  Eigen::MatrixXd FullA = Eigen::MatrixXd::Zero(num_full_arows, dim); // same size with VMA coefficient matrix
  FullA.block(0, 0, dim * var_lag, dim) = var_coef.block(0, 0, dim * var_lag, dim); // fill first mp row with VAR coefficient matrix
  Eigen::MatrixXd Im = Eigen::MatrixXd::Identity(dim, dim); // identity matrix
  Eigen::MatrixXd ma = Eigen::MatrixXd::Zero(ma_rows, dim); // VMA [W1^T, W2^T, ..., W(lag_max)^T]^T, ma_rows = m * lag_max
  ma.block(0, 0, dim, dim) = Im; // W0 = Im
  ma.block(dim, 0, dim, dim) = FullA.block(0, 0, dim, dim) * ma.block(0, 0, dim, dim); // W1^T = B1^T * W1^T
  if (lag_max == 1) {
    return ma;
  }
  for (int i = 2; i < (lag_max + 1); i++) { // from W2: m-th row
    for (int k = 0; k < i; k++) {
      ma.block(i * dim, 0, dim, dim) += FullA.block(k * dim, 0, dim, dim) * ma.block((i - k - 1) * dim, 0, dim, dim); // Wi = sum(W(i - k)^T * Bk^T)
    }
  }
  return ma;
}

inline Eigen::MatrixXd convert_vma_ortho(Eigen::MatrixXd var_coef, Eigen::MatrixXd var_covmat, int var_lag, int lag_max) {
  int dim = var_covmat.cols(); // num_rows = num_cols
  if ((dim != var_covmat.rows()) && (dim != var_coef.cols())) {
    Rcpp::stop("Wrong covariance matrix format: `var_covmat`.");
  }
  if ((var_coef.rows() != var_lag * dim + 1) && (var_coef.rows() != var_lag * dim)) {
    Rcpp::stop("Wrong VAR coefficient format: `var_coef`.");
  }
  Eigen::MatrixXd ma = convert_var_to_vma(var_coef, var_lag, lag_max);
  Eigen::MatrixXd res(ma.rows(), dim);
  Eigen::LLT<Eigen::MatrixXd> lltOfcovmat(Eigen::Map<Eigen::MatrixXd>(var_covmat.data(), dim, dim)); // cholesky decomposition for Sigma
  Eigen::MatrixXd chol_covmat = lltOfcovmat.matrixU();
  for (int i = 0; i < lag_max + 1; i++) {
    res.block(i * dim, 0, dim, dim) = chol_covmat * ma.block(i * dim, 0, dim, dim);
  }
  return res;
}

inline Eigen::MatrixXd convert_vhar_to_vma(Eigen::MatrixXd vhar_coef, Eigen::MatrixXd HARtrans_mat, int lag_max, int month) {
  int dim = vhar_coef.cols(); // dimension of time series
  Eigen::MatrixXd coef_mat = HARtrans_mat.transpose() * vhar_coef; // bhat = tilde(T)^T * Phi
  if (lag_max < 1) {
    Rcpp::stop("'lag_max' must larger than 0");
  }
  int ma_rows = dim * (lag_max + 1);
  int num_full_arows = ma_rows;
  if (lag_max < month) num_full_arows = month * dim; // for VMA coefficient q < VAR(p)
  Eigen::MatrixXd FullA = Eigen::MatrixXd::Zero(num_full_arows, dim); // same size with VMA coefficient matrix
  FullA.block(0, 0, month * dim, dim) = coef_mat.block(0, 0, month * dim, dim); // fill first mp row with VAR coefficient matrix
  Eigen::MatrixXd Im = Eigen::MatrixXd::Identity(dim, dim); // identity matrix
  Eigen::MatrixXd ma = Eigen::MatrixXd::Zero(ma_rows, dim); // VMA [W1^T, W2^T, ..., W(lag_max)^T]^T, ma_rows = m * lag_max
  ma.block(0, 0, dim, dim) = Im; // W0 = Im
  ma.block(dim, 0, dim, dim) = FullA.block(0, 0, dim, dim) * ma.block(0, 0, dim, dim); // W1^T = B1^T * W1^T
  if (lag_max == 1) return ma;
  for (int i = 2; i < (lag_max + 1); i++) { // from W2: m-th row
    for (int k = 0; k < i; k++) {
      ma.block(i * dim, 0, dim, dim) += FullA.block(k * dim, 0, dim, dim) * ma.block((i - k - 1) * dim, 0, dim, dim); // Wi = sum(W(i - k)^T * Bk^T)
    }
  }
  return ma;
}

inline Eigen::MatrixXd convert_vhar_vma_ortho(Eigen::MatrixXd vhar_coef, Eigen::MatrixXd vhar_covmat, Eigen::MatrixXd HARtrans_mat, int lag_max, int month) {
  int dim = vhar_covmat.cols(); // num_rows = num_cols
  if ((dim != vhar_covmat.rows()) && (dim != vhar_coef.cols())) {
    Rcpp::stop("Wrong covariance matrix format: `vhar_covmat`.");
  }
  if ((vhar_coef.rows() != 3 * dim + 1) && (vhar_coef.rows() != 3 * dim)) {
    Rcpp::stop("Wrong VAR coefficient format: `vhar_coef`.");
  }
  Eigen::MatrixXd ma = convert_vhar_to_vma(vhar_coef, HARtrans_mat, lag_max, month);
  Eigen::MatrixXd res(ma.rows(), dim);
  Eigen::LLT<Eigen::MatrixXd> lltOfcovmat(Eigen::Map<Eigen::MatrixXd>(vhar_covmat.data(), dim, dim)); // cholesky decomposition for Sigma
  Eigen::MatrixXd chol_covmat = lltOfcovmat.matrixU();
  for (int i = 0; i < lag_max + 1; i++) {
    res.block(i * dim, 0, dim, dim) = chol_covmat * ma.block(i * dim, 0, dim, dim);
  }
  return res;
}

inline Eigen::MatrixXd compute_vma_fevd(Eigen::MatrixXd vma_coef, Eigen::MatrixXd cov_mat, bool normalize) {
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

inline Eigen::MatrixXd compute_sp_index(Eigen::MatrixXd fevd) {
  return fevd.bottomRows(fevd.cols()) * 100;
}

inline Eigen::VectorXd compute_to(Eigen::MatrixXd spillover) {
  Eigen::MatrixXd diag_mat = spillover.diagonal().asDiagonal();
  return (spillover - diag_mat).colwise().sum();
}

inline Eigen::VectorXd compute_from(Eigen::MatrixXd spillover) {
  Eigen::MatrixXd diag_mat = spillover.diagonal().asDiagonal();
  return (spillover - diag_mat).rowwise().sum();
}

inline double compute_tot(Eigen::MatrixXd spillover) {
  Eigen::MatrixXd diag_mat = spillover.diagonal().asDiagonal();
  return (spillover - diag_mat).sum() / spillover.cols();
}

inline Eigen::MatrixXd compute_net(Eigen::MatrixXd spillover) {
  return (spillover.transpose() - spillover) / spillover.cols();
}

} // namespace bvhar

#endif // BVHARSTRUCTURAL_H