#ifndef BVHARDESIGN_H
#define BVHARDESIGN_H

#include <RcppEigen.h>

namespace bvhar {

inline Eigen::MatrixXd build_y0(const Eigen::MatrixXd& y, int var_lag, int index) {
  int num_design = y.rows() - var_lag; // s = n - p
  int dim = y.cols(); // m: dimension of the multivariate time series
  Eigen::MatrixXd res(num_design, dim); // Yj (or Y0)
  for (int i = 0; i < num_design; i++) {
    res.row(i) = y.row(index + i - 1);
  }
  return res;
}

inline Eigen::MatrixXd build_x0(const Eigen::MatrixXd& y, int var_lag, bool include_mean) {
  int num_design = y.rows() - var_lag; // s = n - p
  int dim = y.cols(); // m: dimension of the multivariate time series
  int dim_design = dim * var_lag + 1; // k = mp + 1
  Eigen::MatrixXd res(num_design, dim_design); // X0 = [Yp, ... Y1, 1]: s x k
  for (int t = 0; t < var_lag; t++) {
    res.block(0, t * dim, num_design, dim) = build_y0(y, var_lag, var_lag - t); // Yp to Y1
  }
  if (!include_mean) {
    return res.block(0, 0, num_design, dim_design - 1);
  }
  for (int i = 0; i < num_design; i++) {
    res(i, dim_design - 1) = 1.0; // the last column for constant term
  }
  return res;
}

inline Eigen::MatrixXd build_vhar(int dim, int week, int month, bool include_mean) {
  // if (week > month) {
  //   Rcpp::stop("'month' should be larger than 'week'.");
  // }
  Eigen::MatrixXd HAR = Eigen::MatrixXd::Zero(3, month);
  Eigen::MatrixXd HARtrans(3 * dim + 1, month * dim + 1); // 3m x (month * m)
  Eigen::MatrixXd Im = Eigen::MatrixXd::Identity(dim, dim);
  HAR(0, 0) = 1.0;
  for (int i = 0; i < week; i++) {
    HAR(1, i) = 1.0 / week;
  }
  for (int i = 0; i < month; i++) {
    HAR(2, i) = 1.0 / month;
  }
  // T otimes Im
  HARtrans.block(0, 0, 3 * dim, month * dim) = Eigen::kroneckerProduct(HAR, Im).eval();
  HARtrans.block(0, month * dim, 3 * dim, 1) = Eigen::MatrixXd::Zero(3 * dim, 1);
  HARtrans.block(3 * dim, 0, 1, month * dim) = Eigen::MatrixXd::Zero(1, month * dim);
  HARtrans(3 * dim, month * dim) = 1.0;
  if (include_mean) {
    return HARtrans;
  }
  return HARtrans.block(0, 0, 3 * dim, month * dim);
}

inline Eigen::MatrixXd build_ydummy(int p, const Eigen::VectorXd& sigma, double lambda,
																		const Eigen::VectorXd& daily, const Eigen::VectorXd& weekly, const Eigen::VectorXd& monthly,
																		bool include_mean) {
  int dim = sigma.size();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim * p + dim + 1, dim); // Yp
  // first block------------------------
  res.block(0, 0, dim, dim).diagonal() = daily.array() * sigma.array() / lambda; // deltai * sigma or di * sigma
  if (p > 1) {
    // avoid error when p = 1
    res.block(dim, 0, dim, dim).diagonal() = weekly.array() * sigma.array() / lambda; // wi * sigma
    res.block(2 * dim, 0, dim, dim).diagonal() = monthly.array() * sigma.array() / lambda; // mi * sigma
  }
  // second block-----------------------
  res.block(dim * p, 0, dim, dim).diagonal() = sigma;
  if (include_mean) {
    return res;
  }
  return res.topRows(dim * p + dim);
}

inline Eigen::MatrixXd build_xdummy(const Eigen::VectorXd& lag_seq,
																		double lambda, const Eigen::VectorXd& sigma,
																		double eps, bool include_mean) {
  int dim = sigma.size();
  int var_lag = lag_seq.size();
  Eigen::MatrixXd Sig = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim * var_lag + dim + 1, dim * var_lag + 1);
  // first block------------------
  Eigen::MatrixXd Jp = Eigen::MatrixXd::Zero(var_lag, var_lag);
  Jp.diagonal() = lag_seq;
  Sig.diagonal() = sigma / lambda;
  res.block(0, 0, dim * var_lag, dim * var_lag) = Eigen::kroneckerProduct(Jp, Sig);
  // third block------------------
  res(dim * var_lag + dim, dim * var_lag) = eps;
  if (include_mean) {
    return res;
  }
  return res.block(0, 0, dim * var_lag + dim, dim * var_lag);
}

} // namespace bvhar

#endif // BVHARDESIGN_H