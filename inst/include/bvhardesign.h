#ifndef BVHARDESIGN_H
#define BVHARDESIGN_H

#include <RcppEigen.h>

Eigen::MatrixXd build_y0(Eigen::MatrixXd y, int var_lag, int index);

Eigen::MatrixXd build_design(Eigen::MatrixXd y, int var_lag, bool include_mean);

Eigen::MatrixXd scale_har(int dim, int week, int month, bool include_mean);

namespace bvhar {

inline Eigen::MatrixXd build_ydummy(int p, const Eigen::VectorXd& sigma, double lambda,
																		const Eigen::VectorXd& daily, const Eigen::VectorXd weekly, const Eigen::VectorXd& monthly,
																		bool include_mean) {
  int dim = sigma.size();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim * p + dim + 1, dim); // Yp
  // first block------------------------
  res.block(0, 0, dim, dim).diagonal() = daily.array() * sigma.array(); // deltai * sigma or di * sigma
  if (p > 1) {
    // avoid error when p = 1
    res.block(dim, 0, dim, dim).diagonal() = weekly.array() * sigma.array(); // wi * sigma
    res.block(2 * dim, 0, dim, dim).diagonal() = monthly.array() * sigma.array(); // mi * sigma
  }
  // second block-----------------------
  res.block(dim * p, 0, dim, dim).diagonal() = sigma;
  if (include_mean) {
    return res;
  }
  return res.topRows(dim * p + dim);
}

inline Eigen::MatrixXd build_xdummy(Eigen::VectorXd lag_seq, double lambda, Eigen::VectorXd sigma, double eps, bool include_mean) {
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