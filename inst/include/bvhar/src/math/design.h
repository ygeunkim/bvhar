#ifndef BVHAR_MATH_DESIGN_H
#define BVHAR_MATH_DESIGN_H

#include "../core/eigen.h"

namespace baecon {
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

inline Eigen::MatrixXd build_exogen0(int num_design, int var_lag, const Eigen::MatrixXd& exogen, int exogen_lag) {
	int x_dim = exogen.cols();
	Eigen::MatrixXd res(num_design, x_dim * (exogen_lag + 1));
	for (int i = 0; i < exogen_lag + 1; ++i) {
		res.middleCols(i * x_dim, x_dim) = exogen.middleRows(var_lag - i, num_design); // X(p + 1) to X(p + 1 - s)
	}
	return res;
}

inline Eigen::MatrixXd build_x0(const Eigen::MatrixXd& y, const Eigen::MatrixXd& exogen, int var_lag, int exogen_lag, bool include_mean) {
  int num_design = y.rows() - var_lag; // n = T - p
  int dim = y.cols();
	int x_dim = exogen.cols();
  // int dim_design = dim * var_lag + x_dim * exogen_lag + 1;
	int dim_endog = include_mean ? dim * var_lag + 1 : dim * var_lag;
	Eigen::MatrixXd res(num_design, dim_endog + x_dim * (exogen_lag + 1)); // X0 = [Yp, ... Y1, 1, X(p + 1), ..., X(p + 1 - s)]: n x (dim * lag + x_dim * (s + 1) + 1)
  for (int t = 0; t < var_lag; ++t) {
		res.middleCols(t * dim, dim) = y.middleRows(var_lag - t - 1, num_design); // Yp to Y1
  }
	if (include_mean) {
		res.col(dim * var_lag) = Eigen::VectorXd::Ones(num_design); // after endogenous
	}
	for (int t = 0; t < exogen_lag + 1; ++t) {
		// res.middleCols(dim_endog + t * x_dim, x_dim) = exogen.middleRows(var_lag - t, num_design); // X(p + 1) to X(p - s)
		res.middleCols(dim_endog + t * x_dim, x_dim) = exogen.middleRows(var_lag - t, num_design); // X(p + 1) to X(p + 1 - s)
  }
	// res.rightCols(x_dim * (exogen_lag + 1)) = build_exogen0(num_design, var_lag, exogen, exogen_lag);
  // if (!include_mean) {
  //   return res.leftCols(dim_design - 1);
  // }
	// res.col(dim_design - 1) = Eigen::VectorXd::Ones(num_design); // the last column for constant term
  return res;
}

inline Eigen::MatrixXd build_har_matrix(int week, int month) {
	Eigen::MatrixXd HAR = Eigen::MatrixXd::Zero(3, month);
  HAR(0, 0) = 1.0;
  for (int i = 0; i < week; ++i) {
    HAR(1, i) = 1.0 / week;
  }
  for (int i = 0; i < month; ++i) {
    HAR(2, i) = 1.0 / month;
  }
	return HAR;
}

inline Eigen::MatrixXd build_vhar(int dim, int week, int month, bool include_mean) {
  // if (week > month) {
  //   Rcpp::stop("'month' should be larger than 'week'.");
  // }
  // Eigen::MatrixXd HAR = Eigen::MatrixXd::Zero(3, month);
	Eigen::MatrixXd HAR = build_har_matrix(week, month);
  Eigen::MatrixXd HARtrans(3 * dim + 1, month * dim + 1); // 3m x (month * m)
  Eigen::MatrixXd Im = Eigen::MatrixXd::Identity(dim, dim);
  // T otimes Im
  HARtrans.block(0, 0, 3 * dim, month * dim) = Eigen::kroneckerProduct(HAR, Im).eval();
  HARtrans.block(0, month * dim, 3 * dim, 1) = Eigen::MatrixXd::Zero(3 * dim, 1);
  HARtrans.block(3 * dim, 0, 1, month * dim) = Eigen::MatrixXd::Zero(1, month * dim);
  HARtrans(3 * dim, month * dim) = 1.0;
  if (include_mean) {
    return HARtrans;
  }
  return HARtrans.topLeftCorner(3 * dim, month * dim);
}

/**
 * @brief Build Minnesota-type Group Matrix
 * 
 * @param lag 
 * @param dim 
 * @param group_type 1: no group, 2: short-run 3: long-run
 * @return Eigen::MatrixXi 
 */
inline Eigen::MatrixXi build_grpmat(int lag, int dim, int group_type = 3) {
	Eigen::MatrixXi grp_mat = Eigen::MatrixXi::Zero(lag * dim, dim);
	switch (group_type) {
		case 1: {
			grp_mat.setOnes();
			return grp_mat;
		}
		case 2: {
			grp_mat.topRows(dim).setIdentity();
			grp_mat.topRows(dim).array() += 1;
			for (int i = 1; i < lag; ++i) {
				grp_mat.middleRows(i * dim, dim).setOnes();
				grp_mat.middleRows(i * dim, dim).array() += i + 1;
			}
			return grp_mat;
		}
		case 3: {
			for (int i = 0; i < lag; ++i) {
				grp_mat.middleRows(i * dim, dim).setIdentity();
				grp_mat.middleRows(i * dim, dim).array() += 2 * i + 1;
			}
		}
	}
	return grp_mat;
}

inline Eigen::VectorXi build_own_id(int lag, int group_type = 3) {
	Eigen::VectorXi own_id = Eigen::VectorXi::LinSpaced(lag, 2, 2 * lag);
	if (group_type == 1) {
		own_id.resize(1);
		own_id = Eigen::VectorXi::Ones(1);
	} else if (group_type == 2) {
		own_id.resize(1);
		own_id = Eigen::VectorXi::Constant(1, 2);
	}
	return own_id;
}

inline Eigen::VectorXi build_cross_id(int lag, int group_type = 3) {
	Eigen::VectorXi cross_id = Eigen::VectorXi::LinSpaced(lag, 1, 2 * lag);
	if (group_type == 1) {
		cross_id.resize(1);
		cross_id = Eigen::VectorXi::Constant(1, 2);
	} else if (group_type == 2) {
		cross_id[0] = 1;
		int id_val = 3;
		for (int i = 1; i < lag; ++i) {
			cross_id[i] = id_val++;
		}
	}
	return cross_id;
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
} // namespace baecon

#endif // BVHAR_MATH_DESIGN_H