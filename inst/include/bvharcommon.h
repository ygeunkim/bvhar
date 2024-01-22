#ifndef BVHARCOMMON_H
#define BVHARCOMMON_H

#include <RcppEigen.h>

typedef Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMajorMatrixXd;

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, 1> vectorize_eigen(const Eigen::MatrixBase<Derived>& x) {
	// should use x.eval() when x is expression such as block or transpose. Use matrix().eval() if array.
	return Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, 1>::Map(x.derived().data(), x.size());
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> unvectorize(const Eigen::MatrixBase<Derived>& x, int num_cols) {
	int num_rows = x.size() / num_cols;
	return Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Map(x.derived().data(), num_rows, num_cols);
}

// Eigen::MatrixXd kronecker_eigen(Eigen::MatrixXd x, Eigen::MatrixXd y);

inline Eigen::MatrixXd kronecker_eigen(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) {
  Eigen::MatrixXd res = Eigen::kroneckerProduct(x, y).eval();
  return res;
}

#endif