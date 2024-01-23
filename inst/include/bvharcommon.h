#ifndef BVHARCOMMON_H
#define BVHARCOMMON_H

#include <RcppEigen.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/beta_distribution.hpp>

namespace bvhar {

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

// Gamma function
inline double gammafn(double x) {
	return Rf_gammafn(x);
}

inline double mgammafn(double x, int p) {
  if (p < 1) {
    Rcpp::stop("'p' should be larger than or same as 1.");
  }
  if (x <= 0) {
    Rcpp::stop("'x' should be larger than 0.");
  }
  if (p == 1) {
    return gammafn(x);
  }
  if (2 * x < p) {
    Rcpp::stop("'x / 2' should be larger than 'p'.");
  }
  double res = pow(M_PI, p * (p - 1) / 4.0);
  for (int i = 0; i < p; i++) {
    res *= gammafn(x - i / 2.0); // x + (1 - j) / 2
  }
  return res;
}

// Density functions--------------------------
inline double lgammafn(double x) {
	return Rf_lgammafn(x);
}

inline double gamma_dens(double x, double shp, double scl, bool lg) {
	return Rf_dgamma(x, shp, scl, lg);
}

// RNG----------------------------------------
inline double bindom_rand(int n, double prob) {
	return Rf_rbinom(n, prob);
}

inline double normal_rand(boost::random::mt19937& rng) {
	boost::random::normal_distribution<> rdist(0.0, 1.0);
	return rdist(rng);
}

inline double chisq_rand(double df) {
	return Rf_rchisq(df);
}

inline double gamma_rand(double shp, double scl) {
	return Rf_rgamma(shp, scl); // 2nd: scale
}

inline double gamma_rand(double shp, double scl, boost::random::mt19937& rng) {
	boost::random::gamma_distribution<> rdist(shp, scl); // 2nd: scale
	return rdist(rng);
}

inline double ber_rand(double prob, boost::random::mt19937& rng) {
	boost::random::bernoulli_distribution<> rdist(prob); // Bernoulli supported -> use this instead of binomial
	return rdist(rng) * 1.0; // change to int later: now just use double to match Rf_rbinom
}

inline double unif_rand(double min, double max) {
	return Rf_runif(min, max);
}

inline double unif_rand(double min, double max, boost::random::mt19937& rng) {
	boost::random::uniform_real_distribution<> rdist(min, max);
	return rdist(rng);
}

inline double beta_rand(double s1, double s2) {
	return Rf_rbeta(s1, s2);
}

inline double beta_rand(double s1, double s2, boost::random::mt19937& rng) {
	boost::random::beta_distribution<> rdist(s1, s2);
	return rdist(rng);
}

} // namespace bvhar

#endif // BVHARCOMMON_H