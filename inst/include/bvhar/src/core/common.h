#ifndef BVHAR_CORE_COMMON_H
#define BVHAR_CORE_COMMON_H

#include "./eigen.h"
#include "./spdlog.h"

namespace boost {

inline void assertion_failed(char const * expr, char const * function, char const * file, long line) {
	STOP("Boost assertion failed: %s in function %s at %s:%ld", expr, function, file, line);
}

inline void assertion_failed_msg(char const * expr, char const * msg, char const * function, char const * file, long line) {
  STOP("Boost assertion failed: %s (%s) in function %s at %s:%ld", expr, msg, function, file, line);
}

} // namespace boost

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/chi_squared_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/tail_quantile.hpp>
#include <boost/version.hpp>

#define BHRNG boost::random::mt19937

// Remove after boost upgrade including inverse_gaussian and generalized_inverse_gaussian
// https://github.com/boostorg/random/pull/124
// https://github.com/boostorg/random/pull/126
// Expected to be in the next release: 1.88.0
#if BOOST_VERSION < 108800
namespace boost {
namespace random {

template<class RealType> class inverse_gaussian_distribution;
template<class RealType> class generalized_inverse_gaussian_distribution;

/**
 * @brief boost.Random's inverse_gaussian generator
 * Will be included in the boost/random in the next release (1.88.0)
 * Refer to my PR https://github.com/boostorg/random/pull/124
 * This will be removed after the new release is available.
 * 
 * @tparam RealType 
 */
template<class RealType = double>
class inverse_gaussian_distribution {
public:
	typedef RealType result_type;
  typedef RealType input_type;

	class param_type {
	public:
		typedef inverse_gaussian_distribution distribution_type;

		explicit param_type(RealType alpha_arg = RealType(1.0),
											  RealType beta_arg = RealType(1.0)) : _alpha(alpha_arg), _beta(beta_arg) {
			BOOST_ASSERT(alpha_arg > 0);
			BOOST_ASSERT(beta_arg > 0);
		}

    RealType alpha() const { return _alpha; }
    RealType beta() const { return _beta; }

    BOOST_RANDOM_DETAIL_OSTREAM_OPERATOR(os, param_type, parm) { os << parm._alpha << ' ' << parm._beta; return os; }

    BOOST_RANDOM_DETAIL_ISTREAM_OPERATOR(is, param_type, parm) { is >> parm._alpha >> std::ws >> parm._beta; return is; }

    BOOST_RANDOM_DETAIL_EQUALITY_OPERATOR(param_type, lhs, rhs) { return lhs._alpha == rhs._alpha && lhs._beta == rhs._beta; }

    BOOST_RANDOM_DETAIL_INEQUALITY_OPERATOR(param_type)

	private:
		RealType _alpha;
		RealType _beta;
	};

#ifndef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
    BOOST_STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer);
#endif

	explicit inverse_gaussian_distribution(RealType alpha_arg = RealType(1.0),
											 									 RealType beta_arg = RealType(1.0)) : _alpha(alpha_arg), _beta(beta_arg) {
		BOOST_ASSERT(alpha_arg > 0);
		BOOST_ASSERT(beta_arg > 0);
		init();
	}

	explicit inverse_gaussian_distribution(const param_type& parm) : _alpha(parm.alpha()), _beta(parm.beta()) {
		init();
	}

  template<class URNG>
  RealType operator()(URNG& urng) const {
#ifndef BOOST_NO_STDC_NAMESPACE
		using std::sqrt;
#endif
		RealType w = _alpha * chi_squared_distribution<RealType>(result_type(1))(urng);
		RealType cand = _alpha + _c * (w - sqrt(w * (result_type(4) * _beta + w)));
		RealType u = uniform_01<RealType>()(urng);
		if (u < _alpha / (_alpha + cand)) {
			return cand;
		}
    return _alpha * _alpha / cand;
  }

  template<class URNG>
  RealType operator()(URNG& urng, const param_type& parm) const {
    return inverse_gaussian_distribution(parm)(urng);
  }

  RealType alpha() const { return _alpha; }
  RealType beta() const { return _beta; }

  RealType min BOOST_PREVENT_MACRO_SUBSTITUTION () const { return RealType(0.0); }
  RealType max BOOST_PREVENT_MACRO_SUBSTITUTION () const { return (std::numeric_limits<RealType>::infinity)(); }

  param_type param() const { return param_type(_alpha, _beta); }
  void param(const param_type& parm) {
    _alpha = parm.alpha();
    _beta = parm.beta();
		init();
  }

  void reset() { }

  BOOST_RANDOM_DETAIL_OSTREAM_OPERATOR(os, inverse_gaussian_distribution, wd) {
    os << wd.param();
    return os;
  }

  BOOST_RANDOM_DETAIL_ISTREAM_OPERATOR(is, inverse_gaussian_distribution, wd) {
    param_type parm;
    if(is >> parm) {
      wd.param(parm);
    }
    return is;
  }

  BOOST_RANDOM_DETAIL_EQUALITY_OPERATOR(inverse_gaussian_distribution, lhs, rhs) { return lhs._alpha == rhs._alpha && lhs._beta == rhs._beta; }

  BOOST_RANDOM_DETAIL_INEQUALITY_OPERATOR(inverse_gaussian_distribution)

private:
	result_type _alpha;
	result_type _beta;
	result_type _c;

	void init() {
		_c = _alpha / (result_type(2) * _beta);
  }
};

/**
 * @brief boost.Random's generalized_inverse_gaussian generator
 * Will be included in the boost/random in the next release (1.88.0)
 * Refer to my PR https://github.com/boostorg/random/pull/126
 * This will be removed after the new release is available.
 * 
 * @tparam RealType 
 */
template<class RealType = double>
class generalized_inverse_gaussian_distribution {
public:
	typedef RealType result_type;
	typedef RealType input_type;

	class param_type {
	public:
		typedef generalized_inverse_gaussian_distribution distribution_type;

		explicit param_type(RealType p_arg = RealType(1.0),
		                   	RealType a_arg = RealType(1.0),
												RealType b_arg = RealType(1.0))
		: _p(p_arg), _a(a_arg), _b(b_arg) {
			BOOST_ASSERT(
				(p_arg > RealType(0) && a_arg > RealType(0) && b_arg >= RealType(0)) ||
				(p_arg == RealType(0) && a_arg > RealType(0) && b_arg > RealType(0)) ||
				(p_arg < RealType(0) && a_arg >= RealType(0) && b_arg > RealType(0))
			);
		}

		RealType p() const { return _p; }
		RealType a() const { return _a; }
		RealType b() const { return _b; }

		BOOST_RANDOM_DETAIL_OSTREAM_OPERATOR(os, param_type, parm) {
			os << parm._p << ' ' << parm._a << ' ' << parm._b;
			return os;
		}

		BOOST_RANDOM_DETAIL_ISTREAM_OPERATOR(is, param_type, parm) {
			is >> parm._p >> std::ws >> parm._a >> std::ws >> parm._b;
			return is;
		}

		BOOST_RANDOM_DETAIL_EQUALITY_OPERATOR(param_type, lhs, rhs) { return lhs._p == rhs._p && lhs._a == rhs._a && lhs._b == rhs._b; }

		BOOST_RANDOM_DETAIL_INEQUALITY_OPERATOR(param_type)

	private:
		RealType _p;
		RealType _a;
		RealType _b;
	};

#ifndef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
	BOOST_STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer);
#endif

	explicit generalized_inverse_gaussian_distribution(RealType p_arg = RealType(1.0),
                       								   						 RealType a_arg = RealType(1.0),
													   						 						 RealType b_arg = RealType(1.0))
  : _p(p_arg), _a(a_arg), _b(b_arg) {
		BOOST_ASSERT(
			(p_arg > RealType(0) && a_arg > RealType(0) && b_arg >= RealType(0)) ||
			(p_arg == RealType(0) && a_arg > RealType(0) && b_arg > RealType(0)) ||
			(p_arg < RealType(0) && a_arg >= RealType(0) && b_arg > RealType(0))
		);
		init();
	}

	explicit generalized_inverse_gaussian_distribution(const param_type& parm)
	: _p(parm.p()), _a(parm.a()), _b(parm.b()) {
		init();
	}

	template<class URNG>
	RealType operator()(URNG& urng) const {
#ifndef BOOST_NO_STDC_NAMESPACE
		using std::sqrt;
		using std::log;
		using std::min;
		using std::exp;
		using std::cosh;
#endif
		RealType t = result_type(1);
		RealType s = result_type(1);
		RealType log_concave = -psi(result_type(1));
		if (log_concave >= result_type(.5) && log_concave <= result_type(2)) {
			t = result_type(1);
		} else if (log_concave > result_type(2)) {
			t = sqrt(result_type(2) / (_alpha + _abs_p));
		} else if (log_concave < result_type(.5)) {
			t = log(result_type(4) / (_alpha + result_type(2) * _abs_p));
		}
		log_concave = -psi(result_type(-1));
		if (log_concave >= result_type(.5) && log_concave <= result_type(2)) {
			s = result_type(1);
		} else if (log_concave > result_type(2)) {
			s = sqrt(result_type(4) / (_alpha * cosh(1) + _abs_p));
		} else if (log_concave < result_type(.5)) {
			s = min(result_type(1) / _abs_p, log(result_type(1) + result_type(1) / _alpha + sqrt(result_type(1) / (_alpha * _alpha) + result_type(2) / _alpha)));
		}
		RealType eta = -psi(t);
		RealType zeta = -psi_deriv(t);
		RealType theta = -psi(-s);
		RealType xi = psi_deriv(-s);
		RealType p = result_type(1) / xi;
		RealType r = result_type(1) / zeta;
		RealType t_deriv = t - r * eta;
		RealType s_deriv = s - p * theta;
		RealType q = t_deriv + s_deriv;
		RealType u = result_type(0);
		RealType v = result_type(0);
		RealType w = result_type(0);
		RealType cand = result_type(0);
		do {
			u = uniform_01<RealType>()(urng);
			v = uniform_01<RealType>()(urng);
			w = uniform_01<RealType>()(urng);
			if (u < q / (p + q + r)) {
				cand = -s_deriv + q * v;
			} else if (u < (q + r) / (p + q + r)) {
				cand = t_deriv - r * log(v);
			} else {
				cand = -s_deriv + p * log(v);
			}
		} while (w * chi(cand, s, t, s_deriv, t_deriv, eta, zeta, theta, xi) > exp(psi(cand)));
		cand = (_abs_p / _omega + sqrt(result_type(1) + _abs_p * _abs_p / (_omega * _omega))) * exp(cand);
		return _p > 0 ? cand * sqrt(_b / _a) : sqrt(_b / _a) / cand;
	}

	template<class URNG>
	result_type operator()(URNG& urng, const param_type& parm) const {
		return generalized_inverse_gaussian_distribution(parm)(urng);
	}

	RealType p() const { return _p; }
	RealType a() const { return _a; }
	RealType b() const { return _b; }

	RealType min BOOST_PREVENT_MACRO_SUBSTITUTION () const { return RealType(0.0); }
	RealType max BOOST_PREVENT_MACRO_SUBSTITUTION () const { return (std::numeric_limits<RealType>::infinity)(); }

	param_type param() const { return param_type(_p, _a, _b); }
	void param(const param_type& parm) {
		_p = parm.p();
		_a = parm.a();
		_b = parm.b();
		init();
	}

	void reset() { }

	BOOST_RANDOM_DETAIL_OSTREAM_OPERATOR(os, generalized_inverse_gaussian_distribution, wd) {
		os << wd.param();
		return os;
	}

	BOOST_RANDOM_DETAIL_ISTREAM_OPERATOR(is, generalized_inverse_gaussian_distribution, wd) {
		param_type parm;
		if(is >> parm) {
			wd.param(parm);
		}
		return is;
	}

	BOOST_RANDOM_DETAIL_EQUALITY_OPERATOR(generalized_inverse_gaussian_distribution, lhs, rhs) { return lhs._p == rhs._p && lhs._a == rhs._a && lhs._b == rhs._b; }

	BOOST_RANDOM_DETAIL_INEQUALITY_OPERATOR(generalized_inverse_gaussian_distribution)

private:
	RealType _p;
	RealType _a;
	RealType _b;
	RealType _abs_p;
	RealType _omega;
	RealType _alpha;

	void init() {
#ifndef BOOST_NO_STDC_NAMESPACE
		using std::abs;
		using std::sqrt;
#endif
    _abs_p = abs(_p);
		_omega = sqrt(_a * _b); // two-parameter representation (p, omega)
		_alpha = sqrt(_omega * _omega + _abs_p * _abs_p) - _abs_p;
    }

	result_type psi(const RealType& x) const {
#ifndef BOOST_NO_STDC_NAMESPACE
		using std::cosh;
		using std::exp;
#endif
		return -_alpha * (cosh(x) - result_type(1)) - _abs_p * (exp(x) - x - result_type(1));
	}

	result_type psi_deriv(const RealType& x) const {
#ifndef BOOST_NO_STDC_NAMESPACE
		using std::sinh;
		using std::exp;
#endif
		return -_alpha * sinh(x) - _abs_p * (exp(x) - result_type(1));
	}

	static result_type chi(const RealType& x,
						   					 const RealType& s, const RealType& t,
						   					 const RealType& s_deriv, const RealType& t_deriv,
						   					 const RealType& eta, const RealType& zeta, const RealType& theta, const RealType& xi) {
#ifndef BOOST_NO_STDC_NAMESPACE
		using std::exp;
#endif
		if (x >= -s_deriv && x <= t_deriv) {
			return result_type(1);
		} else if (x > t_deriv) {
			return exp(-eta - zeta * (x - t));
		}
		return exp(-theta + xi * (x + s));
	}
};

} // namespace random

using random::inverse_gaussian_distribution;
using random::generalized_inverse_gaussian_distribution;

} // namespace boost
#else

#include <boost/random/inverse_gaussian_distribution.hpp>
#include <boost/random/generalized_inverse_gaussian_distribution.hpp>

#endif

#if defined(__cpp_lib_optional)

#include <optional>

template <typename T>
using Optional = std::optional<T>;

#define NULLOPT std::nullopt

#else

#include <boost/optional.hpp>

template <typename T>
using Optional = boost::optional<T>;

#define NULLOPT boost::none

#endif

namespace bvhar {

using ColMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

using VectorXb = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, 1> vectorize_eigen(const Eigen::MatrixBase<Derived>& x) {
	// should use x.eval() when x is expression such as block or transpose. Use matrix().eval() if array.
	return x.reshaped();
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> unvectorize(const Eigen::MatrixBase<Derived>& x, int num_cols) {
	// should use x.eval() when x is expression such as block or transpose. Otherwise, can get wrong result.
	int num_rows = x.size() / num_cols;
	// return Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Map(x.derived().data(), num_rows, num_cols);
	return x.reshaped(num_rows, num_cols);
}

template<typename Derived1, typename Derived2>
inline Eigen::Matrix<typename Derived1::Scalar, Derived1::RowsAtCompileTime, Derived2::ColsAtCompileTime> 
kronecker_eigen(const Eigen::MatrixBase<Derived1>& x, const Eigen::MatrixBase<Derived2>& y) {
	// should use x.eval() when x is expression such as block or transpose.
  return Eigen::kroneckerProduct(x, y).eval();
}

// Gamma function
inline double gammafn(double x) {
	return Rf_gammafn(x);
}

inline double mgammafn(double x, int p) {
  if (p < 1) {
    STOP("'p' should be larger than or same as 1.");
  }
  if (x <= 0) {
    STOP("'x' should be larger than 0.");
  }
  if (p == 1) {
    return gammafn(x);
  }
  if (2 * x < p) {
    STOP("'x / 2' should be larger than 'p'.");
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

// Log of Multivariate Gamma Function
// 
// Compute log of multivariate gamma function numerically
// 
// @param x Double, non-negative argument
// @param p Integer, dimension
inline double lmgammafn(double x, int p) {
  // if (p < 1) {
  //   Rcpp::stop("'p' should be larger than or same as 1.");
  // }
  // if (x <= 0) {
  //   Rcpp::stop("'x' should be larger than 0.");
  // }
  if (p == 1) {
    return lgammafn(x);
  }
  // if (2 * x < p) {
  //   Rcpp::stop("'x / 2' should be larger than 'p'.");
  // }
  double res = p * (p - 1) / 4.0 * log(M_PI);
  for (int i = 0; i < p; i++) {
    res += lgammafn(x - i / 2.0);
  }
  return res;
}

// Density of Inverse Gamma Distribution
// 
// Compute the pdf of Inverse Gamma distribution
// 
// @param x non-negative argument
// @param shp Shape of the distribution
// @param scl Scale of the distribution
// @param lg If true, return log(f)
inline double invgamma_dens(double x, double shp, double scl, bool lg) {
  if (x < 0 ) {
    STOP("'x' should be larger than 0.");
  }
  if (shp <= 0 ) {
    STOP("'shp' should be larger than 0.");
  }
  if (scl <= 0 ) {
    STOP("'scl' should be larger than 0.");
  }
  double res = pow(scl, shp) * pow(x, -shp - 1) * exp(-scl / x) / bvhar::gammafn(shp);
  if (lg) {
    return log(res);
  }
  return res;
}

// RNG----------------------------------------
inline void cut_param(double& param) {
	if (param < std::numeric_limits<double>::min() || std::isnan(param)) {
		param = std::numeric_limits<double>::min();
	} else if (param > std::numeric_limits<double>::max() || std::isinf(param)) {
		param = std::numeric_limits<double>::max();
	}
}

#ifdef USE_RCPP
inline double bindom_rand(int n, double prob) {
	return Rf_rbinom(n, prob);
}

inline double chisq_rand(double df) {
	return Rf_rchisq(df);
}

inline double gamma_rand(double shp, double scl) {
	return Rf_rgamma(shp, scl); // 2nd: scale
}

inline double unif_rand(double min, double max) {
	return Rf_runif(min, max);
}

inline double beta_rand(double s1, double s2) {
	return Rf_rbeta(s1, s2);
}
#endif

inline double normal_rand(BHRNG& rng) {
	boost::random::normal_distribution<> rdist(0.0, 1.0);
	return rdist(rng);
}

inline double chisq_rand(double df, BHRNG& rng) {
	boost::random::chi_squared_distribution<> rdist(df);
	return rdist(rng);
}

inline double gamma_rand(double shp, double scl, BHRNG& rng) {
	cut_param(scl);
	boost::random::gamma_distribution<> rdist(shp, scl); // 2nd: scale
	return rdist(rng);
}

inline double ber_rand(double prob, BHRNG& rng) {
	boost::random::bernoulli_distribution<> rdist(prob); // Bernoulli supported -> use this instead of binomial
	return rdist(rng) * 1.0; // change to int later: now just use double to match Rf_rbinom
}

inline double unif_rand(double min, double max, BHRNG& rng) {
	boost::random::uniform_real_distribution<> rdist(min, max);
	return rdist(rng);
}

inline double unif_rand(BHRNG& rng) {
	boost::random::uniform_01<> rdist;
	return rdist(rng);
}

inline double beta_rand(double s1, double s2, BHRNG& rng) {
	boost::random::beta_distribution<> rdist(s1, s2);
	return rdist(rng);
}

inline int cat_rand(const Eigen::VectorXd& probabilities, BHRNG& rng) {
	boost::random::discrete_distribution<> rdist(probabilities.data(), probabilities.data() + probabilities.size());
	return rdist(rng);
}

inline double quantile_lower(const Eigen::Ref<Eigen::VectorXd>& x, double prob) {
	boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::tail_quantile<boost::accumulators::left>>> acc(
		boost::accumulators::tag::tail<boost::accumulators::left>::cache_size = x.size()
	);
	for (const auto &val : x) {
		acc(val);
	}
	return boost::accumulators::tail_quantile(acc, boost::accumulators::quantile_probability = prob);
}

inline double quantile_upper(const Eigen::Ref<Eigen::VectorXd>& x, double prob) {
	boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::tail_quantile<boost::accumulators::right>>> acc(
		boost::accumulators::tag::tail<boost::accumulators::right>::cache_size = x.size()
	);
	for (const auto &val : x) {
		acc(val);
	}
	return boost::accumulators::tail_quantile(acc, boost::accumulators::quantile_probability = prob);
}

} // namespace bvhar

#endif // BVHAR_CORE_COMMON_H