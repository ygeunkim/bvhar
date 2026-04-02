/**
 * @file commondefs.h
 * @author Young Geun Kim (ygeunkimstat@gmail.com)
 * @brief Common header
 */

#ifndef BVHAR_CORE_COMMONDEFS_H
#define BVHAR_CORE_COMMONDEFS_H

#if defined(BVHAR_USE_RCPP)
	// #include <RcppEigen.h>
	#include <Rcpp.h>
	#include <cmath>
	#include <string>
	// #include <RcppSpdlog>
	// #include <RcppThread.h>
	#include <RcppThread/Rcout.hpp>

	#define BVHAR_STOP(...) Rcpp::stop(__VA_ARGS__)

	#define BVHAR_COUT RcppThread::Rcout
	#define BVHAR_ENDL "\n" << std::flush
	// #define BVHAR_FLUSH Rcpp::Rcout.flush()
	#define BVHAR_FLUSH RcppThread::Rcout << std::flush
	// #define FLUSH std::cout.flush()
	// #define FLUSH R_FlushConsole()
	#define BVHAR_STRING std::string

	// #include <RcppSpdlog>

	// #define BVHAR_SPDLOG_SINK_MT(value) spdlog::r_sink_mt(value)

	// #include <spdlog/spdlog.h>
	// #include <spdlog/sinks/stdout_sinks.h>
	// #define BVHAR_SPDLOG_SINK_MT(value) spdlog::stdout_logger_mt(value)

	#define BVHAR_LIST Rcpp::List
	#define BVHAR_LIST_OF_LIST Rcpp::List
	#define BVHAR_PY_LIST Rcpp::List
	#define BVHAR_WRAP(value) Rcpp::wrap(value)
	#define BVHAR_CAST Rcpp::as
	#define BVHAR_CAST_DOUBLE(value) value
	#define BVHAR_CAST_INT(value) value
	#define BVHAR_CAST_BOOL(value) value
	#define BVHAR_CAST_LIST(value) value
	#define BVHAR_CAST_PY_LIST(value) value
	#define BVHAR_CAST_LIST_OF_LIST(value) value
	#define BVHAR_CONTAINS(container, key) container.containsElementNamed(key)
	#define BVHAR_CREATE_LIST(...) Rcpp::List::create(__VA_ARGS__)
	#define BVHAR_NAMED Rcpp::Named
	#define BVHAR_ACCESS_LIST(iterator, list) iterator
	#define BVHAR_IS_MATRIX(element) (Rcpp::is<Rcpp::NumericMatrix>(element) || Rcpp::is<Rcpp::IntegerMatrix>(element) || Rcpp::is<Rcpp::LogicalMatrix>(element))
	#define BVHAR_IS_VECTOR(element) (Rcpp::is<Rcpp::NumericVector>(element) || Rcpp::is<Rcpp::IntegerVector>(element) || Rcpp::is<Rcpp::LogicalVector>(element))
	#define BVHAR_IS_LOGICAL(element) Rcpp::is<Rcpp::LogicalVector>(element)
	#define BVHAR_CAST_VECTOR(element) element
	#define BVHAR_CAST_MATRIX(element) element
#elif defined(BVHAR_USE_PYBIND11)
	#include <pybind11/pybind11.h>
	#include <string>
	#include <stdexcept>
	#include <iostream>
	// #include <Eigen/Dense>
	// #include <Eigen/Cholesky>
	// #include <Eigen/QR>
	// #include <unsupported/Eigen/KroneckerProduct>
	#include <pybind11/stl.h>
	// #include <pybind11/eigen.h>
	// #include <spdlog/spdlog.h>
	// #include <spdlog/sinks/stdout_sinks.h>
	
	namespace py = pybind11;

	void stop_fmt(const std::string& msg) {
		throw py::value_error(msg);
	}
	
	template<typename... Args>
	void stop_fmt(const std::string& msg, Args&&... args) {
		throw py::value_error(py::str(msg).format(std::forward<Args>(args)...));
	}

	#define BVHAR_STOP(...) stop_fmt(__VA_ARGS__)

	#define BVHAR_COUT std::cout
	#define BVHAR_ENDL std::endl
	#define BVHAR_FLUSH std::cout.flush()
	#define BVHAR_STRING py::str
	// #define BVHAR_SPDLOG_SINK_MT(value) spdlog::stdout_logger_mt(value)

	#define BVHAR_LIST py::dict
	#define BVHAR_LIST_OF_LIST std::vector<py::dict>
	#define BVHAR_PY_LIST py::list
	#define BVHAR_WRAP(value) value
  #define BVHAR_CAST py::cast
	#define BVHAR_CAST_DOUBLE(value) py::cast<double>(value)
	#define BVHAR_CAST_INT(value) py::int_(value)
	#define BVHAR_CAST_BOOL(value) py::cast<bool>(value)
	#define BVHAR_CAST_LIST(value) value
	#define BVHAR_CAST_PY_LIST(value) value
	#define BVHAR_CAST_LIST_OF_LIST(value) value
	#define BVHAR_CONTAINS(container, key) container.contains(key)
	#define BVHAR_CREATE_LIST(...) py::dict(__VA_ARGS__)
	#define BVHAR_NAMED py::arg
	#define BVHAR_ACCESS_LIST(iterator, list) list[iterator.first]
	#define BVHAR_IS_MATRIX(element) (py::detail::type_caster<Eigen::MatrixXd>().load(element, false) || py::detail::type_caster<Eigen::MatrixXi>().load(element, false) || py::detail::type_caster<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>().load(element, false))
	#define BVHAR_IS_VECTOR(element) (py::detail::type_caster<Eigen::VectorXd>().load(element, false) || py::detail::type_caster<Eigen::VectorXi>().load(element, false) || py::detail::type_caster<Eigen::Matrix<bool, Eigen::Dynamic, 1>>().load(element, false))
	#define BVHAR_IS_LOGICAL(element) py::detail::type_caster<Eigen::Matrix<bool, Eigen::Dynamic, 1>>().load(element, false)
	#define BVHAR_CAST_VECTOR(element) py::cast<Eigen::VectorXd>(element)
	#define BVHAR_CAST_MATRIX(element) py::cast<Eigen::MatrixXd>(element)
#else
	#include "./commoncpp.h"

	#define BVHAR_LIST BvharList
	#define BVHAR_LIST_OF_LIST std::vector<BvharList>
	#define BVHAR_PY_LIST std::vector<std::any>
	#define BVHAR_WRAP(value) value
	#define BVHAR_CAST std::any_cast
  #define BVHAR_CAST_DOUBLE(value) std::any_cast<double>(value)
  #define BVHAR_CAST_INT(value) std::any_cast<int>(value)
  #define BVHAR_CAST_BOOL(value) std::any_cast<bool>(value)
	#define BVHAR_CAST_LIST(value) std::any_cast<BVHAR_LIST>(value)
	#define BVHAR_CAST_PY_LIST(value) std::any_cast<BVHAR_PY_LIST>(value)
	#define BVHAR_CAST_LIST_OF_LIST(value) std::any_cast<BVHAR_LIST_OF_LIST>(value)
	#define BVHAR_CONTAINS(container, key) (container.find(key) != container.end())
	#define BVHAR_CREATE_LIST(...) create_bvhar_list(__VA_ARGS__)
  #define BVHAR_NAMED BvharNamed
  #define BVHAR_ACCESS_LIST(iterator, list) iterator.second

  #define BVHAR_STOP(...) stop_fmt(__VA_ARGS__)
  #define BVHAR_COUT std::cout
  #define BVHAR_ENDL std::endl
  #define BVHAR_FLUSH std::cout.flush()
  #define BVHAR_STRING std::string

	#define BVHAR_IS_MATRIX(element) (element.type() == typeid(Eigen::MatrixXd) || element.type() == typeid(Eigen::MatrixXi) || element.type() == typeid(Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>))
  #define BVHAR_IS_VECTOR(element) (element.type() == typeid(Eigen::VectorXd) || element.type() == typeid(Eigen::VectorXi) || element.type() == typeid(Eigen::Matrix<bool, Eigen::Dynamic, 1>))
  #define BVHAR_IS_LOGICAL(element) (element.type() == typeid(bool))
  
  #define BVHAR_CAST_VECTOR(element) std::any_cast<Eigen::VectorXd>(element.eval())
  #define BVHAR_CAST_MATRIX(element) std::any_cast<Eigen::MatrixXd>(element.eval())
#endif

#ifndef M_PI
	// Some platform does not have M_PI defined - to the same value as in Rmath.h
	#define M_PI 3.141592653589793238462643383280
#endif

#if !defined(BVHAR_USE_RCPP)
	#include <cmath>

	#define Rf_gammafn(x) std::tgamma(x)
	#define Rf_lgammafn(x) std::lgamma(x)
	#define Rf_dgamma(x, shp, scl, lg) (lg ? std::log((shp - 1) * std::log(x) - x / scl - std::lgamma(shp) - shp * std::log(scl)) : std::exp((shp - 1) * std::log(x) - x / scl - std::lgamma(shp) - shp * std::log(scl)))
#endif

#include <memory>
#include <type_traits>

#if !defined(__cpp_lib_make_unique)
namespace std {

template <typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args) {
	return unique_ptr<T>(new T(forward<Args>(args)...));
}

} // namespace std
#endif

#ifdef __cpp_if_constexpr
	#define BVHAR_IF_CONSTEXPR(element) \
		if constexpr (element) 
#else
  #define BVHAR_IF_CONSTEXPR(element) \
		if (std::integral_constant<bool, element>::value)
#endif

#endif // BVHAR_CORE_COMMONDEFS_H