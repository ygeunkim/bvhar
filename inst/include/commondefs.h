/**
 * @file commondefs.h
 * @author Young Geun Kim (ygeunkimstat@gmail.com)
 * @brief Common header
 */

#ifndef COMMONDEFS_H
#define COMMONDEFS_H

#ifdef USE_RCPP
	// #include <RcppEigen.h>
	#include <Rcpp.h>
	#include <cmath>
	#include <string>
	// #include <RcppSpdlog>
	// #include <RcppThread.h>

	#define STOP(...) Rcpp::stop(__VA_ARGS__)

	#define COUT Rcpp::Rcout
	#define ENDL "\n"
	#define FLUSH Rcpp::Rcout.flush()
	// #define FLUSH std::cout.flush()
	// #define FLUSH R_FlushConsole()
	#define STRING std::string

	// #include <RcppSpdlog>

	// #define SPDLOG_SINK_MT(value) spdlog::r_sink_mt(value)

	// #include <spdlog/spdlog.h>
	// #include <spdlog/sinks/stdout_sinks.h>
	// #define SPDLOG_SINK_MT(value) spdlog::stdout_logger_mt(value)

	#define LIST Rcpp::List
	#define LIST_OF_LIST Rcpp::List
	#define PY_LIST Rcpp::List
	#define WRAP(value) Rcpp::wrap(value)
	#define CAST Rcpp::as
	#define CAST_DOUBLE(value) value
	#define CAST_INT(value) value
	#define CONTAINS(container, key) container.containsElementNamed(key)
	#define CREATE_LIST(...) Rcpp::List::create(__VA_ARGS__)
	#define NAMED Rcpp::Named
	#define ACCESS_LIST(iterator, list) iterator
	#define IS_MATRIX(element) Rcpp::is<Rcpp::NumericMatrix>(element)
	#define CAST_VECTOR(element) element
	#define CAST_MATRIX(element) element
#else
	#include <pybind11/pybind11.h>
	#include <cmath>
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

	#define Rf_gammafn(x) std::tgamma(x)
	#define Rf_lgammafn(x) std::lgamma(x)
	#define Rf_dgamma(x, shp, scl, lg) (lg ? log((shp - 1) * log(x) - x / scl - std::lgamma(shp) - shp * log(scl)) : exp((shp - 1) * log(x) - x / scl - std::lgamma(shp) - shp * log(scl)))
	
	namespace py = pybind11;

	void stop_fmt(const std::string& msg) {
		throw py::value_error(msg);
	}
	
	template<typename... Args>
	void stop_fmt(const std::string& msg, Args&&... args) {
		throw py::value_error(py::str(msg).format(std::forward<Args>(args)...));
	}

	#define STOP(...) stop_fmt(__VA_ARGS__)

	#define COUT std::cout
	#define ENDL std::endl
	#define FLUSH std::cout.flush()
	#define STRING py::str
	// #define SPDLOG_SINK_MT(value) spdlog::stdout_logger_mt(value)

	#define LIST py::dict
	#define LIST_OF_LIST std::vector<py::dict>
	#define PY_LIST py::list
	#define WRAP(value) value
  #define CAST py::cast
	#define CAST_DOUBLE(value) py::cast<double>(value)
	#define CAST_INT(value) py::int_(value)
	#define CONTAINS(container, key) container.contains(key)
	#define CREATE_LIST(...) py::dict(__VA_ARGS__)
	#define NAMED py::arg
	#define ACCESS_LIST(iterator, list) list[iterator.first]
	#define IS_MATRIX(element) py::detail::type_caster<Eigen::MatrixXd>().load(element, false)
	#define CAST_VECTOR(element) py::cast<Eigen::VectorXd>(element)
	#define CAST_MATRIX(element) py::cast<Eigen::MatrixXd>(element)

	#ifndef M_PI
		// Some platform does not have M_PI defined - to the same value as in Rmath.h
		#define M_PI 3.141592653589793238462643383280
	#endif
#endif

#include <memory>

#if !defined(__cpp_lib_make_unique)
namespace std {

template <typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args) {
	return unique_ptr<T>(new T(forward<Args>(args)...));
}

} // namespace std
#endif

#endif // COMMONDEFS_H