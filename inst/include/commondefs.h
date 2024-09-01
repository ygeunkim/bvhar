#ifndef COMMONDEFS_H
#define COMMONDEFS_H

#ifdef USE_RCPP
	#include <RcppEigen.h>
	#define STOP(...) Rcpp::stop(__VA_ARGS__)

	#define LIST Rcpp::List
	#define CAST Rcpp::as
	#define CAST_DOUBLE(value) value
	#define CAST_INT(value) value
	#define CONTAINS(container, key) container.containsElementNamed(key)
	#define CREATE_LIST(...) Rcpp::List::create(__VA_ARGS__)
	#define NAMED Rcpp::Named
	#define ACCESS_LIST(iterator, list) iterator
	#define IS_MATRIX(element) Rcpp::is<Rcpp::NumericMatrix>(element)
	#define CAST_MATRIX(element) element
#else
	#include <pybind11/pybind11.h>
	#include <cmath>
	#include <string>
	#include <stdexcept>
	#include <Eigen/Dense>
	#include <Eigen/Cholesky>
	#include <Eigen/QR>
	#include <unsupported/Eigen/KroneckerProduct>

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

	#define LIST py::dict
  #define CAST py::cast
	#define CAST_DOUBLE(value) py::cast<double>(value)
	#define CAST_INT(value) py::cast<int>(value)
	#define CONTAINS(container, key) container.contains(key)
	#define CREATE_LIST(...) py::dict(__VA_ARGS__)
	#define NAMED py::arg
	#define ACCESS_LIST(iterator, list) list[iterator.first]
	#define IS_MATRIX(element) py::detail::type_caster<Eigen::MatrixXd>().load(element, false)
	#define CAST_MATRIX(element) py::cast<Eigen::MatrixXd>(element)
#endif

#endif // COMMONDEFS_H