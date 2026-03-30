#ifndef BVHAR_CORE_EIGEN_H
#define BVHAR_CORE_EIGEN_H

#include "./commondefs.h"

#undef eigen_assert
#define eigen_assert(x) \
  if (!(x)) { BVHAR_STOP("Eigen assertion failed: " #x); }

#ifdef BVHAR_USE_RCPP
	#include <RcppEigen.h>
#else
	#include <Eigen/Eigen>
	#include <unsupported/Eigen/KroneckerProduct>
	#ifdef BVHAR_USE_PYBIND11
		#include <pybind11/eigen.h>
	#endif
#endif

#endif // BVHAR_CORE_EIGEN_H