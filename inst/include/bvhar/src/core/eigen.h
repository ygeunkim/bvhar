#ifndef BVHAR_CORE_EIGEN_H
#define BVHAR_CORE_EIGEN_H

#include "./commondefs.h"

#undef eigen_assert
#define eigen_assert(x) \
  if (!(x)) { STOP("Eigen assertion failed: " #x); }

#ifdef USE_RCPP
	#include <RcppEigen.h>
#else
	#include <Eigen/Eigen>
	#include <unsupported/Eigen/KroneckerProduct>
	#include <pybind11/eigen.h>
#endif

#endif // BVHAR_CORE_EIGEN_H