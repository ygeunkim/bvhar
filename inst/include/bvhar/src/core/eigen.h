#ifndef BVHAR_CORE_EIGEN_H
#define BVHAR_CORE_EIGEN_H

#include "./commondefs.h"

#undef eigen_assert
#define eigen_assert(x) \
  if (!(x)) { BVHAR_STOP("Eigen assertion failed: " #x); }

// Add a method to Eigen's MatrixBase
#define EIGEN_MATRIXBASE_PLUGIN "bvhar/src/core/eigen_plugins.h"

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