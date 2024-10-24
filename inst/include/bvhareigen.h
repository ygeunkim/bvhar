#ifndef BVHAREIGEN_H
#define BVHAREIGEN_H

#include "commondefs.h"

#undef eigen_assert
#define eigen_assert(x) \
  if (!(x)) { STOP("Eigen assertion failed: " #x); }

#ifdef USE_RCPP
	#include <RcppEigen.h>
#else
	#include <Eigen/Dense>
	#include <Eigen/Cholesky>
	#include <Eigen/QR>
	#include <unsupported/Eigen/KroneckerProduct>
	#include <pybind11/eigen.h>
#endif

#endif // BVHAREIGEN_H