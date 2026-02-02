#ifndef BVHAR_CORE_LBFGSPP_H
#define BVHAR_CORE_LBFGSPP_H

#ifdef BVHAR_USE_RCPP

// #include <Func.h>
// #include <RcppNumerical.h>
// #include <optimization/LBFGS.h>
#include <optimization/LBFGSB.h>

#else

#include <LBFGSB.h>

#endif // BVHAR_USE_RCPP

#endif // BVHAR_CORE_LBFGSPP_H