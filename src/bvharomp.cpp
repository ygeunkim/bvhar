#include <Rcpp.h>
#include "bvharomp.h"

// [[Rcpp::export]]
int get_maxomp() {
	return omp_get_max_threads();
}

// [[Rcpp::export]]
void check_omp() {
#ifdef _OPENMP
  Rcpp::Rcout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
	Rcpp::Rcout << "OpenMP not available in this machine." << "\n";
#endif
}

// [[Rcpp::export]]
bool is_omp() {
#ifdef _OPENMP
  return true;
#else
	return false;
#endif
}