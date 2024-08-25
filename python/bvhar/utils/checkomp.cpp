#include <pybind11/pybind11.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace py = pybind11;

void check_omp() {
#ifdef _OPENMP
  // std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
	py::print("OpenMP threads: ", omp_get_max_threads());
#else
	// Rcpp::Rcout << "OpenMP not available in this machine." << "\n";
	py::print("OpenMP not available in this machine.");
#endif
}

PYBIND11_MODULE(checkomp, m) {
	m.def("check_omp", &check_omp, "Check OpenMP");
}