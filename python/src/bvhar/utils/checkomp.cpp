#include <bvhar/utils>

namespace py = pybind11;

int get_maxomp() {
	return omp_get_max_threads();
}

bool is_omp() {
#ifdef _OPENMP
  return true;
#else
	return false;
#endif
}

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
	m.doc() = "Check OpenMP configuration";

	m.def("get_maxomp", &get_maxomp, "Show the maximum thread numbers");
	m.def("is_omp", &is_omp, "Give boolean for OpenMP");
	m.def("check_omp", &check_omp, "Print if OpenMP is enabled");
}