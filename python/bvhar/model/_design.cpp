#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <bvhardesign.h>

namespace py = pybind11;

Eigen::MatrixXd build_response(Eigen::Ref<Eigen::MatrixXd> y, int var_lag, int index) {
	return bvhar::build_y0(y, var_lag, index);
}

PYBIND11_MODULE(_design, m) {
	m.def("build_response", &build_response, "Build Response Matrix");
}
