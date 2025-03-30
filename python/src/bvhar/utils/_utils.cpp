#include <bvhar/utils>

Eigen::MatrixXd build_response(Eigen::Ref<Eigen::MatrixXd> y, int var_lag, int index) {
	return bvhar::build_y0(y, var_lag, index);
}

Eigen::MatrixXd build_design(Eigen::Ref<Eigen::MatrixXd> y, int var_lag, bool include_mean) {
	return bvhar::build_x0(y, var_lag, include_mean);
}

Eigen::MatrixXd build_design(Eigen::Ref<Eigen::MatrixXd> y, int week, int month, bool include_mean) {
	return bvhar::build_x0(y, month, include_mean) * bvhar::build_vhar(y.cols(), week, month, include_mean).transpose();
}

PYBIND11_MODULE(_utils, m) {
	m.def("build_response", &build_response, "Build response matrix");
	m.def("build_design", py::overload_cast<Eigen::Ref<Eigen::MatrixXd>, int, bool>(&build_design), "Build design matrix");
	m.def("build_design", py::overload_cast<Eigen::Ref<Eigen::MatrixXd>, int, int, bool>(&build_design), "Build VHAR design matrix");
}
