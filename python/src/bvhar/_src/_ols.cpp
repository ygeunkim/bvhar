#include <pybind11/eigen.h>
#include <ols.h>

PYBIND11_MODULE(_ols, m) {
	m.doc() = "OLS for VAR and VHAR";

  py::class_<bvhar::MultiOls>(m, "MultiOls")
    .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>())
    .def("returnOlsRes", &bvhar::MultiOls::returnOlsRes);
	
	py::class_<bvhar::LltOls, bvhar::MultiOls>(m, "LltOls")
    .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>());
	
	py::class_<bvhar::QrOls, bvhar::MultiOls>(m, "QrOls")
    .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>());
	
	py::class_<bvhar::OlsVar>(m, "OlsVar")
		.def(
			py::init<const Eigen::MatrixXd&, int, const bool, int>(),
			py::arg("y"), py::arg("lag") = 1, py::arg("include_mean") = true, py::arg("method") = 1
		)
		.def("returnOlsRes", &bvhar::OlsVar::returnOlsRes);
	
	py::class_<bvhar::OlsVhar>(m, "OlsVhar")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, const bool, int>(),
			py::arg("y"), py::arg("week") = 5, py::arg("month") = 22, py::arg("include_mean") = true, py::arg("method") = 1
		)
		.def("returnOlsRes", &bvhar::OlsVhar::returnOlsRes);
}
