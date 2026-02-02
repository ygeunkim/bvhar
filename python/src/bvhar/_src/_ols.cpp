#include <bvhar/ols>

PYBIND11_MODULE(_ols, m) {
	m.doc() = "OLS for VAR and VHAR";

  // py::class_<baecon::bvhar::MultiOls>(m, "MultiOls")
  //   .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>())
  //   .def("returnOlsRes", &baecon::bvhar::MultiOls::returnOlsRes);
	
	// py::class_<baecon::bvhar::LltOls, baecon::bvhar::MultiOls>(m, "LltOls")
  //   .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>());
	
	// py::class_<baecon::bvhar::QrOls, baecon::bvhar::MultiOls>(m, "QrOls")
  //   .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>());
	
	py::class_<baecon::bvhar::OlsVar>(m, "OlsVar")
		.def(
			py::init<const Eigen::MatrixXd&, int, const bool, int>(),
			py::arg("y"), py::arg("lag") = 1, py::arg("include_mean") = true, py::arg("method") = 1
		)
		.def("returnOlsRes", &baecon::bvhar::OlsVar::returnOlsRes);
	
	py::class_<baecon::bvhar::OlsVhar>(m, "OlsVhar")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, const bool, int>(),
			py::arg("y"), py::arg("week") = 5, py::arg("month") = 22, py::arg("include_mean") = true, py::arg("method") = 1
		)
		.def("returnOlsRes", &baecon::bvhar::OlsVhar::returnOlsRes);

	py::class_<baecon::bvhar::OlsForecastRun>(m, "OlsForecast")
		.def(
			py::init<int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&, bool>(),
			py::arg("lag"), py::arg("step"), py::arg("response_mat"), py::arg("coef_mat"), py::arg("include_mean")
		)
		.def(
			py::init<int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&, bool>(),
			py::arg("week") = 5, py::arg("month") = 22, py::arg("step"), py::arg("response_mat"), py::arg("coef_mat"), py::arg("include_mean")
		)
		.def("returnForecast", &baecon::bvhar::OlsForecastRun::returnForecast);
	
	py::class_<baecon::bvhar::VarOutforecastRun<baecon::bvhar::OlsRollforecastRun>>(m, "OlsVarRoll")
		.def(py::init<const Eigen::MatrixXd&, int, bool, int, const Eigen::MatrixXd&, int, int>())
		.def("returnForecast", &baecon::bvhar::VarOutforecastRun<baecon::bvhar::OlsRollforecastRun>::returnForecast);
	
	py::class_<baecon::bvhar::VarOutforecastRun<baecon::bvhar::OlsExpandforecastRun>>(m, "OlsVarExpand")
		.def(py::init<const Eigen::MatrixXd&, int, bool, int, const Eigen::MatrixXd&, int, int>())
		.def("returnForecast", &baecon::bvhar::VarOutforecastRun<baecon::bvhar::OlsExpandforecastRun>::returnForecast);
	
	py::class_<baecon::bvhar::VharOutforecastRun<baecon::bvhar::OlsRollforecastRun>>(m, "OlsVharRoll")
		.def(py::init<const Eigen::MatrixXd&, int, int, bool, int, const Eigen::MatrixXd&, int, int>())
		.def("returnForecast", &baecon::bvhar::VharOutforecastRun<baecon::bvhar::OlsRollforecastRun>::returnForecast);
	
	py::class_<baecon::bvhar::VharOutforecastRun<baecon::bvhar::OlsExpandforecastRun>>(m, "OlsVharExpand")
		.def(py::init<const Eigen::MatrixXd&, int, int, bool, int, const Eigen::MatrixXd&, int, int>())
		.def("returnForecast", &baecon::bvhar::VharOutforecastRun<baecon::bvhar::OlsExpandforecastRun>::returnForecast);

	py::class_<baecon::bvhar::OlsSpilloverRun>(m, "OlsSpillover")
		.def(py::init<int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&>())
		.def(py::init<int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&>())
		.def("returnSpillover", &baecon::bvhar::OlsSpilloverRun::returnSpillover);
	
	py::class_<baecon::bvhar::OlsDynamicSpillover>(m, "OlsDynamicSpillover")
		.def(py::init<const Eigen::MatrixXd&, int, int, int, bool, int, int>())
		.def(py::init<const Eigen::MatrixXd&, int, int, int, bool, int, int, int>())
		.def("returnSpillover", &baecon::bvhar::OlsDynamicSpillover::returnSpillover);
}
