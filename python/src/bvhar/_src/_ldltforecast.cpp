#include <regforecaster.h>

PYBIND11_MODULE(_ldltforecast, m) {
	py::class_<bvhar::McmcForecastRun<bvhar::RegForecaster>>(m, "LdltForecast")
		.def(py::init<int, int, int, const Eigen::MatrixXd&, bool, double, py::dict&, const Eigen::VectorXi&, bool, bool, int, bool>())
		.def(py::init<int, int, int, int, const Eigen::MatrixXd&, bool, double, py::dict&, const Eigen::VectorXi&, bool, bool, int, bool>())
		.def("returnForecast", &bvhar::McmcForecastRun<bvhar::RegForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster>>(m, "LdltVarRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster>>(m, "LdltVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster>>(m, "LdltVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster>>(m, "LdltVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster>::returnForecast);
}