#include <svforecaster.h>

PYBIND11_MODULE(_svforecast, m) {
	py::class_<bvhar::McmcForecastRun<bvhar::SvForecaster>>(m, "SvForecast")
		.def(py::init<int, int, int, const Eigen::MatrixXd&, bool, double, py::dict&, const Eigen::VectorXi&, bool, bool, int, bool>())
		.def(py::init<int, int, int, int, const Eigen::MatrixXd&, bool, double, py::dict&, const Eigen::VectorXi&, bool, bool, int, bool>())
		.def("returnForecast", &bvhar::McmcForecastRun<bvhar::SvForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster>>(m, "SvVarRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster>>(m, "SvVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster>>(m, "SvVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster>>(m, "SvVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster, false>>(m, "SvGrpVarRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster, false>::returnForecast);
	
	py::class_<bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster, false>>(m, "SvGrpVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster, false>::returnForecast);
	
	py::class_<bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster, false>>(m, "SvGrpVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster, false>::returnForecast);
	
	py::class_<bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster, false>>(m, "SvGrpVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster, false>::returnForecast);
}