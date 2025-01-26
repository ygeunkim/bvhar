#include <bvhar/forecast>

PYBIND11_MODULE(_forecast, m) {
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
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster>>(m, "LdltVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster>>(m, "LdltVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster>>(m, "LdltVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster, false>>(m, "LdltGrpVarRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster, false>::returnForecast);
	
	py::class_<bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster, false>>(m, "LdltGrpVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster, false>::returnForecast);
	
	py::class_<bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster, false>>(m, "LdltGrpVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::RegForecaster, false>::returnForecast);
	
	py::class_<bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster, false>>(m, "LdltGrpVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::RegForecaster, false>::returnForecast);
	
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
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster>>(m, "SvVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster>>(m, "SvVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster>>(m, "SvVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster>::returnForecast);
	
	py::class_<bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster, false>>(m, "SvGrpVarRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVarforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster, false>::returnForecast);
	
	py::class_<bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster, false>>(m, "SvGrpVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVarforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster, false>::returnForecast);
	
	py::class_<bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster, false>>(m, "SvGrpVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVharforecastRun<bvhar::McmcRollforecastRun, bvhar::SvForecaster, false>::returnForecast);
	
	py::class_<bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster, false>>(m, "SvGrpVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::McmcVharforecastRun<bvhar::McmcExpandforecastRun, bvhar::SvForecaster, false>::returnForecast);
}