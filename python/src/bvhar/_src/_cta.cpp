#include <bvhar/triangular>

PYBIND11_MODULE(_cta, m) {
	py::class_<bvhar::McmcRun<bvhar::McmcReg>>(m, "McmcLdlt")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&, py::dict&,
			std::vector<py::dict>&, int, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &bvhar::McmcRun<bvhar::McmcReg>::returnRecords);
	
	py::class_<bvhar::McmcRun<bvhar::McmcReg, false>>(m, "McmcLdltGrp")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&, py::dict&,
			std::vector<py::dict>&, int, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &bvhar::McmcRun<bvhar::McmcReg, false>::returnRecords);
	
	py::class_<bvhar::McmcRun<bvhar::McmcSv>>(m, "SvMcmc")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&, py::dict&,
			std::vector<py::dict>&, int, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &bvhar::McmcRun<bvhar::McmcSv>::returnRecords);
	
	py::class_<bvhar::McmcRun<bvhar::McmcSv, false>>(m, "SvGrpMcmc")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&, py::dict&,
			std::vector<py::dict>&, int, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &bvhar::McmcRun<bvhar::McmcSv, false>::returnRecords);
	
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
	
	py::class_<bvhar::McmcSpilloverRun<bvhar::LdltRecords>>(m, "LdltSpillover")
		.def(py::init<int, int, py::dict&, bool>())
		.def(py::init<int, int, int, py::dict&, bool>())
		.def("returnSpillover", &bvhar::McmcSpilloverRun<bvhar::LdltRecords>::returnSpillover);
	
	py::class_<bvhar::DynamicLdltSpillover>(m, "LdltDynamicSpillover")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int, int, bool,
			py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int, bool,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, const Eigen::MatrixXi&, int>()
		)
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int, int, int, bool,
			py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int, bool,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, const Eigen::MatrixXi&, int>()
		)
		.def("returnSpillover", &bvhar::DynamicLdltSpillover::returnSpillover);
	
	py::class_<bvhar::McmcSpilloverRun<bvhar::SvRecords>>(m, "SvSpillover")
		.def(py::init<int, int, py::dict&, bool>())
		.def(py::init<int, int, int, py::dict&, bool>())
		.def("returnSpillover", &bvhar::McmcSpilloverRun<bvhar::SvRecords>::returnSpillover);
	
	py::class_<bvhar::DynamicSvSpillover>(m, "SvDynamicSpillover")
		.def(py::init<int, int, int, py::dict&, bool, bool, int>())
		.def(py::init<int, int, int, int, py::dict&, bool, bool, int>())
		.def("returnSpillover", &bvhar::DynamicSvSpillover::returnSpillover);
}