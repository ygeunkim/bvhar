#include <bvhar/triangular>

PYBIND11_MODULE(_cta, m) {
	py::class_<bvhar::CtaRun<bvhar::McmcReg>>(m, "McmcLdlt")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &bvhar::CtaRun<bvhar::McmcReg>::returnRecords);
	
	py::class_<bvhar::CtaRun<bvhar::McmcReg, false>>(m, "McmcLdltGrp")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &bvhar::CtaRun<bvhar::McmcReg, false>::returnRecords);
	
	py::class_<bvhar::CtaRun<bvhar::McmcSv>>(m, "SvMcmc")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &bvhar::CtaRun<bvhar::McmcSv>::returnRecords);
	
	py::class_<bvhar::CtaRun<bvhar::McmcSv, false>>(m, "SvGrpMcmc")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &bvhar::CtaRun<bvhar::McmcSv, false>::returnRecords);
	
	py::class_<bvhar::CtaForecastRun<bvhar::RegForecaster>>(m, "LdltForecast")
		.def(py::init<int, int, int, const Eigen::MatrixXd&, bool, double, py::dict&, const Eigen::VectorXi&, bool, bool, int, bool>())
		.def(py::init<int, int, int, int, const Eigen::MatrixXd&, bool, double, py::dict&, const Eigen::VectorXi&, bool, bool, int, bool>())
		.def("returnForecast", &bvhar::CtaForecastRun<bvhar::RegForecaster>::returnForecast);
	
	py::class_<bvhar::CtaVarforecastRun<bvhar::CtaRollforecastRun, bvhar::RegForecaster>>(m, "LdltVarRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVarforecastRun<bvhar::CtaRollforecastRun, bvhar::RegForecaster>::returnForecast);
	
	py::class_<bvhar::CtaVarforecastRun<bvhar::CtaExpandforecastRun, bvhar::RegForecaster>>(m, "LdltVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVarforecastRun<bvhar::CtaExpandforecastRun, bvhar::RegForecaster>::returnForecast);
	
	py::class_<bvhar::CtaVharforecastRun<bvhar::CtaRollforecastRun, bvhar::RegForecaster>>(m, "LdltVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVharforecastRun<bvhar::CtaRollforecastRun, bvhar::RegForecaster>::returnForecast);
	
	py::class_<bvhar::CtaVharforecastRun<bvhar::CtaExpandforecastRun, bvhar::RegForecaster>>(m, "LdltVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVharforecastRun<bvhar::CtaExpandforecastRun, bvhar::RegForecaster>::returnForecast);
	
	py::class_<bvhar::CtaVarforecastRun<bvhar::CtaRollforecastRun, bvhar::RegForecaster, false>>(m, "LdltGrpVarRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVarforecastRun<bvhar::CtaRollforecastRun, bvhar::RegForecaster, false>::returnForecast);
	
	py::class_<bvhar::CtaVarforecastRun<bvhar::CtaExpandforecastRun, bvhar::RegForecaster, false>>(m, "LdltGrpVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVarforecastRun<bvhar::CtaExpandforecastRun, bvhar::RegForecaster, false>::returnForecast);
	
	py::class_<bvhar::CtaVharforecastRun<bvhar::CtaRollforecastRun, bvhar::RegForecaster, false>>(m, "LdltGrpVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVharforecastRun<bvhar::CtaRollforecastRun, bvhar::RegForecaster, false>::returnForecast);
	
	py::class_<bvhar::CtaVharforecastRun<bvhar::CtaExpandforecastRun, bvhar::RegForecaster, false>>(m, "LdltGrpVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVharforecastRun<bvhar::CtaExpandforecastRun, bvhar::RegForecaster, false>::returnForecast);
	
	py::class_<bvhar::CtaForecastRun<bvhar::SvForecaster>>(m, "SvForecast")
		.def(py::init<int, int, int, const Eigen::MatrixXd&, bool, double, py::dict&, const Eigen::VectorXi&, bool, bool, int, bool>())
		.def(py::init<int, int, int, int, const Eigen::MatrixXd&, bool, double, py::dict&, const Eigen::VectorXi&, bool, bool, int, bool>())
		.def("returnForecast", &bvhar::CtaForecastRun<bvhar::SvForecaster>::returnForecast);
	
	py::class_<bvhar::CtaVarforecastRun<bvhar::CtaRollforecastRun, bvhar::SvForecaster>>(m, "SvVarRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVarforecastRun<bvhar::CtaRollforecastRun, bvhar::SvForecaster>::returnForecast);
	
	py::class_<bvhar::CtaVarforecastRun<bvhar::CtaExpandforecastRun, bvhar::SvForecaster>>(m, "SvVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVarforecastRun<bvhar::CtaExpandforecastRun, bvhar::SvForecaster>::returnForecast);
	
	py::class_<bvhar::CtaVharforecastRun<bvhar::CtaRollforecastRun, bvhar::SvForecaster>>(m, "SvVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVharforecastRun<bvhar::CtaRollforecastRun, bvhar::SvForecaster>::returnForecast);
	
	py::class_<bvhar::CtaVharforecastRun<bvhar::CtaExpandforecastRun, bvhar::SvForecaster>>(m, "SvVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVharforecastRun<bvhar::CtaExpandforecastRun, bvhar::SvForecaster>::returnForecast);
	
	py::class_<bvhar::CtaVarforecastRun<bvhar::CtaRollforecastRun, bvhar::SvForecaster, false>>(m, "SvGrpVarRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVarforecastRun<bvhar::CtaRollforecastRun, bvhar::SvForecaster, false>::returnForecast);
	
	py::class_<bvhar::CtaVarforecastRun<bvhar::CtaExpandforecastRun, bvhar::SvForecaster, false>>(m, "SvGrpVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVarforecastRun<bvhar::CtaExpandforecastRun, bvhar::SvForecaster, false>::returnForecast);
	
	py::class_<bvhar::CtaVharforecastRun<bvhar::CtaRollforecastRun, bvhar::SvForecaster, false>>(m, "SvGrpVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVharforecastRun<bvhar::CtaRollforecastRun, bvhar::SvForecaster, false>::returnForecast);
	
	py::class_<bvhar::CtaVharforecastRun<bvhar::CtaExpandforecastRun, bvhar::SvForecaster, false>>(m, "SvGrpVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &bvhar::CtaVharforecastRun<bvhar::CtaExpandforecastRun, bvhar::SvForecaster, false>::returnForecast);
	
	py::class_<bvhar::McmcSpilloverRun<bvhar::LdltRecords>>(m, "LdltSpillover")
		.def(py::init<int, int, py::dict&, bool>())
		.def(py::init<int, int, int, py::dict&, bool>())
		.def("returnSpillover", &bvhar::McmcSpilloverRun<bvhar::LdltRecords>::returnSpillover);
	
	py::class_<bvhar::DynamicLdltSpillover>(m, "LdltDynamicSpillover")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int, int, bool,
			py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int, bool,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, const Eigen::MatrixXi&, int>()
		)
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int, int, int, bool,
			py::dict&, py::dict&, py::dict&, std::vector<py::dict>&, int, bool,
			py::dict&, std::vector<py::dict>&, int,
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