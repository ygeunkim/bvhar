#include <bvhar/triangular>

PYBIND11_MODULE(_cta, m) {
	py::class_<baecon::bvhar::CtaRun<baecon::bvhar::McmcReg>>(m, "McmcLdlt")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &baecon::bvhar::CtaRun<baecon::bvhar::McmcReg>::returnRecords);
	
	py::class_<baecon::bvhar::CtaRun<baecon::bvhar::McmcReg, false>>(m, "McmcLdltGrp")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &baecon::bvhar::CtaRun<baecon::bvhar::McmcReg, false>::returnRecords);
	
	py::class_<baecon::bvhar::CtaRun<baecon::bvhar::McmcSv>>(m, "SvMcmc")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &baecon::bvhar::CtaRun<baecon::bvhar::McmcSv>::returnRecords);
	
	py::class_<baecon::bvhar::CtaRun<baecon::bvhar::McmcSv, false>>(m, "SvGrpMcmc")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &baecon::bvhar::CtaRun<baecon::bvhar::McmcSv, false>::returnRecords);
	
	py::class_<baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>>(m, "LdltForecast")
		.def(py::init<int, int, int, const Eigen::MatrixXd&, bool, double, py::dict&, const Eigen::VectorXi&, bool, bool, int, bool>())
		.def(py::init<int, int, int, int, const Eigen::MatrixXd&, bool, double, py::dict&, const Eigen::VectorXi&, bool, bool, int, bool>())
		.def("returnForecast", &baecon::bvhar::CtaForecastRun<baecon::bvhar::RegForecaster>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster>>(m, "LdltVarRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster>>(m, "LdltVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster>>(m, "LdltVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster>>(m, "LdltVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster, false>>(m, "LdltGrpVarRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster, false>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster, false>>(m, "LdltGrpVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster, false>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster, false>>(m, "LdltGrpVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::RegForecaster, false>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster, false>>(m, "LdltGrpVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::RegForecaster, false>::returnForecast);
	
	py::class_<baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>>(m, "SvForecast")
		.def(py::init<int, int, int, const Eigen::MatrixXd&, bool, double, py::dict&, const Eigen::VectorXi&, bool, bool, int, bool>())
		.def(py::init<int, int, int, int, const Eigen::MatrixXd&, bool, double, py::dict&, const Eigen::VectorXi&, bool, bool, int, bool>())
		.def("returnForecast", &baecon::bvhar::CtaForecastRun<baecon::bvhar::SvForecaster>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster>>(m, "SvVarRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster>>(m, "SvVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster>>(m, "SvVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster>>(m, "SvVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster, false>>(m, "SvGrpVarRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster, false>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster, false>>(m, "SvGrpVarExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVarforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster, false>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster, false>>(m, "SvGrpVharRoll")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaRollforecastRun, baecon::bvhar::SvForecaster, false>::returnForecast);
	
	py::class_<baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster, false>>(m, "SvGrpVharExpand")
		.def(
			py::init<const Eigen::MatrixXd&, int, int, int, int, int, int,
			bool, double, py::dict&, py::dict&, py::dict&,
			py::dict&, std::vector<py::dict>&, int,
			py::dict&, std::vector<py::dict>&, int,
			const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::MatrixXi&,
			bool, bool, int, const Eigen::MatrixXd&, bool, bool,
			const Eigen::MatrixXi&, const Eigen::VectorXi&, bool, int, bool>()
		)
		.def("returnForecast", &baecon::bvhar::CtaVharforecastRun<baecon::bvhar::CtaExpandforecastRun, baecon::bvhar::SvForecaster, false>::returnForecast);
	
	py::class_<baecon::bvhar::McmcSpilloverRun<baecon::bvhar::LdltRecords>>(m, "LdltSpillover")
		.def(py::init<int, int, py::dict&, bool>())
		.def(py::init<int, int, int, py::dict&, bool>())
		.def("returnSpillover", &baecon::bvhar::McmcSpilloverRun<baecon::bvhar::LdltRecords>::returnSpillover);
	
	py::class_<baecon::bvhar::DynamicLdltSpillover>(m, "LdltDynamicSpillover")
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
		.def("returnSpillover", &baecon::bvhar::DynamicLdltSpillover::returnSpillover);
	
	py::class_<baecon::bvhar::McmcSpilloverRun<baecon::bvhar::SvRecords>>(m, "SvSpillover")
		.def(py::init<int, int, py::dict&, bool>())
		.def(py::init<int, int, int, py::dict&, bool>())
		.def("returnSpillover", &baecon::bvhar::McmcSpilloverRun<baecon::bvhar::SvRecords>::returnSpillover);
	
	py::class_<baecon::bvhar::DynamicSvSpillover>(m, "SvDynamicSpillover")
		.def(py::init<int, int, int, py::dict&, bool, bool, int>())
		.def(py::init<int, int, int, int, py::dict&, bool, bool, int>())
		.def("returnSpillover", &baecon::bvhar::DynamicSvSpillover::returnSpillover);
}