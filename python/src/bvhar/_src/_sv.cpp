#include <bvharmcmc.h>

PYBIND11_MODULE(_sv, m) {
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
}