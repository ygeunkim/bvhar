#include <bvharmcmc.h>

PYBIND11_MODULE(_ldlt, m) {
	// py::class_<McmcLdlt>(m, "McmcLdlt")
	py::class_<bvhar::McmcRun<bvhar::McmcReg>>(m, "McmcLdlt")
		.def(
			py::init<int, int, int, int, const Eigen::MatrixXd&, const Eigen::MatrixXd&,
			py::dict&, py::dict&, py::dict&,
			std::vector<py::dict>&, int, const Eigen::VectorXi&, const Eigen::VectorXi&, const Eigen::VectorXi&,
			const Eigen::MatrixXi&, bool, const Eigen::VectorXi&, bool, int>()
		)
		.def("returnRecords", &bvhar::McmcRun<bvhar::McmcReg>::returnRecords);
		// .def("returnRecords", &McmcLdlt::returnRecords);
}