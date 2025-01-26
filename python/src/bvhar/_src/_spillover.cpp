#include <bvharspillover.h>

PYBIND11_MODULE(_ldltspillover, m) {
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
}