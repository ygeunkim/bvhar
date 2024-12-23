#include <bvharspillover.h>

PYBIND11_MODULE(_svspillover, m) {
	py::class_<bvhar::McmcSpilloverRun<bvhar::SvRecords>>(m, "SvSpillover")
		.def(py::init<int, int, py::dict&, bool>())
		.def(py::init<int, int, int, py::dict&, bool>())
		.def("returnSpillover", &bvhar::McmcSpilloverRun<bvhar::SvRecords>::returnSpillover);
	
	py::class_<bvhar::DynamicSvSpillover>(m, "SvDynamicSpillover")
		.def(py::init<int, int, int, py::dict&, bool, bool, int>())
		.def(py::init<int, int, int, int, py::dict&, bool, bool, int>())
		.def("returnSpillover", &bvhar::DynamicSvSpillover::returnSpillover);
}