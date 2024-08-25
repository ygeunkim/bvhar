#include <pybind11/pybind11.h>

int add2 (int i, int j) {
	return i + j;
}

PYBIND11_MODULE(minimal2, m) {
	m.def("add", &add2, "A function that adds two numbers");
}