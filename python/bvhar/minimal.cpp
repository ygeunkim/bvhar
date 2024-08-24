#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

int add (int i, int j) {
	return i + j;
}

PYBIND11_MODULE(minimal, m) {
	m.def("add", &add, "A function that adds two numbers");
}