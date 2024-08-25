#include <pybind11/pybind11.h>

int multiply(int i, int j) {
	return i * j;
}

PYBIND11_MODULE(subminimal, m) {
	m.def("multiply", &multiply, "A function that multiplies two numbers");
}