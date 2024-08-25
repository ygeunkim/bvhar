#include <pybind11/pybind11.h>
#include <bvharcommon.h>
#include <pybind11/eigen.h>

double tmp(unsigned int seed) {
	boost::random::mt19937 rng(seed);
	return bvhar::normal_rand(rng);
}

PYBIND11_MODULE(minimal2, m) {
	m.def("tmp", &tmp, "Temporary to check inst/include works");
}

// int add2 (int i, int j) {
// 	return i + j;
// }

// PYBIND11_MODULE(minimal2, m) {
// 	m.def("add", &add2, "A function that adds two numbers");
// }