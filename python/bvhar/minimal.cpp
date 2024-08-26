#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

void process_matrix(Eigen::Ref<const Eigen::MatrixXd> matrix) {
    std::cout << "Matrix size: " << matrix.rows() << "x" << matrix.cols() << std::endl;
    std::cout << "First element: " << matrix(0,0) << std::endl;
}

void modify_matrix(Eigen::Ref<Eigen::MatrixXd> matrix) {
    matrix *= 2;
}

PYBIND11_MODULE(minimal, m) {
    m.def("process_matrix", &process_matrix, "Process a matrix");
    m.def("modify_matrix", &modify_matrix, "Modify a matrix");
}
