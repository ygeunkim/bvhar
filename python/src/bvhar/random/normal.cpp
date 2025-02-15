// #include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <bvhar/src/math/random.h>

// namespace py = pybind11;

// MVN
Eigen::MatrixXd generate_mnormal(int num_sim, Eigen::VectorXd mean, Eigen::MatrixXd covariance, unsigned int seed, int method) {
	BHRNG rng(seed);
	int dim = covariance.cols();
	if (covariance.rows() != dim) {
		throw py::value_error("Invalid 'covariance' dimension.");
	}
	if (dim != mean.size()) {
		throw py::value_error("Invalid 'mean' size.");
	}
	Eigen::MatrixXd standard_normal(num_sim, dim);
	Eigen::MatrixXd res(num_sim, dim);
	for (int i = 0; i < num_sim; i++) {
    for (int j = 0; j < standard_normal.cols(); j++) {
      standard_normal(i, j) = bvhar::normal_rand(rng);
    }
  }
	switch (method) {
		case 1: {
			// res = standard_normal * covariance.sqrt();
			throw py::value_error("Use eigen decomposition later");
			break;
		}
		case 2: {
			res = standard_normal * covariance.llt().matrixU(); // use upper because now dealing with row vectors
			break;
		}
		default: {
			throw py::value_error("No 'method' defined");
		}
	}
	res.rowwise() += mean.transpose();
	return res;
}

// MNIW

// GIG

PYBIND11_MODULE(normal, m) {
	m.doc() = "Random samplers related to Gaussain";

	m.def("generate_mnormal", &generate_mnormal, "Generates multivariate gaussian random vectors");
}
