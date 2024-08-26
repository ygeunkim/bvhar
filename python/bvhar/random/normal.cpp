#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <bvharsim.h>

namespace py = pybind11;

Eigen::MatrixXd sim_mgaussian(int num_sim, Eigen::Ref<Eigen::VectorXd> mean, Eigen::Ref<Eigen::MatrixXd> covariance, unsigned int seed, int method) {
	boost::random::mt19937 rng(seed);
	int dim = covariance.cols();
	if (covariance.rows() != dim) {
		// STOP
	}
	if (dim != mean.size()) {
		// STOP
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
			res = standard_normal * covariance.sqrt();
			res.rowwise() += mean.transpose();
			break;
		}
		case 2: {
			res = standard_normal * covariance.llt().matrixU(); // use upper because now dealing with row vectors
			res.rowwise() += covariance.transpose();
			break;
		}
		// default: STOP
	}
	return res;
}

PYBIND11_MODULE(normal, m) {
	m.doc() = "Samples n x muti-dimensional normal random matrix";
	m.def("sim_mgaussian", &sim_mgaussian, "A function that generates multivariate gaussian random vectors");
}
