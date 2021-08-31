#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' @useDynLib bvhar
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd AAt_eigen (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd res(x.rows(), y.cols());
  res = x * y.adjoint();
  return res;
}

//' @useDynLib bvhar
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd tAA_eigen (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd res(x.cols(), y.rows());
  res = x.adjoint() * y;
  return res;
}

//' @useDynLib bvhar
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd AtAit_eigen (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd res(x.rows(), x.rows());
  res = x * y.inverse() * x.adjoint();
  return res;
}

//' @useDynLib bvhar
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd kroneckerprod (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd res(x.rows() * y.rows(), x.cols() * y.cols());
  res = kroneckerProduct(x, y).eval();
  return res;
}

//' @useDynLib bvhar
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_mgaussian (int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig) {
  int dim = sig.cols();
  Eigen::MatrixXd standard_normal(num_sim, dim);
  Eigen::MatrixXd res(num_sim, dim); // result: each column indicates variable
  for (int i = 0; i < num_sim; i++) {
    standard_normal.row(i) = Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(dim, 0.0, 1.0)); // Z1, ..., Zm ~ iid N(0, 1)
    standard_normal.row(i) += mu; // mean mu
  }
  res = standard_normal * sig.sqrt(); // epsilon(t) = Sigma^{1/2} Z(t)
  return res;
}
