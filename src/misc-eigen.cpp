#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd AAt_eigen (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd res(x.rows(), y.cols());
  res = x * y.transpose();
  return res;
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd tAA_eigen (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd res(x.cols(), y.rows());
  res = x.transpose() * y;
  return res;
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd AtAit_eigen (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd res(x.rows(), x.rows());
  res = x * y.inverse() * x.transpose();
  return res;
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd kroneckerprod (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd res(x.rows() * y.rows(), x.cols() * y.cols());
  res = kroneckerProduct(x, y).eval();
  return res;
}

//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd compute_eigenvalues(Eigen::Map<Eigen::MatrixXd> x) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(x);
  return es.eigenvalues();
}

//' Generate Multivariate Normal Random Vector with Zero Mean
//' 
//' This function samples n x muti-dimensional normal random matrix with zero mean vector.
//' 
//' @param num_sim Number to generate process
//' @param sig Variance matrix
//' 
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_mgaussian (int num_sim, Eigen::MatrixXd sig) {
  int dim = sig.cols();
  if (sig.rows() != dim) Rcpp::stop("Invalid 'sig' dimension.");
  Eigen::MatrixXd standard_normal(num_sim, dim);
  Eigen::MatrixXd res(num_sim, dim); // result: each column indicates variable
  for (int i = 0; i < num_sim; i++) {
    standard_normal.row(i) = Rcpp::as<Eigen::VectorXd>(Rcpp::rnorm(dim, 0.0, 1.0)); // Z1, ..., Zm ~ iid N(0, 1)
  }
  res = standard_normal * sig.sqrt(); // epsilon(t) = Sigma^{1/2} Z(t)
  return res;
}

