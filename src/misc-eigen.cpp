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
