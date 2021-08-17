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
