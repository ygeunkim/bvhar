#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP AAt_eigen (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd res(x.rows(), y.cols());
  res = x * y.adjoint();
  return Rcpp::wrap(res);
}

//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP tAA_eigen (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd res(x.cols(), y.rows());
  res = x.adjoint() * y;
  return Rcpp::wrap(res);
}

//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP kroneckerprod (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd res(x.rows() * y.rows(), x.cols() * y.cols());
  res = kroneckerProduct(x, y).eval();
  return Rcpp::wrap(res);
}
