#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP compute_var (Eigen::MatrixXd z, int s, int k) {
  Eigen::MatrixXd Sig(z.cols(), z.cols());
  Sig = z.adjoint() * z / (s - k);
  return Rcpp::wrap(Sig);
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
