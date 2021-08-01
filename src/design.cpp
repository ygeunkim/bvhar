#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' Build Y0 matrix in VAR(p)
//' 
//' @param x Matrix, time series data
//' @param p VAR lag
//' @param t starting index to extract
//' 
//' @details
//' Given data Y,
//' \deqn{Y0 = [y_t^T, y_{t + 1}^T, \ldots, y_{t + n - p - 1}^T]^T}
//' is the (n - p) x m matrix.
//' 
//' In case of Y0, t = p + 1.
//' This function is used when constructing X0.
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP build_y0(Eigen::MatrixXd x, int p, int t) {
  int s = x.rows() - p;
  int m = x.cols();
  Eigen::MatrixXd res(s, m); // Y0
  for (int i = 0; i < s; i++) {
    res.row(i) = x.row(t + i - 1);
  }
  return Rcpp::wrap(res);
}

//' Build X0 matrix in VAR(p)
//' 
//' @param x Matrix, time series data
//' @param p VAR lag
//' 
//' @details
//' X0 is
//' \deqn{X0 = [Y_p, \ldots, Y_1, 1]}
//' i.e. (n - p) x (mp + 1) matrix
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP build_design(Eigen::MatrixXd x, int p) {
  int s = x.rows() - p;
  int m = x.cols();
  int k = m * p + 1;
  Eigen::MatrixXd res(s, k); // X0
  for (int t = 0; t < p; t++) {
    res.block(0, t * m, s, m) = Rcpp::as<Eigen::MatrixXd>(build_y0(x, p, p - t));
  }
  for (int i = 0; i < s; i++) {
    res(i, k - 1) = 1.0;
  }
  return Rcpp::wrap(res);
}
