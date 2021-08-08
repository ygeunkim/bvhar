#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' Build a Linear Transformation Matrix for Vector HAR
//' 
//' @param m integer, dimension
//' @details
//' VHAR is linearly restricted VAR(22) in Y0 = X0 B + Z.
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP scale_har (int m) {
  Eigen::MatrixXd HAR = Eigen::MatrixXd::Zero(3, 22);
  Eigen::MatrixXd HARtrans(3 * m + 1, 22 * m + 1); // 3m x 22m
  Eigen::MatrixXd Im(m, m);
  Im.setIdentity(m, m);
  HAR(0, 0) = 1.0;
  for (int i = 0; i < 5; i++) {
    HAR(1, i) = 1.0 / 5;
  }
  for (int i = 0; i < 22; i++) {
    HAR(2, i) = 1.0 / 22;
  }
  // T otimes Im
  HARtrans.block(0, 0, 3 * m, 22 * m) = Eigen::kroneckerProduct(HAR, Im).eval();
  HARtrans.block(0, 22 * m, 3 * m, 1) = Eigen::MatrixXd::Zero(3 * m, 1);
  HARtrans.block(3 * m, 0, 1, 22 * m) = Eigen::MatrixXd::Zero(1, 22 * m);
  HARtrans(3 * m, 22 * m) = 1.0;
  return Rcpp::wrap(HARtrans);
}

//' Compute Vector HAR Coefficient Matrices and Fitted Values
//' 
//' @param x X0 processed by \code{\link{build_design}}
//' @param y Y0 processed by \code{\link{build_y0}}
//' @details
//' Given Y0 and Y0, the function estimate least squares
//' Y0 = X1 Phi + Z
//' 
//' @references
//' Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' 
//' Corsi, F. (2008). \emph{A Simple Approximate Long-Memory Model of Realized Volatility}. Journal of Financial Econometrics, 7(2), 174–196. \url{https://doi:10.1093/jjfinec/nbp001}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP estimate_har (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  int h = 3 * y.cols() + 1;
  Eigen::MatrixXd x1(y.rows(), h); // HAR design matrix
  Eigen::MatrixXd Phi(h, y.cols()); // HAR estimator
  Eigen::MatrixXd yhat(y.rows(), y.cols());
  Eigen::MatrixXd HARtrans = Rcpp::as<Eigen::MatrixXd>(scale_har(y.cols())); // linear transformation
  x1 = x * HARtrans.adjoint();
  Phi = (x1.adjoint() * x1).inverse() * x1.adjoint() * y; // estimation
  yhat = x1 * Phi;
  return Rcpp::List::create(
    Rcpp::Named("HARtrans") = Rcpp::wrap(HARtrans),
    Rcpp::Named("phihat") = Rcpp::wrap(Phi),
    Rcpp::Named("fitted") = Rcpp::wrap(yhat)
  );
}
