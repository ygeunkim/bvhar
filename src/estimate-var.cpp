#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

//' Compute VAR(p) Coefficient Matrices and Fitted Values
//' 
//' @param x X0 processed by \code{\link{build_design}}
//' @param y Y0 processed by \code{\link{build_y0}}
//' @details
//' Given Y0 and Y0, the function estimate least squares
//' Y0 = X0 B + Z
//' 
//' @references Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
//' 
//' @useDynLib bvhar
//' @importFrom Rcpp sourceCpp
//' @export
// [[Rcpp::export]]
SEXP estimate_var (Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd B(x.cols(), y.cols()); // bhat
  Eigen::MatrixXd yhat(y.rows(), y.cols());
  B = (x.adjoint() * x).inverse() * x.adjoint() * y;
  yhat = x * B;
  return Rcpp::List::create(
    Rcpp::Named("bhat") = Rcpp::wrap(B),
    Rcpp::Named("fitted") = Rcpp::wrap(yhat)
  );
}
