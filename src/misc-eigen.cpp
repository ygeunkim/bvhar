#include <RcppEigen.h>
#include "bvharprob.h"

// [[Rcpp::depends(RcppEigen)]]

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd kronecker_eigen(Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd res = Eigen::kroneckerProduct(x, y).eval();
  return res;
}

//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd vectorize_eigen(Eigen::MatrixXd x) {
  Eigen::VectorXd res = Eigen::Map<Eigen::VectorXd>(x.transpose().data(), x.rows() * x.cols());
  return res;
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd unvectorize(Eigen::VectorXd x, int num_rows, int num_cols) {
  // igen::Map<Eigen::MatrixXd>(coef_record.block(num_iter, b * num_coef, 1, num_coef).data(), dim_design, dim);
  Eigen::MatrixXd res = Eigen::Map<Eigen::MatrixXd>(x.data(), num_rows, num_cols);
  return res;
}

//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd compute_eigenvalues(Eigen::Map<Eigen::MatrixXd> x) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(x);
  return es.eigenvalues();
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_inverse(Eigen::MatrixXd x) {
  return x.inverse();
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_choleksy_lower(Eigen::MatrixXd x) {
  Eigen::LLT<Eigen::MatrixXd> lltOfscale(x);
  return lltOfscale.matrixL(); // lower triangular matrix
}

//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd compute_choleksy_upper(Eigen::MatrixXd x) {
  Eigen::LLT<Eigen::MatrixXd> lltOfscale(x);
  return lltOfscale.matrixU(); // upper triangular matrix
}

//' @noRd
// [[Rcpp::export]]
Rcpp::List qr_eigen(Eigen::Map<Eigen::MatrixXd> x) {
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(x); // x = QR
  Eigen::MatrixXd q = qr.householderQ(); // Q orthogonal
  Eigen::MatrixXd r = qr.matrixQR().triangularView<Eigen::Upper>(); // R upper
  return Rcpp::List::create(
    Rcpp::Named("orthogonal") = q,
    Rcpp::Named("upper") = r
  );
}

//' Multivariate Gamma Function
//' 
//' Compute multivariate gamma function numerically
//' 
//' @param x Double, non-negative argument
//' @param p Integer, dimension
//' 
//' @noRd
// [[Rcpp::export]]
double mgammafn(double x, int p) {
  if (p < 1) Rcpp::stop("'p' should be larger than or same as 1.");
  if (x <= 0) Rcpp::stop("'x' should be larger than 0.");
  if (p == 1) return gammafn(x);
  if (2 * x < p) Rcpp::stop("'x / 2' should be larger than 'p'.");
  double res = pow(M_PI, p * (p - 1) / 4.0);
  for (int i = 0; i < p; i++) {
    res *= gammafn(x - i / 2.0); // x + (1 - j) / 2
  }
  return res;
}

//' Log of Multivariate Gamma Function
//' 
//' Compute log of multivariate gamma function numerically
//' 
//' @param x Double, non-negative argument
//' @param p Integer, dimension
//' 
//' @noRd
// [[Rcpp::export]]
double log_mgammafn(double x, int p) {
  if (p < 1) Rcpp::stop("'p' should be larger than or same as 1.");
  if (x <= 0) Rcpp::stop("'x' should be larger than 0.");
  if (p == 1) return gammafn(x);
  if (2 * x < p) Rcpp::stop("'x / 2' should be larger than 'p'.");
  double res = p * (p - 1) / 4.0 * log(M_PI);
  for (int i = 0; i < p; i++) {
    res += lgammafn(x - i / 2.0);
  }
  return res;
}
