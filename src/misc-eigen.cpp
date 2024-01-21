#include <RcppEigen.h>
#include "bvharprob.h"

Eigen::MatrixXd kronecker_eigen(Eigen::MatrixXd x, Eigen::MatrixXd y) {
  Eigen::MatrixXd res = Eigen::kroneckerProduct(x, y).eval();
  return res;
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
  if (p < 1) {
    Rcpp::stop("'p' should be larger than or same as 1.");
  }
  if (x <= 0) {
    Rcpp::stop("'x' should be larger than 0.");
  }
  if (p == 1) {
    return gammafn(x);
  }
  if (2 * x < p) {
    Rcpp::stop("'x / 2' should be larger than 'p'.");
  }
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
  if (p < 1) {
    Rcpp::stop("'p' should be larger than or same as 1.");
  }
  if (x <= 0) {
    Rcpp::stop("'x' should be larger than 0.");
  }
  if (p == 1) {
    return lgammafn(x);
  }
  if (2 * x < p) {
    Rcpp::stop("'x / 2' should be larger than 'p'.");
  }
  double res = p * (p - 1) / 4.0 * log(M_PI);
  for (int i = 0; i < p; i++) {
    res += lgammafn(x - i / 2.0);
  }
  return res;
}

//' Density of Inverse Gamma Distribution
//' 
//' Compute the pdf of Inverse Gamma distribution
//' 
//' @param x non-negative argument
//' @param shp Shape of the distribution
//' @param scl Scale of the distribution
//' @param lg If true, return log(f)
//' 
//' @noRd
// [[Rcpp::export]]
double invgamma_dens(double x, double shp, double scl, bool lg) {
  if (x < 0 ) {
    Rcpp::stop("'x' should be larger than 0.");
  }
  if (shp <= 0 ) {
    Rcpp::stop("'shp' should be larger than 0.");
  }
  if (scl <= 0 ) {
    Rcpp::stop("'scl' should be larger than 0.");
  }
  double res = pow(scl, shp) * pow(x, -shp - 1) * exp(-scl / x) / gammafn(shp);
  if (lg) {
    return log(res);
  }
  return res;
}
