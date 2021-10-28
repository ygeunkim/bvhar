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
    for (int j = 0; j < standard_normal.cols(); j++) {
      standard_normal(i, j) = norm_rand();
    }
  }
  res = standard_normal * sig.sqrt(); // epsilon(t) = Sigma^{1/2} Z(t)
  return res;
}

//' Generate Matrix Normal Random Matrix
//' 
//' This function samples one matrix gaussian matrix.
//' 
//' @param mat_mean Mean matrix
//' @param mat_scale_u First scale matrix
//' @param mat_scale_v Second scale matrix
//' 
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_matgaussian(Eigen::MatrixXd mat_mean, Eigen::Map<Eigen::MatrixXd> mat_scale_u, Eigen::Map<Eigen::MatrixXd> mat_scale_v) {
  Eigen::LLT<Eigen::MatrixXd> lltOfscaleu(mat_scale_u);
  Eigen::LLT<Eigen::MatrixXd> lltOfscalev(mat_scale_v);
  // Cholesky decomposition (lower triangular)
  Eigen::MatrixXd chol_scale_u = lltOfscaleu.matrixL();
  Eigen::MatrixXd chol_scale_v = lltOfscalev.matrixL();
  // standard normal
  int num_rows = mat_mean.rows();
  int num_cols = mat_mean.cols();
  Eigen::MatrixXd mat_norm(num_rows, num_cols);
  // Eigen::MatrixXd res(num_rows, num_cols, num_sim);
  Eigen::MatrixXd res(num_rows, num_cols);
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      mat_norm(i, j) = norm_rand();
    }
  }
  res = mat_mean + chol_scale_u * mat_norm * chol_scale_v.transpose();
  return res;
}
