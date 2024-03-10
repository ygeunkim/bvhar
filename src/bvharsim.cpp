#include "bvharsim.h"

//' Generate Multivariate Normal Random Vector
//' 
//' This function samples n x muti-dimensional normal random matrix.
//' 
//' @param num_sim Number to generate process
//' @param mu Mean vector
//' @param sig Variance matrix
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_mgaussian(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig) {
  int dim = sig.cols();
  if (sig.rows() != dim) {
    Rcpp::stop("Invalid 'sig' dimension.");
  }
  if (dim != mu.size()) {
    Rcpp::stop("Invalid 'mu' size.");
  }
  Eigen::MatrixXd standard_normal(num_sim, dim);
  Eigen::MatrixXd res(num_sim, dim); // result: each column indicates variable
  for (int i = 0; i < num_sim; i++) {
    for (int j = 0; j < standard_normal.cols(); j++) {
      standard_normal(i, j) = norm_rand();
    }
  }
  res = standard_normal * sig.sqrt(); // epsilon(t) = Sigma^{1/2} Z(t)
  res.rowwise() += mu.transpose();
  return res;
}

//' Generate Multivariate Normal Random Vector using Cholesky Decomposition
//' 
//' This function samples n x muti-dimensional normal random matrix with using Cholesky decomposition.
//' 
//' @param num_sim Number to generate process
//' @param mu Mean vector
//' @param sig Variance matrix
//' @details
//' This function computes \eqn{\Sigma^{1/2}} by choleksy decomposition.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_mgaussian_chol(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig) {
  int dim = sig.cols();
  if (sig.rows() != dim) {
    Rcpp::stop("Invalid 'sig' dimension.");
  }
  if (dim != mu.size()) {
    Rcpp::stop("Invalid 'mu' size.");
  }
  Eigen::MatrixXd standard_normal(num_sim, dim);
  Eigen::MatrixXd res(num_sim, dim); // result: each column indicates variable
  for (int i = 0; i < num_sim; i++) {
    for (int j = 0; j < standard_normal.cols(); j++) {
      standard_normal(i, j) = norm_rand();
    }
  }
  // Eigen::LLT<Eigen::MatrixXd> lltOfscale(sig);
  // Eigen::MatrixXd sig_sqrt = lltOfscale.matrixU(); // use upper because now dealing with row vectors
  res = standard_normal * sig.llt().matrixU(); // use upper because now dealing with row vectors
  res.rowwise() += mu.transpose();
  return res;
}
// overloading: add rng instance
Eigen::MatrixXd sim_mgaussian_chol(int num_sim, Eigen::VectorXd mu, Eigen::MatrixXd sig, boost::random::mt19937& rng) {
  int dim = sig.cols();
  Eigen::MatrixXd standard_normal(num_sim, dim);
  Eigen::MatrixXd res(num_sim, dim);
  for (int i = 0; i < num_sim; i++) {
    for (int j = 0; j < standard_normal.cols(); j++) {
      standard_normal(i, j) = bvhar::normal_rand(rng);
    }
  }
  res = standard_normal * sig.llt().matrixU(); // use upper because now dealing with row vectors
  res.rowwise() += mu.transpose();
  return res;
}

//' Generate Multivariate t Random Vector
//' 
//' This function samples n x muti-dimensional normal random matrix.
//' 
//' @param num_sim Number to generate process
//' @param df Degrees of freedom
//' @param mu Location vector
//' @param sig Scale matrix
//' @param method Method to compute \eqn{\Sigma^{1/2}}. 1: spectral decomposition, 2: Cholesky.
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd sim_mstudent(int num_sim, double df, Eigen::VectorXd mu, Eigen::MatrixXd sig, int method) {
  int dim = sig.cols();
  if (sig.rows() != dim) {
    Rcpp::stop("Invalid 'sig' dimension.");
  }
  if (dim != mu.size()) {
    Rcpp::stop("Invalid 'mu' size.");
  }
  Eigen::MatrixXd res(num_sim, dim);
  switch (method) {
  case 1:
    res = sim_mgaussian(num_sim, Eigen::VectorXd::Zero(dim), sig);
    break;
  case 2:
    res = sim_mgaussian_chol(num_sim, Eigen::VectorXd::Zero(dim), sig);
    break;
  default:
    Rcpp::stop("Invalid 'method' option.");
  }
  for (int i = 0; i < num_sim; i++) {
    res.row(i) *= sqrt(df / bvhar::chisq_rand(df));
  }
  res.rowwise() += mu.transpose();
  return res;
}

//' Generate Matrix Normal Random Matrix
//' 
//' This function samples one matrix gaussian matrix.
//' 
//' @param mat_mean Mean matrix
//' @param mat_scale_u First scale matrix
//' @param mat_scale_v Second scale matrix
//' @details
//' Consider n x k matrix \eqn{Y_1, \ldots, Y_n \sim MN(M, U, V)} where M is n x k, U is n x n, and V is k x k.
//' 
//' 1. Lower triangular Cholesky decomposition: \eqn{U = P P^T} and \eqn{V = L L^T}
//' 2. Standard normal generation: s x m matrix \eqn{Z_i = [z_{ij} \sim N(0, 1)]} in row-wise direction.
//' 3. \eqn{Y_i = M + P Z_i L^T}
//' 
//' This function only generates one matrix, i.e. \eqn{Y_1}.
//' @return One n x k matrix following MN distribution.
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_matgaussian(Eigen::MatrixXd mat_mean, Eigen::MatrixXd mat_scale_u, Eigen::MatrixXd mat_scale_v) {
  if (mat_scale_u.rows() != mat_scale_u.cols()) {
    Rcpp::stop("Invalid 'mat_scale_u' dimension.");
  }
  if (mat_mean.rows() != mat_scale_u.rows()) {
    Rcpp::stop("Invalid 'mat_scale_u' dimension.");
  }
  if (mat_scale_v.rows() != mat_scale_v.cols()) {
    Rcpp::stop("Invalid 'mat_scale_v' dimension.");
  }
  if (mat_mean.cols() != mat_scale_v.rows()) {
    Rcpp::stop("Invalid 'mat_scale_v' dimension.");
  }
  // Eigen::LLT<Eigen::MatrixXd> lltOfscaleu(mat_scale_u);
  // Eigen::LLT<Eigen::MatrixXd> lltOfscalev(mat_scale_v);
  // // Cholesky decomposition (lower triangular)
  // Eigen::MatrixXd chol_scale_u = lltOfscaleu.matrixL();
  // Eigen::MatrixXd chol_scale_v = lltOfscalev.matrixL();
  // // standard normal
  // Eigen::MatrixXd mat_norm(num_rows, num_cols);
  // // Eigen::MatrixXd res(num_rows, num_cols, num_sim);
  // Eigen::MatrixXd res(num_rows, num_cols);
  // for (int i = 0; i < num_rows; i++) {
  //   for (int j = 0; j < num_cols; j++) {
  //     mat_norm(i, j) = norm_rand();
  //   }
  // }
  // res = mat_mean + chol_scale_u * mat_norm * chol_scale_v.transpose();
  // return res;
	return bvhar::sim_mn(mat_mean, mat_scale_u, mat_scale_v);
}

//' Generate Inverse-Wishart Random Matrix
//' 
//' This function samples one matrix IW matrix.
//' 
//' @param mat_scale Scale matrix
//' @param shape Shape
//' @details
//' Consider \eqn{\Sigma \sim IW(\Psi, \nu)}.
//' 
//' 1. Upper triangular Bartlett decomposition: k x k matrix \eqn{Q = [q_{ij}]} upper triangular with
//'     1. \eqn{q_{ii}^2 \chi_{\nu - i + 1}^2}
//'     2. \eqn{q_{ij} \sim N(0, 1)} with i < j (upper triangular)
//' 2. Lower triangular Cholesky decomposition: \eqn{\Psi = L L^T}
//' 3. \eqn{A = L (Q^{-1})^T}
//' 4. \eqn{\Sigma = A A^T \sim IW(\Psi, \nu)}
//' @return One k x k matrix following IW distribution
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sim_iw(Eigen::MatrixXd mat_scale, double shape) {
  // Eigen::MatrixXd chol_res = bvhar::sim_iw_tri(mat_scale, shape);
  // Eigen::MatrixXd res = chol_res * chol_res.transpose(); // dim x dim
  // return res;
	return bvhar::sim_inv_wishart(mat_scale, shape);
}

//' Generate Normal-IW Random Family
//' 
//' This function samples normal inverse-wishart matrices.
//' 
//' @param num_sim Number to generate
//' @param mat_mean Mean matrix of MN
//' @param mat_scale_u First scale matrix of MN
//' @param mat_scale Scale matrix of IW
//' @param shape Shape of IW
//' @noRd
// [[Rcpp::export]]
Rcpp::List sim_mniw_export(int num_sim, Eigen::MatrixXd mat_mean, Eigen::MatrixXd mat_scale_u, Eigen::MatrixXd mat_scale, double shape) {
  // int ncol_mn = mat_mean.cols();
  // int nrow_mn = mat_mean.rows();
  // int dim_iw = mat_scale.cols();
  // if (dim_iw != mat_scale.rows()) {
  //   Rcpp::stop("Invalid 'mat_scale' dimension.");
  // }
  // Eigen::MatrixXd chol_res(dim_iw, dim_iw);
  // Eigen::MatrixXd mat_scale_v(dim_iw, dim_iw);
  // // result matrices: bind in column wise
  // Eigen::MatrixXd res_mn(nrow_mn, num_sim * ncol_mn); // [Y1, Y2, ..., Yn]
  // Eigen::MatrixXd res_iw(dim_iw, num_sim * dim_iw); // [Sigma1, Sigma2, ... Sigma2]
  // for (int i = 0; i < num_sim; i++) {
  //   chol_res = bvhar::sim_iw_tri(mat_scale, shape);
  //   mat_scale_v = chol_res * chol_res.transpose();
  //   res_iw.block(0, i * dim_iw, dim_iw, dim_iw) = mat_scale_v;
  //   // MN(mat_mean, mat_scale_u, mat_scale_v)
  //   res_mn.block(0, i * ncol_mn, nrow_mn, ncol_mn) = sim_matgaussian(
  //     mat_mean, 
  //     mat_scale_u, 
  //     mat_scale_v
  //   );
  // }
	// return Rcpp::List::create(
  //   Rcpp::Named("mn") = res_mn,
  //   Rcpp::Named("iw") = res_iw
  // );
	// Rcpp::List res = Rcpp::wrap(bvhar::sim_mn_iw(mat_mean, mat_scale_u, mat_scale, shape));
	// res.attr("names") = Rcpp::CharacterVector::create("mn", "iw");
	// return res;
	// std::vector<std::vector<Eigen::MatrixXd>> res(num_horizon, std::vector<Eigen::MatrixXd>(num_chains));
	std::vector<std::vector<Eigen::MatrixXd>> res(num_sim, std::vector<Eigen::MatrixXd>(2));
	for (int i = 0; i < num_sim; i++) {
		res[i] = bvhar::sim_mn_iw(mat_mean, mat_scale_u, mat_scale, shape);
  }
	return Rcpp::wrap(res);
}

//' Generate Generalized Inverse Gaussian
//' 
//' This function samples GIG(lambda, psi, chi) random variates.
//' 
//' @param num_sim Number to generate process
//' @param lambda Index of modified Bessel function of third kind.
//' @param psi Second parameter of GIG
//' @param chi Third parameter of GIG
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd sim_gig_export(int num_sim, double lambda, double psi, double chi) {
	return bvhar::sim_gig(num_sim, lambda, psi, chi);
}
