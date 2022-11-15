#include <RcppEigen.h>
#include "bvharmisc.h"
#include "bvharprob.h"

// [[Rcpp::depends(RcppEigen)]]

//' Building Spike-and-slab SD Diagonal Matrix
//' 
//' In MCMC process of SSVS, this function computes diagonal matrix \eqn{D} or \eqn{D_j} defined by spike-and-slab sd.
//' 
//' @param spike_sd Standard deviance for Spike normal distribution
//' @param slab_sd Standard deviance for Slab normal distribution
//' @param mixture_dummy Indicator vector (0-1) corresponding to each element
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd build_ssvs_sd(Eigen::VectorXd spike_sd,
                              Eigen::VectorXd slab_sd,
                              Eigen::VectorXd mixture_dummy) {
  int num_param = spike_sd.size();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(num_param, num_param);
  // diagonal term = spike_sd if mixture_dummy = 0 while slab_sd if mixture_dummy = 1
  for (int i = 0; i < num_param; i++) {
    res(i, i) = (1 - mixture_dummy[i]) * spike_sd[i] + mixture_dummy[i] * slab_sd[i];
  }
  return res;
}

//' Generating the Diagonal Component of Cholesky Factor in SSVS Gibbs Sampler
//' 
//' In MCMC process of SSVS, this function generates the diagonal component \eqn{\Psi} from variance matrix
//' 
//' @param sse_mat The result of \eqn{Z_0^T Z_0 = (Y_0 - X_0 \hat{A})^T (Y_0 - X_0 \hat{A})}
//' @param inv_DRD Inverse of matrix product between \eqn{D_j} and correlation matrix \eqn{R_j}
//' @param shape Gamma shape parameters for precision matrix
//' @param rate Gamma rate parameters for precision matrix
//' @param num_design The number of sample used, \eqn{n = T - p}
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_chol_diag(Eigen::MatrixXd sse_mat,
                               Eigen::MatrixXd inv_DRD,
                               Eigen::VectorXd shape,
                               Eigen::VectorXd rate,
                               int num_design) {
  int dim = sse_mat.cols();
  Eigen::VectorXd res(dim);
  Eigen::VectorXd sse_colvec(dim - 1); // sj = (s1j, ..., s(j-1, j)) from SSE
  shape.array() += (double)num_design / 2;
  rate[0] += sse_mat(0, 0) / 2;
  res[0] = sqrt(gamma_rand(shape[0], 1 / rate[0])); // psi[11]^2 ~ Gamma(shape, rate)
  int block_id = 0;
  for (int j = 1; j < dim; j++) {
    sse_colvec.segment(0, j) = sse_mat.block(0, j, j, 1); // (s1j, ..., sj-1,j)
    rate[j] += (
      sse_mat(j, j) - 
        sse_colvec.segment(0, j).transpose() * 
        (sse_mat.block(0, 0, j, j) + inv_DRD.block(block_id, block_id, j, j)).inverse() * 
        sse_colvec.segment(0, j)
    ) / 2;
    res[j] = sqrt(gamma_rand(shape[j], 1 / rate[j])); // psi[jj]^2 ~ Gamma(shape, rate)
    block_id += j;
  }
  return res;
}

//' Generating the Off-Diagonal Component of Cholesky Factor in SSVS Gibbs Sampler
//' 
//' In MCMC process of SSVS, this function generates the off-diagonal component \eqn{\Psi} of variance matrix
//' 
//' @param sse_mat The result of \eqn{Z_0^T Z_0 = (Y_0 - X_0 \hat{A})^T (Y_0 - X_0 \hat{A})}
//' @param chol_diag Diagonal element of the cholesky factor
//' @param inv_DRD Inverse of matrix product between \eqn{D_j} and correlation matrix \eqn{R_j}
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_chol_off(Eigen::MatrixXd sse_mat, 
                              Eigen::VectorXd chol_diag, 
                              Eigen::MatrixXd inv_DRD) {
  int dim = sse_mat.cols();
  Eigen::MatrixXd normal_variance(dim - 1, dim - 1);
  Eigen::VectorXd sse_colvec(dim - 1); // sj = (s1j, ..., s(j-1, j)) from SSE
  Eigen::VectorXd normal_mean(dim - 1);
  Eigen::VectorXd res(inv_DRD.cols());
  int block_id = 0;
  for (int j = 1; j < dim; j++) {
    sse_colvec.segment(0, j) = sse_mat.block(0, j, j, 1);
    normal_variance.block(0, 0, j, j) = (sse_mat.block(0, 0, j, j) + inv_DRD.block(block_id, block_id, j, j)).inverse();
    normal_mean.segment(0, j) = -chol_diag[j] * normal_variance.block(0, 0, j, j) * sse_colvec.segment(0, j);
    res.segment(block_id, j) = vectorize_eigen(sim_mgaussian_chol(1, normal_mean.segment(0, j), normal_variance.block(0, 0, j, j)));
    block_id += j;
  }
  return res;
}

//' Filling Cholesky Factor Upper Triangular Matrix
//' 
//' This function builds a cholesky factor matrix \eqn{\Psi} (upper triangular) using diagonal component vector and off-diagonal component vector.
//' 
//' @param diag_vec Diagonal components
//' @param off_diagvec Off-diagonal components
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd build_chol(Eigen::VectorXd diag_vec, Eigen::VectorXd off_diagvec) {
  int dim = diag_vec.size();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim, dim);
  res.diagonal() = diag_vec; // psi
  int id = 0; // length of eta = m(m-1)/2
  // assign eta_j = (psi_1j, ..., psi_j-1,j)
  for (int j = 1; j < dim; j++) {
    for (int i = 0; i < j; i++) {
      res(i, j) = off_diagvec[id + i]; // assign i-th row = psi_ij
    }
    id += j;
  }
  return res;
}

//' Generating Dummy Vector for Parameters in SSVS Gibbs Sampler
//' 
//' In MCMC process of SSVS, this function generates latent \eqn{\gamma_j} or \eqn{\omega_{ij}} conditional posterior.
//' 
//' @param param_obs Realized parameters vector
//' @param spike_sd Standard deviance for Spike normal distribution
//' @param slab_sd Standard deviance for Slab normal distribution
//' @param slab_weight Proportion of nonzero coefficients
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_chol_dummy(Eigen::VectorXd chol_upper, 
                                Eigen::VectorXd spike_sd,
                                Eigen::VectorXd slab_sd,
                                Eigen::VectorXd slab_weight) {
  double bernoulli_param_spike;
  double bernoulli_param_slab;
  int num_latent = slab_weight.size();
  Eigen::VectorXd res(num_latent); // latentj | Y0, -latentj ~ Bernoulli(u1 / (u1 + u2))
  for (int i = 0; i < num_latent; i++) {
    bernoulli_param_slab = slab_weight[i] * exp(- pow(chol_upper[i], 2.0) / (2 * pow(slab_sd[i], 2.0)) ) / slab_sd[i];
    bernoulli_param_spike = (1.0 - slab_weight[i]) * exp(- pow(chol_upper[i], 2.0) / (2 * pow(spike_sd[i], 2.0)) ) / spike_sd[i];
    res[i] = binom_rand(1.0, bernoulli_param_slab / (bernoulli_param_slab + bernoulli_param_spike)); // qj-bar
  }
  return res;
}

//' Generating Coefficient Vector in SSVS Gibbs Sampler
//' 
//' In MCMC process of SSVS, this function generates \eqn{\alpha_j} conditional posterior.
//' 
//' @param prior_mean The prior mean vector of the VAR coefficient vector
//' @param prior_prec The prior precision matrix of the VAR coefficient vector
//' @param XtX The result of design matrix arithmetic \eqn{X_0^T X_0}
//' @param coef_ols OLS (MLE) estimator of the VAR coefficient
//' @param chol_factor Cholesky factor of variance matrix
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_coef(Eigen::VectorXd prior_mean,
                          Eigen::MatrixXd prior_prec,
                          Eigen::MatrixXd XtX,
                          Eigen::VectorXd coef_ols,
                          Eigen::MatrixXd chol_factor) {
  Eigen::MatrixXd sig_inv_xtx = Eigen::kroneckerProduct(
    chol_factor * chol_factor.transpose(), 
    XtX
  ).eval(); // Sigma^(-1) otimes (X_0^T X_0) where Sigma^(-1) = chol * chol^T
  Eigen::MatrixXd normal_variance = (sig_inv_xtx + prior_prec).inverse(); // Delta
  Eigen::VectorXd normal_mean = normal_variance * (sig_inv_xtx * coef_ols + prior_prec * prior_mean); // mu
  return vectorize_eigen(sim_mgaussian_chol(1, normal_mean, normal_variance));
}

//' Generating Dummy Vector for Parameters in SSVS Gibbs Sampler
//' 
//' In MCMC process of SSVS, this function generates latent \eqn{\gamma_j} or \eqn{\omega_{ij}} conditional posterior.
//' 
//' @param param_obs Realized parameters vector
//' @param spike_sd Standard deviance for Spike normal distribution
//' @param slab_sd Standard deviance for Slab normal distribution
//' @param slab_weight Proportion of nonzero coefficients
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_coef_dummy(Eigen::VectorXd coef, 
                                Eigen::VectorXd spike_sd,
                                Eigen::VectorXd slab_sd,
                                Eigen::VectorXd slab_weight) {
  double bernoulli_param_spike;
  double bernoulli_param_slab;
  int num_latent = slab_weight.size();
  Eigen::VectorXd res(num_latent); // latentj | Y0, -latentj ~ Bernoulli(u1 / (u1 + u2))
  for (int i = 0; i < num_latent; i++) {
    bernoulli_param_spike = slab_weight[i] * exp(- pow(coef[i], 2.0) / (2 * pow(spike_sd[i], 2.0)) ) / spike_sd[i];
    bernoulli_param_slab = (1.0 - slab_weight[i]) * exp(- pow(coef[i], 2.0) / (2 * pow(slab_sd[i], 2.0)) ) / slab_sd[i];
    res[i] = binom_rand(1.0, bernoulli_param_spike / (bernoulli_param_slab + bernoulli_param_spike)); // qj-bar
  }
  return res;
}
