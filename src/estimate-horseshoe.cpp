#ifdef _OPENMP
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#endif
#include <RcppEigen.h>
#include "bvharmisc.h"
#include "bvharprob.h"
#include <progress.hpp>
#include <progress_bar.hpp>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppProgress)]]

//' Building a Diagonal Matrix by Global and Local Hyperparameters
//' 
//' In MCMC process of Horseshoe, this function computes diagonal matrix \eqn{\Lambda_\ast} defined by
//' global and local sparsity levels.
//' 
//' @param global_hyperparam Global sparsity hyperparameter
//' @param local_hyperparam Local sparsity hyperparameters
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd build_shrink_mat(double global_hyperparam,
                                 Eigen::VectorXd local_hyperparam) {
  int num_param = local_hyperparam.size();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(num_param, num_param);
  res.diagonal() = local_hyperparam;
  return res * global_hyperparam;
}

//' Generating the Coefficient Vector in Horseshoe Gibbs Sampler
//' 
//' In MCMC process of Horseshoe prior, this function generates the coefficients vector.
//' 
//' @param response_vec Response vector for vectorized formulation
//' @param design_mat Design matrix for vectorized formulation
//' @param prior_var Variance constant of the likelihood
//' @param shrink_mat Diagonal matrix made by global and local sparsity hyperparameters
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd horseshoe_coef(Eigen::VectorXd response_vec,
                               Eigen::MatrixXd design_mat,
                               double prior_var,
                               Eigen::MatrixXd shrink_mat) {
  Eigen::MatrixXd unscaled_var = design_mat.transpose() * design_mat + shrink_mat.inverse();
  return vectorize_eigen(
    sim_mgaussian_chol(1, unscaled_var.inverse() * design_mat.transpose() * response_vec, prior_var * unscaled_var)
  );
}

//' Generating the Local Sparsity Hyperparameters Vector in Horseshoe Gibbs Sampler
//' 
//' In MCMC process of Horseshoe prior, this function generates the local sparsity hyperparameters vector.
//' 
//' @param local_latent Latent vectors defined for local sparsity vector
//' @param global_hyperparam Global sparsity hyperparameter
//' @param coef_vec Coefficients vector
//' @param prior_var Variance constant of the likelihood
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd horseshoe_local_sparsity(Eigen::VectorXd local_latent,
                                         double global_hyperparam,
                                         Eigen::VectorXd coef_vec,
                                         double prior_var) {
  int dim = coef_vec.size();
  Eigen::VectorXd res(dim);
  for (int i = 0; i < dim; i++) {
    res[i] = sqrt(1 /
        gamma_rand(
          1.0,
          1 / ( 1 / local_latent[i] + pow(coef_vec[i], 2.0) / (2 * pow(global_hyperparam * prior_var, 2.0)) )
        ));
  }
  return res;
}

//' Generating the Global Sparsity Hyperparameter in Horseshoe Gibbs Sampler
//' 
//' In MCMC process of Horseshoe prior, this function generates the global sparsity hyperparameter.
//' 
//' @param global_latent Latent variable defined for global sparsity hyperparameter
//' @param local_hyperparam Local sparsity hyperparameters vector
//' @param coef_vec Coefficients vector
//' @param prior_var Variance constant of the likelihood
//' @noRd
// [[Rcpp::export]]
double horseshoe_global_sparsity(double global_latent,
                                 Eigen::VectorXd local_hyperparam,
                                 Eigen::VectorXd coef_vec,
                                 double prior_var) {
  int dim = coef_vec.size();
  double invgam_scl = 1 / global_latent;
  for (int i = 0; i < dim; i++) {
    invgam_scl += pow(coef_vec[i], 2.0) / (2 * pow(local_hyperparam[i] * prior_var, 2.0));
  }
  return sqrt(1 / gamma_rand((dim + 1) / 2, 1 / invgam_scl));
}

//' Generating the Latent Vector for Local Sparsity Hyperparameters in Horseshoe Gibbs Sampler
//' 
//' In MCMC process of Horseshoe prior, this function generates the latent vector for local sparsity hyperparameters.
//' 
//' @param local_hyperparam Local sparsity hyperparameters vector
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd horseshoe_latent_local(Eigen::VectorXd local_hyperparam) {
  int dim = local_hyperparam.size();
  Eigen::VectorXd res(dim);
  for (int i = 0; i < dim; i++) {
    res[i] = 1 /
      gamma_rand(
        1.0,
        1 / (1 + 1 / pow(local_hyperparam[i], 2.0))
      );
  }
  return res;
}

//' Generating the Latent Vector for Local Sparsity Hyperparameters in Horseshoe Gibbs Sampler
//' 
//' In MCMC process of Horseshoe prior, this function generates the latent vector for global sparsity hyperparameters.
//' 
//' @param global_hyperparam Global sparsity hyperparameter
//' @noRd
// [[Rcpp::export]]
double horseshoe_latent_global(double global_hyperparam) {
  return 1 /
    gamma_rand(
      1.0,
      1 / (1 + 1 / pow(global_hyperparam, 2.0))
    );
}

//' Generating the Prior Variance Constant in Horseshoe Gibbs Sampler
//' 
//' In MCMC process of Horseshoe prior, this function generates the prior variance.
//' 
//' @param response_vec Response vector for vectorized formulation
//' @param design_mat Design matrix for vectorized formulation
//' @param coef_vec Coefficients vector
//' @param shrink_mat Diagonal matrix made by global and local sparsity hyperparameters
//' @noRd
// [[Rcpp::export]]
double horseshoe_prior_var(Eigen::VectorXd response_vec,
                           Eigen::MatrixXd design_mat,
                           Eigen::VectorXd coef_vec,
                           Eigen::MatrixXd shrink_mat) {
  Eigen::VectorXd resid = response_vec - design_mat * coef_vec;
  return 1 /
    gamma_rand(
      (response_vec.size() + coef_vec.size()) / 2,
      1 / (resid.sqrt().sum() / 2 + coef_vec.transpose() * shrink_mat.inverse() * coef_vec)
    );
}


