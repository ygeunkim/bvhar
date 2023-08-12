#include "bvharomp.h"
#include <RcppEigen.h>
#include "bvharmisc.h"
#include "bvharprob.h"
#include <progress.hpp>
#include <progress_bar.hpp>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppProgress)]]

//' Building a Inverse Diagonal Matrix by Global and Local Hyperparameters
//' 
//' In MCMC process of Horseshoe, this function computes diagonal matrix \eqn{\Lambda_\ast^{-1}} defined by
//' global and local sparsity levels.
//' 
//' @param global_hyperparam Global sparsity hyperparameter
//' @param local_hyperparam Local sparsity hyperparameters
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd build_shrink_mat(double global_hyperparam, Eigen::VectorXd local_hyperparam) {
  int num_param = local_hyperparam.size();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(num_param, num_param);
  res.diagonal() = 1 / local_hyperparam.array().square();
  return res / pow(global_hyperparam, 2.0);
}

//' Generating the Coefficient Vector in Horseshoe Gibbs Sampler
//' 
//' In MCMC process of Horseshoe prior, this function generates the coefficients vector.
//' 
//' @param response_vec Response vector for vectorized formulation
//' @param design_mat Design matrix for vectorized formulation
//' @param shrink_mat Diagonal matrix made by global and local sparsity hyperparameters
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd horseshoe_coef(Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat, double var, Eigen::MatrixXd shrink_mat) {
  Eigen::MatrixXd prec_mat = (design_mat.transpose() * design_mat + shrink_mat).llt().solve(
    Eigen::MatrixXd::Identity(design_mat.cols(), design_mat.cols())
  );
  return sim_mgaussian_chol(1, prec_mat * design_mat.transpose() * response_vec, var * prec_mat);
}

//' Generating the Coefficient Vector using Fast Sampling
//' 
//' In MCMC process of Horseshoe prior, this function generates the coefficients vector.
//' 
//' @param response_vec Response vector for vectorized formulation
//' @param design_mat Design matrix for vectorized formulation
//' @param shrink_mat Diagonal matrix made by global and local sparsity hyperparameters
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd horseshoe_fast_coef(Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat, Eigen::MatrixXd shrink_mat) {
  int num_coef = design_mat.cols(); // k^2 kp(+1)
  int num_sur = response_vec.size(); // nk-dim
  // Eigen::VectorXd scaled_y = response_vec / sqrt(var);
  // Eigen::MatrixXd scaled_x = design_mat / sqrt(var); // num_sur x num_coef
  // Eigen::MatrixXd scaled_lambda = var * shrink_mat; // num_coef x num_coef
  Eigen::MatrixXd sur_identity = Eigen::MatrixXd::Identity(num_sur, num_sur);
  Eigen::VectorXd u_vec = vectorize_eigen(sim_mgaussian_chol(1, Eigen::VectorXd::Zero(num_coef), shrink_mat));
  Eigen::VectorXd delta_vec = vectorize_eigen(sim_mgaussian_chol(1, Eigen::VectorXd::Zero(num_sur), sur_identity));
  Eigen::VectorXd nu = design_mat * u_vec + delta_vec;
  Eigen::VectorXd lin_solve = (design_mat * shrink_mat * design_mat.transpose() + sur_identity).llt().solve(
    response_vec - nu
  );
  return u_vec + shrink_mat * design_mat.transpose() * lin_solve;
  // return u_vec + var * shrink_mat * design_mat.transpose() * lin_solve;
}

//' Generating the Coefficient Vector in Horseshoe Gibbs Sampler
//' 
//' In MCMC process of Horseshoe prior, this function generates the coefficients vector.
//' 
//' @param response_vec Response vector for vectorized formulation
//' @param design_mat Design matrix for vectorized formulation
//' @param shrink_mat Diagonal matrix made by global and local sparsity hyperparameters
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd horseshoe_coef_var(Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat, Eigen::MatrixXd shrink_mat) {
  int dim = design_mat.cols();
  Eigen::VectorXd res(dim + 1);
  int sample_size = response_vec.size();
  Eigen::MatrixXd prec_mat = (design_mat.transpose() * design_mat + shrink_mat).llt().solve(
    Eigen::MatrixXd::Identity(dim, dim)
  );
  double scl = response_vec.transpose() * (Eigen::MatrixXd::Identity(sample_size, sample_size) - design_mat * prec_mat * design_mat.transpose()) * response_vec;
  res[0] = 1 / gamma_rand(sample_size / 2, scl / 2);
  res.tail(dim) = vectorize_eigen(
    sim_mgaussian_chol(1, prec_mat * design_mat.transpose() * response_vec, res[0] * prec_mat)
  );
  return res;
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
double horseshoe_var(Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat, Eigen::MatrixXd shrink_mat) {
  int sample_size = response_vec.size();
  double scl = response_vec.transpose() * (Eigen::MatrixXd::Identity(sample_size, sample_size) - design_mat * shrink_mat * design_mat.transpose()) * response_vec;
  scl *= .5;
  return 1 / gamma_rand(sample_size / 2, scl);
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
Eigen::VectorXd horseshoe_local_sparsity(Eigen::VectorXd local_latent, double global_hyperparam, Eigen::VectorXd coef_vec, double prior_var) {
  int dim = coef_vec.size();
  Eigen::VectorXd res(dim);
  for (int i = 0; i < dim; i++) {
    res[i] = sqrt(1 / gamma_rand(1.0, 1 / ( 1 / local_latent[i] + pow(coef_vec[i], 2.0) / (2 * prior_var * pow(global_hyperparam, 2.0))) ));
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
double horseshoe_global_sparsity(double global_latent, Eigen::VectorXd local_hyperparam, Eigen::VectorXd coef_vec, double prior_var) {
  int dim = coef_vec.size();
  double invgam_scl = 1 / global_latent;
  for (int i = 0; i < dim; i++) {
    invgam_scl += pow(coef_vec[i], 2.0) / (2 * prior_var * pow(local_hyperparam[i], 2.0));
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
    res[i] = 1 / gamma_rand(1.0, 1 / (1 + 1 / pow(local_hyperparam[i], 2.0)));
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
  return 1 / gamma_rand(1.0, 1 / (1 + 1 / pow(global_hyperparam, 2.0)));
}

//' Gibbs Sampler for Horseshoe BVAR SUR Parameterization
//' 
//' This function conducts Gibbs sampling for horseshoe prior BVAR(p).
//' 
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param init_priorvar Initial variance constant
//' @param init_local Initial local shrinkage hyperparameters
//' @param init_global Initial global shrinkage hyperparameter
//' @param init_sigma Initial sigma
//' @param display_progress Progress bar
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_sur_horseshoe(int num_iter, int num_burn,
                                  Eigen::MatrixXd x, Eigen::MatrixXd y,
                                  Eigen::VectorXd init_local, double init_global,
                                  double init_sigma,
                                  int blocked_gibbs,
                                  bool fast,
                                  bool display_progress) {
  int dim = y.cols();
  int dim_design = x.cols(); // dim*p(+1)
  int num_design = y.rows(); // n = T - p
  int num_coef = dim * dim_design;
  // if (blocked_gibbs == 2 && fast) {
  //   Rcpp::stop("Invalid option.");
  // }
  // record------------------------------------------------
  Eigen::MatrixXd coef_record(num_iter + 1, num_coef);
  Eigen::MatrixXd local_record(num_iter + 1, num_coef);
  Eigen::VectorXd global_record(num_iter + 1);
  Eigen::VectorXd sig_record(num_iter + 1);
  Eigen::MatrixXd shrink_record(num_iter + 1, num_coef);
  local_record.row(0) = init_local;
  global_record[0] = init_global;
  sig_record[0] = init_sigma;
  shrink_record.row(0) = 1 / (1 + (init_global * init_local).array().square());
  // Some variables----------------------------------------
  Eigen::VectorXd latent_local(num_coef);
  double latent_global = 0.0;
  Eigen::VectorXd block_coef(num_coef + 1);
  Eigen::MatrixXd design_mat = kronecker_eigen(Eigen::MatrixXd::Identity(dim, dim), x);
  Eigen::VectorXd response_vec = vectorize_eigen(y);
  Eigen::MatrixXd lambda_mat = Eigen::MatrixXd::Zero(num_coef, num_coef);
  // Start Gibbs sampling-----------------------------------
  Progress p(num_iter - 1, display_progress);
  for (int i = 1; i < num_iter + 1; i++) {
    if (Progress::check_abort()) {
      return Rcpp::List::create(
        Rcpp::Named("alpha_record") = coef_record,
        Rcpp::Named("lambda_record") = local_record,
        Rcpp::Named("tau_record") = global_record,
        Rcpp::Named("sigma_record") = sig_record,
        Rcpp::Named("kappa_record") = shrink_record
      );
    }
    p.increment();
    // 1. alpha (coefficient)
    lambda_mat = build_shrink_mat(global_record[i - 1], local_record.row(i - 1));
    switch (blocked_gibbs) {
    case 1:
      // alpha and sigma each
      if (fast) {
        coef_record.row(i) = horseshoe_fast_coef(
          response_vec / sqrt(sig_record[i - 1]),
          design_mat / sqrt(sig_record[i - 1]),
          sig_record[i - 1] * lambda_mat
        );
      } else {
        coef_record.row(i) = horseshoe_coef(response_vec, design_mat, sig_record[i - 1], lambda_mat);
      }
      sig_record[i] = horseshoe_var(response_vec, design_mat, lambda_mat);
    case 2:
      // blocked gibbs
      block_coef = horseshoe_coef_var(response_vec, design_mat, lambda_mat);
      coef_record.row(i) = block_coef.tail(num_coef);
      sig_record[i] = block_coef[0];
    }
    // 3. nuj (local latent)
    latent_local = horseshoe_latent_local(local_record.row(i - 1));
    // 4. xi (global latent)
    latent_global = horseshoe_latent_global(global_record[i - 1]);
    // 5. lambdaj (local shrinkage)
    local_record.row(i) = horseshoe_local_sparsity(latent_local, global_record[i - 1], coef_record.row(i), block_coef[0]);
    // 6. tau (global shrinkage)
    global_record[i] = horseshoe_global_sparsity(latent_global, local_record.row(i), coef_record.row(i), block_coef[0]);
    // kappa
    shrink_record.row(i) = 1 / (1 + (global_record[i] * local_record.row(i)).array().square());
  }
  return Rcpp::List::create(
    Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("lambda_record") = local_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("tau_record") = global_record.tail(num_iter - num_burn),
    Rcpp::Named("sigma_record") = sig_record.tail(num_iter - num_burn),
    Rcpp::Named("kappa_record") = shrink_record.bottomRows(num_iter - num_burn)
  );
}
