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

//' Fast Sampling Gaussian Scale Mixture Representation
//' 
//' This function generates full conditional coefficients of horseshoe prior fast.
//' 
//' @param diag_mat Diagonal covariance matrix
//' @param scaled_x Phi
//' @param scaled_y alpha
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd horseshoe_fastcoef(Eigen::MatrixXd diag_mat, Eigen::MatrixXd scaled_x, Eigen::VectorXd scaled_y) {
  int dim = scaled_x.cols();
  int sample_size = scaled_x.rows();
  // Eigen::VectorXd gaussian_u = sim_mgaussian_chol(1, Eigen::VectorXd::Zero(dim), diag_mat);
  Eigen::VectorXd gaussian_u(dim);
  Eigen::VectorXd gaussian_delta(sample_size);
  for (int i = 0; i < dim; i++) {
    gaussian_u[i] = sqrt(diag_mat(i, i)) * norm_rand();
  }
  Rcpp::Rcout << "u sampled" << std::endl;
  for (int i = 0; i < sample_size; i++) {
    gaussian_delta[i] = norm_rand();
  }
  Rcpp::Rcout << "delta sampled" << std::endl;
  Eigen::LLT<Eigen::MatrixXd> lltOflin(scaled_x * diag_mat * scaled_x.transpose() + Eigen::MatrixXd::Identity(sample_size, sample_size));
  Rcpp::Rcout << "Cholesky decomposition done" << std::endl;
  Eigen::VectorXd lin_solve = lltOflin.solve(scaled_y - scaled_x * gaussian_u - gaussian_delta);
  Rcpp::Rcout << "Linear system solved" << std::endl;
  return gaussian_u + diag_mat * scaled_x.transpose() * lin_solve;
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
Eigen::MatrixXd horseshoe_coef(Eigen::MatrixXd x,
                               Eigen::MatrixXd y,
                               Eigen::MatrixXd sigma,
                               Eigen::MatrixXd shrink_mat) {
  Eigen::MatrixXd prec_mat = (x.transpose() * x + shrink_mat.inverse()).inverse();
  return sim_matgaussian(prec_mat * x.transpose() * y, prec_mat, sigma);
}

// Eigen::VectorXd horseshoe_coef(Eigen::VectorXd response_vec,
//                                Eigen::MatrixXd design_mat,
//                                double prior_var,
//                                Eigen::MatrixXd shrink_mat) {
//   Eigen::MatrixXd unscaled_var = design_mat.transpose() * design_mat + shrink_mat.inverse();
//   return vectorize_eigen(
//     sim_mgaussian_chol(1, unscaled_var.inverse() * design_mat.transpose() * response_vec, prior_var * unscaled_var)
//   );
//   // return horseshoe_fastcoef(prior_var * shrink_mat, design_mat / sqrt(prior_var), response_vec / sqrt(prior_var));
// }


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
                                         Eigen::MatrixXd coef,
                                         Eigen::MatrixXd sigma) {
  int dim_design = coef.rows();
  Eigen::MatrixXd latent_mat = Eigen::MatrixXd::Zero(dim_design, dim_design);
  latent_mat.diagonal() = local_latent;
  Eigen::MatrixXd invgam_scl = coef.transpose() * sigma.inverse() * coef / (2 * pow(global_hyperparam, 2.0)) + local_latent.inverse();
  Eigen::VectorXd res(dim_design);
  for (int i = 0; i < dim_design; i++) {
    res[i] = sqrt(1 / gamma_rand(1.0, invgam_scl(i, i)));
  }
  return res;
}

// Eigen::VectorXd horseshoe_local_sparsity(Eigen::VectorXd local_latent,
//                                          double global_hyperparam,
//                                          Eigen::VectorXd coef_vec,
//                                          double prior_var) {
//   int dim = coef_vec.size();
//   Eigen::VectorXd res(dim);
//   for (int i = 0; i < dim; i++) {
//     res[i] = sqrt(1 /
//       gamma_rand(
//         1.0,
//         1 / ( 1 / local_latent[i] + pow(coef_vec[i], 2.0) / (2 * prior_var * pow(global_hyperparam, 2.0)) )
//       ));
//   }
//   return res;
// }


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
                                 Eigen::MatrixXd coef,
                                 Eigen::MatrixXd sigma) {
  int dim_design = coef.rows();
  int dim = coef.cols();
  Eigen::MatrixXd local_mat = Eigen::MatrixXd::Zero(dim_design, dim_design);
  local_mat.diagonal() = 1 / local_hyperparam.array();
  double invgam_scl = (sigma.inverse() * coef.transpose() * local_mat * coef).trace() / 2 + 1 / global_latent;
  return sqrt(1 / gamma_rand((dim_design * dim + 1) / 2, 1 / invgam_scl));
}

// double horseshoe_global_sparsity(double global_latent,
//                                  Eigen::VectorXd local_hyperparam,
//                                  Eigen::VectorXd coef_vec,
//                                  double prior_var) {
//   int dim = coef_vec.size();
//   double invgam_scl = 1 / global_latent;
//   for (int i = 0; i < dim; i++) {
//     invgam_scl += pow(coef_vec[i], 2.0) / (2 * prior_var * pow(local_hyperparam[i], 2.0));
//   }
//   return sqrt(1 / gamma_rand((dim + 1) / 2, 1 / invgam_scl));
// }


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
Eigen::MatrixXd horseshoe_prior_var(Eigen::MatrixXd x,
                                    Eigen::MatrixXd y,
                                    Eigen::MatrixXd coef,
                                    Eigen::MatrixXd shrink_mat) {
  Eigen::MatrixXd resid = y - x * coef;
  return sim_iw(resid.transpose() * resid + coef.transpose() * shrink_mat.inverse() * coef, y.rows() + coef.rows());
}

// double horseshoe_prior_var(Eigen::VectorXd response_vec,
//                            Eigen::MatrixXd design_mat,
//                            Eigen::VectorXd coef_vec,
//                            Eigen::MatrixXd shrink_mat) {
//   Eigen::VectorXd resid = response_vec - design_mat * coef_vec;
//   return 1 /
//     gamma_rand(
//       (response_vec.size() + coef_vec.size()) / 2,
//       1 / (resid.squaredNorm() / 2 + coef_vec.transpose() * shrink_mat.inverse() * coef_vec)
//     );
// }



//' Gibbs Sampler for Horseshoe BVAR Estimator
//' 
//' This function conducts Gibbs sampling for horseshoe prior BVAR(p).
//' 
//' @param num_iter Number of iteration for MCMC
//' @param num_warm Number of warm-up (burn-in) for MCMC
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param init_local Initial local shrinkage hyperparameters
//' @param init_global Initial global shrinkage hyperparameter
//' @param init_priorvar Initial variance constant
//' @param chain The number of MCMC chains.
//' @param display_progress Progress bar
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_horseshoe_niw(int num_iter,
                                  int num_warm,
                                  Eigen::MatrixXd x,
                                  Eigen::MatrixXd y,
                                  Eigen::VectorXd init_local,
                                  Eigen::VectorXd init_global,
                                  Eigen::MatrixXd init_priorvar,
                                  int chain,
                                  bool display_progress) {
  int dim = y.cols();
  int dim_design = x.cols(); // dim*p(+1)
  int num_coef = dim * dim_design;
  // record------------------------------------------------
  Eigen::MatrixXd coef_record = Eigen::MatrixXd::Zero(num_iter, num_coef * chain);
  Eigen::MatrixXd local_record = Eigen::MatrixXd::Zero(num_iter, num_coef * chain);
  Eigen::MatrixXd global_record = Eigen::MatrixXd::Zero(num_iter, chain);
  Eigen::MatrixXd latent_local_record = Eigen::MatrixXd::Zero(num_iter, num_coef * chain);
  Eigen::MatrixXd latent_global_record = Eigen::MatrixXd::Zero(num_iter, chain);
  Eigen::MatrixXd sig_record = Eigen::MatrixXd::Zero(dim * num_iter, dim * chain);
  local_record.row(0) = init_local;
  global_record.row(0) = init_global;
  // coef_record.row(0) = vectorize_eigen(init_coef);
  // sig_record.row(0) = init_priorvar;
  sig_record.block(0, 0, dim, dim) = init_priorvar;
  // Some variables----------------------------------------
  Eigen::MatrixXd coef_mat = Eigen::MatrixXd::Zero(dim_design, dim);
  Eigen::MatrixXd sigma = init_priorvar;
  Eigen::MatrixXd lambda_mat = Eigen::MatrixXd::Zero(num_coef, num_coef);
  // Start Gibbs sampling-----------------------------------
  Progress p(chain * (num_iter - 1), display_progress);
#ifdef _OPENMP
  Rcpp::Rcout << "Parallel chains" << std::endl;
#pragma                  \
  omp                    \
    parallel             \
    for                  \
      num_threads(chain) \
      shared(y, x, num_coef)
  for (int b = 0; b < chain; b++) {
    for (int i = 1; i < num_iter; i++) {
      if (Progress::check_abort()) {
        return Rcpp::List::create(
          Rcpp::Named("alpha_record") = coef_record,
          Rcpp::Named("lambda_record") = local_record,
          Rcpp::Named("tau_record") = global_record,
          Rcpp::Named("sigma_record") = sig_record,
          Rcpp::Named("chain") = chain
        );
      }
      p.increment();
      // 1. alpha (coefficient)
      lambda_mat = build_shrink_mat(global_record(i - 1, chain), local_record.block(i - 1, b * num_coef, 1, num_coef));
      coef_mat = horseshoe_coef(x, y, sig_record.block((i - 1) * dim, b * dim, dim, dim), lambda_mat);
      coef_record.block(i, b * num_coef, 1, num_coef) = vectorize_eigen(coef_mat);
      // 2. sigma (variance)
      sig_record.block(i * dim, b * dim, dim, dim) = horseshoe_prior_var(x, y, coef_mat, lambda_mat);
      // 3. nuj (local latent)
      latent_local_record.block(i, b * num_coef, 1, num_coef) = horseshoe_latent_local(local_record.block(i - 1, b * num_coef, 1, num_coef));
      // 4. xi (global latent)
      latent_global_record(i, chain) = horseshoe_latent_global(global_record(i - 1, chain));
      // 5. lambdaj (local shrinkage)
      local_record.block(i, b * num_coef, 1, num_coef) = horseshoe_local_sparsity(
        latent_local_record.block(i, b * num_coef, 1, num_coef), 
        global_record(i - 1, chain), 
        coef_mat, 
        sig_record.block(i * dim, b * dim, dim, dim)
      );
      // 6. tau (global shrinkage)
      global_record(i, chain) = horseshoe_global_sparsity(
        latent_global_record(i, chain),
        local_record.block(i, b * num_coef, 1, num_coef),
        coef_mat,
        sig_record.block(i * dim, b * dim, dim, dim)
      );
    }
  }
#else
  for (int i = 1; i < num_iter; i++) {
    if (Progress::check_abort()) {
      return Rcpp::List::create(
        Rcpp::Named("alpha_record") = coef_record,
        Rcpp::Named("lambda_record") = local_record,
        Rcpp::Named("tau_record") = global_record,
        Rcpp::Named("sigma_record") = sig_record,
        Rcpp::Named("chain") = chain
      );
    }
    p.increment();
    // 1. alpha (coefficient)
    lambda_mat = build_shrink_mat(global_record(i - 1, 0), local_record.row(i - 1));
    coef_mat = horseshoe_coef(x, y, sig_record.block((i - 1) * dim, 0, dim, dim), lambda_mat);
    coef_record.row(i) = vectorize_eigen(coef_mat);
    // 2. sigma (variance)
    sig_record.block(i * dim, 0, dim, dim) = horseshoe_prior_var(x, y, coef_mat, lambda_mat);
    // 3. nuj (local latent)
    latent_local_record.row(i) = horseshoe_latent_local(local_record.row(i - 1));
    // 4. xi (global latent)
    latent_global_record(i, 0) = horseshoe_latent_global(global_record(i - 1, 0));
    // 5. lambdaj (local shrinkage)
    local_record.row(i) = horseshoe_local_sparsity(
      latent_local_record.row(i), 
      global_record(i - 1, 0), 
      coef_mat, 
      sig_record.block(i * dim, 0, dim, dim)
    );
    // 6. tau (global shrinkage)
    global_record(i, 0) = horseshoe_global_sparsity(
      latent_global_record(i, 0),
      local_record.row(i),
      coef_mat,
      sig_record.block(i * dim, 0, dim, dim)
    );
  }
#endif
  return Rcpp::List::create(
    Rcpp::Named("alpha_record") = coef_record,
    Rcpp::Named("lambda_record") = local_record,
    Rcpp::Named("tau_record") = global_record,
    Rcpp::Named("sigma_record") = sig_record,
    Rcpp::Named("chain") = chain
  );
}

