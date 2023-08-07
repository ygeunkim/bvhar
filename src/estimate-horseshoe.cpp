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

//' Generating the Local Sparsity Hyperparameters Vector in Horseshoe Gibbs Sampler
//' 
//' In MCMC process of Horseshoe prior, this function generates the local sparsity hyperparameters vector.
//' 
//' @param local_latent Latent vectors defined for local sparsity vector
//' @param global_hyperparam Global sparsity hyperparameter
//' @param coef Coefficients matrix
//' @param prec Precision matrix of the likelihood
//' @param prior_coef_mean Prior mean of coefficients
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd horseshoe_local_sparsity(Eigen::VectorXd local_latent, double global_hyperparam,
                                         Eigen::MatrixXd coef, Eigen::MatrixXd prec,
                                         Eigen::MatrixXd prior_coef_mean) {
  int dim_design = coef.rows();
  int dim = coef.cols();
  Eigen::MatrixXd latent_mat = Eigen::MatrixXd::Zero(dim_design, dim_design);
  latent_mat.diagonal() = 1 / local_latent.array();
  Eigen::MatrixXd coef_demean = coef - prior_coef_mean;
  Eigen::MatrixXd invgam_scl = coef_demean * prec * coef_demean.transpose() / (2 * pow(global_hyperparam, 2.0)) + latent_mat;
  Eigen::VectorXd res(dim_design);
  for (int i = 0; i < dim_design; i++) {
    res[i] = sqrt(1 / gamma_rand((dim + 1) / 2, 1 / invgam_scl(i, i)));
  }
  return res;
}

//' Generating the Global Sparsity Hyperparameter in Horseshoe Gibbs Sampler
//' 
//' In MCMC process of Horseshoe prior, this function generates the global sparsity hyperparameter.
//' 
//' @param global_latent Latent variable defined for global sparsity hyperparameter
//' @param local_hyperparam Local sparsity hyperparameters vector
//' @param coef Coefficients matrix
//' @param prec Precision matrix of the likelihood
//' @noRd
// [[Rcpp::export]]
double horseshoe_global_sparsity(double global_latent, Eigen::VectorXd local_hyperparam,
                                 Eigen::MatrixXd coef, Eigen::MatrixXd prec,
                                 Eigen::MatrixXd prior_coef_mean) {
  int dim_design = coef.rows();
  int dim = coef.cols();
  Eigen::MatrixXd local_mat = Eigen::MatrixXd::Zero(dim_design, dim_design);
  local_mat.diagonal() = 1 / local_hyperparam.array().square();
  Eigen::MatrixXd coef_demean = coef - prior_coef_mean;
  double invgam_scl = (prec * coef_demean.transpose() * local_mat * coef_demean).trace() / 2 + 1 / global_latent;
  return sqrt(1 / gamma_rand((dim_design * dim + 1) / 2, 1 / invgam_scl));
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

//' Generating the Coefficient Matrix in Horseshoe Gibbs Sampler
//' 
//' In MCMC process of Horseshoe prior, this function generates the coefficients matrix.
//' 
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param prior_mean Prior mean of coefficients matrix
//' @param sigma Covariance matrix of the likelihood
//' @param shrink_mat Inverse diagonal matrix made by global and local sparsity hyperparameters
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd horseshoe_coef(Eigen::MatrixXd x, Eigen::MatrixXd y,
                               Eigen::MatrixXd prior_mean,
                               Eigen::MatrixXd sigma, Eigen::MatrixXd shrink_mat) {
  Eigen::MatrixXd prec_mat = (x.transpose() * x + shrink_mat).llt().solve(Eigen::MatrixXd::Identity(shrink_mat.rows(), shrink_mat.cols()));
  return sim_matgaussian(prec_mat * (x.transpose() * y + shrink_mat * prior_mean), prec_mat, sigma);
}

//' Generating the Prior Variance Constant in Horseshoe Gibbs Sampler
//' 
//' In MCMC process of Horseshoe prior, this function generates the prior variance.
//' 
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param coef Coefficients matrix
//' @param shrink_mat Inverse diagonal matrix made by global and local sparsity hyperparameters
//' @param mn_shrink Inverse diagonal matrix for column shrinkage
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd horseshoe_cov_mat(Eigen::MatrixXd x, Eigen::MatrixXd y,
                                  Eigen::MatrixXd coef, Eigen::MatrixXd shrink_mat,
                                  Eigen::MatrixXd prior_coef_mean,
                                  Eigen::MatrixXd prior_var, double prior_shape) {
  Eigen::MatrixXd resid = y - x * coef;
  Eigen::MatrixXd coef_demean = coef - prior_coef_mean;
  return sim_iw(
    resid.transpose() * resid + coef_demean.transpose() * shrink_mat * coef_demean + prior_var,
    y.rows() + coef.rows() + prior_shape
  );
}

Rcpp::List horseshoe_coef_cov(Eigen::MatrixXd x, Eigen::MatrixXd y,
                              Eigen::MatrixXd prior_mean, Eigen::MatrixXd shrink_mat,
                              double prior_shape) {
  Eigen::MatrixXd prec_mat = (x.transpose() * x + shrink_mat).llt().solve(Eigen::MatrixXd::Identity(shrink_mat.rows(), shrink_mat.cols()));
  int num_design = y.rows();
  Eigen::MatrixXd mean_shrink = shrink_mat * prior_mean;
  Eigen::MatrixXd iw_scl = y.transpose() * (
    Eigen::MatrixXd::Identity(num_design, num_design) -
      x * prec_mat * x.transpose()) * y -
      mean_shrink.transpose() * prec_mat * mean_shrink;
  return sim_mniw(1, prec_mat * (x.transpose() * y + mean_shrink), prec_mat, iw_scl, num_design + x.cols() + prior_shape);
}

//' Gibbs Sampler for Horseshoe BVAR Estimator
//' 
//' This function conducts Gibbs sampling for horseshoe prior BVAR(p).
//' 
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param init_local Initial local shrinkage hyperparameters
//' @param init_global Initial global shrinkage hyperparameter
//' @param prior_scale Prior scale of IW
//' @param prior_shape Prior shape of IW
//' @param display_progress Progress bar
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_horseshoe(int num_iter,
                                   int num_burn,
                                   Eigen::MatrixXd x,
                                   Eigen::MatrixXd y,
                                   Eigen::VectorXd init_local,
                                   double init_global,
                                   Eigen::MatrixXd init_prec,
                                   Eigen::MatrixXd prior_mean,
                                   Eigen::MatrixXd prior_scale,
                                   double prior_shape,
                                   int blocked_gibbs,
                                   bool display_progress) {
  int dim = y.cols();
  int dim_design = x.cols(); // dim*p(+1)
  int num_coef = dim * dim_design;
  // record------------------------------------------------
  Eigen::MatrixXd coef_record(num_iter + 1, num_coef);
  Eigen::MatrixXd local_record(num_iter + 1, dim_design);
  Eigen::VectorXd global_record(num_iter + 1);
  Eigen::MatrixXd prec_record = Eigen::MatrixXd::Zero(dim * (num_iter + 1), dim);
  prec_record.block(0, 0, dim, dim) = init_prec;
  local_record.row(0) = init_local;
  global_record[0] = init_global;
  // Some variables----------------------------------------
  Eigen::MatrixXd coef_mat(dim_design, dim);
  Eigen::VectorXd latent_local(dim_design);
  Eigen::MatrixXd sig_mat = init_prec.inverse();
  double latent_global = 0.0;
  // Eigen::MatrixXd prior_sigma = prior_prec.inverse();
  Eigen::MatrixXd lambda_mat = Eigen::MatrixXd::Zero(dim_design, dim_design);
  Rcpp::List block_coef = Rcpp::List::create(
    Rcpp::Named("mn") = Eigen::MatrixXd::Zero(dim_design, dim),
    Rcpp::Named("iw") = Eigen::MatrixXd::Zero(dim, dim)
  );
  // Start Gibbs sampling-----------------------------------
  Progress p(num_iter, display_progress);
  for (int i = 1; i < num_iter + 1; i++) {
    if (Progress::check_abort()) {
      return Rcpp::List::create(
        Rcpp::Named("alpha_record") = coef_record,
        Rcpp::Named("lambda_record") = local_record,
        Rcpp::Named("tau_record") = global_record,
        Rcpp::Named("psi_record") = prec_record
      );
    }
    p.increment();
    // 1. alpha (coefficient) and 2. sigma (variance)
    lambda_mat = build_shrink_mat(global_record[i - 1], local_record.row(i - 1));
    switch (blocked_gibbs) {
    case 1:
      // alpha and sigma each
      coef_mat = horseshoe_coef(x, y, prior_mean, sig_mat, lambda_mat);
      coef_record.row(i) = vectorize_eigen(coef_mat);
      sig_mat = horseshoe_cov_mat(x, y, coef_mat, lambda_mat, prior_mean, prior_scale, prior_shape);
    case 2:
      // blocked gibbs
      block_coef = horseshoe_coef_cov(x, y, prior_mean, lambda_mat, prior_shape);
      coef_mat = block_coef["mn"];
      coef_record.row(i) = vectorize_eigen(coef_mat);
      sig_mat = block_coef["iw"];
    }
    prec_record.block(i * dim, 0, dim, dim) = sig_mat.inverse();
    // 3. nuj (local latent)
    latent_local = horseshoe_latent_local(local_record.row(i - 1));
    // 4. xi (global latent)
    latent_global = horseshoe_latent_global(global_record[i - 1]);
    // 5. lambdaj (local shrinkage)
    local_record.row(i) = horseshoe_local_sparsity(
      latent_local,
      global_record[i - 1],
      coef_mat,
      prec_record.block(i * dim, 0, dim, dim),
      prior_mean
    );
    // 6. tau (global shrinkage)
    global_record[i] = horseshoe_global_sparsity(
      latent_global,
      local_record.row(i),
      coef_mat,
      prec_record.block(i * dim, 0, dim, dim),
      prior_mean
    );
  }
  return Rcpp::List::create(
    Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("lambda_record") = local_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("tau_record") = global_record.tail(num_iter - num_burn),
    Rcpp::Named("psi_record") = prec_record.bottomRows(dim * (num_iter - num_burn))
  );
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
Eigen::VectorXd hs_coef(Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat, double var, Eigen::MatrixXd shrink_mat) {
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
//' @noRD
// [[Rcpp::export]]
Eigen::VectorXd hs_fast_coef(Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat, Eigen::MatrixXd shrink_mat) {
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
Eigen::VectorXd hs_coef_var(Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat, Eigen::MatrixXd shrink_mat) {
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
double hs_var(Eigen::VectorXd response_vec, Eigen::MatrixXd design_mat, Eigen::MatrixXd shrink_mat) {
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
Eigen::VectorXd hs_local_sparsity(Eigen::VectorXd local_latent, double global_hyperparam, Eigen::VectorXd coef_vec, double prior_var) {
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
double hs_global_sparsity(double global_latent, Eigen::VectorXd local_hyperparam, Eigen::VectorXd coef_vec, double prior_var) {
  int dim = coef_vec.size();
  double invgam_scl = 1 / global_latent;
  for (int i = 0; i < dim; i++) {
    invgam_scl += pow(coef_vec[i], 2.0) / (2 * prior_var * pow(local_hyperparam[i], 2.0));
  }
  return sqrt(1 / gamma_rand((dim + 1) / 2, 1 / invgam_scl));
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
  
  // if (blocked_gibbs == 2 && fast == true) {
  //   Rcpp::stop("Invalid option.");
  // }
  
  // record------------------------------------------------
  Eigen::MatrixXd coef_record(num_iter + 1, num_coef);
  Eigen::MatrixXd local_record(num_iter + 1, num_coef);
  Eigen::VectorXd global_record(num_iter + 1);
  Eigen::VectorXd sig_record(num_iter + 1);
  local_record.row(0) = init_local;
  global_record[0] = init_global;
  sig_record[0] = init_sigma;
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
        Rcpp::Named("sigma_record") = sig_record
      );
    }
    p.increment();
    // 1. alpha (coefficient)
    lambda_mat = build_shrink_mat(global_record[i - 1], local_record.row(i - 1));
    switch (blocked_gibbs) {
    case 1:
      // alpha and sigma each
      // coef_record.row(i) = hs_coef(response_vec, design_mat, sig_record[i - 1], lambda_mat);
      // coef_record.row(i) = hs_coef(
      //   response_vec / sqrt(sig_record[i - 1]),
      //   design_mat / sqrt(sig_record[i - 1]),
      //   sig_record[i - 1] * lambda_mat
      // );
      if (fast) {
        coef_record.row(i) = hs_fast_coef(
          response_vec / sqrt(sig_record[i - 1]),
          design_mat / sqrt(sig_record[i - 1]),
          sig_record[i - 1] * lambda_mat
        );
      } else {
        coef_record.row(i) = hs_coef(response_vec, design_mat, sig_record[i - 1], lambda_mat);
      }
      sig_record[i] = hs_var(response_vec, design_mat, lambda_mat);
    case 2:
      // blocked gibbs
      block_coef = hs_coef_var(response_vec, design_mat, lambda_mat);
      coef_record.row(i) = block_coef.tail(num_coef);
      sig_record[i] = block_coef[0];
    }
    // block_coef = hs_coef_var(response_vec, design_mat, lambda_mat);
    // coef_record.row(i) = block_coef.tail(num_coef);
    // // 2. sigma (variance)
    // sig_record[i - 1] = block_coef[0];
    
    // 3. nuj (local latent)
    latent_local = horseshoe_latent_local(local_record.row(i - 1));
    // 4. xi (global latent)
    latent_global = horseshoe_latent_global(global_record[i - 1]);
    // 5. lambdaj (local shrinkage)
    local_record.row(i) = hs_local_sparsity(latent_local, global_record[i - 1], coef_record.row(i), block_coef[0]);
    // 6. tau (global shrinkage)
    global_record[i] = hs_global_sparsity(latent_global, local_record.row(i), coef_record.row(i), block_coef[0]);
  }
  return Rcpp::List::create(
    Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("lambda_record") = local_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("tau_record") = global_record.tail(num_iter - num_burn),
    Rcpp::Named("sigma_record") = sig_record.tail(num_iter - num_burn)
  );
}
