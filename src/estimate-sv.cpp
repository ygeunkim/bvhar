#include <RcppEigen.h>
#include "bvhardraw.h"
#include <progress.hpp>
#include <progress_bar.hpp>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppProgress)]]

//' VAR-SV by Gibbs Sampler
//' 
//' This function generates parameters \eqn{\beta, a, \sigma_{h,i}^2, h_{0,i}} and log-volatilities \eqn{h_{i,1}, \ldots, h_{i, n}}.
//' 
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param prior_coef_mean Prior mean matrix of coefficient in Minnesota belief
//' @param prior_coef_prec Prior precision matrix of coefficient in Minnesota belief
//' @param prec_diag Diagonal matrix of sigma of innovation to build Minnesota moment
//' @param display_progress Progress bar
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_var_sv(int num_iter, int num_burn,
                           Eigen::MatrixXd x, Eigen::MatrixXd y,
                           Eigen::MatrixXd prior_coef_mean,
                           Eigen::MatrixXd prior_coef_prec,
                           Eigen::MatrixXd prec_diag,
                           int prior_type,
                           Eigen::VectorXd init_local, double init_global,
                           Eigen::VectorXd coef_spike,
                           Eigen::VectorXd coef_slab,
                           Eigen::VectorXd coef_slab_weight,
                           Eigen::VectorXd intercept_mean,
                           double intercept_sd,
                           bool include_mean,
                           bool display_progress, int nthreads) {
  int dim = y.cols(); // k
  int dim_design = x.cols(); // kp(+1)
  int num_design = y.rows(); // n = T - p
  int num_lowerchol = dim * (dim - 1) / 2;
  int num_coef = dim * dim_design;
  int num_alpha = num_coef - dim;
  if (!include_mean) {
    num_alpha += dim; // always dim^2 p
  }
  // SUR---------------------------------------------------
  Eigen::VectorXd response_vec = vectorize_eigen(y);
  Eigen::MatrixXd design_mat = kronecker_eigen(Eigen::MatrixXd::Identity(dim, dim), x);
  // Default setting---------------------------------------
  Eigen::VectorXd prior_alpha_mean(num_coef); // prior mean vector of alpha
  Eigen::MatrixXd prior_alpha_prec = Eigen::MatrixXd::Zero(num_coef, num_coef); // prior precision of alpha
  switch(prior_type) {
    case 1:
      prior_alpha_mean = vectorize_eigen(prior_coef_mean);
      prior_alpha_prec = kronecker_eigen(prec_diag, prior_coef_prec);
      break;
    case 2:
      if (include_mean) {
        for (int j = 0; j < dim; j++) {
          prior_alpha_mean.segment(j * dim_design, num_alpha / dim) = Eigen::VectorXd::Zero(num_alpha / dim);
          prior_alpha_mean[j * dim_design + num_alpha / dim] = intercept_mean[j];
        }
      } else {
        prior_alpha_mean = Eigen::VectorXd::Zero(num_alpha);
      }
      break;
    case 3:
      prior_alpha_mean = Eigen::VectorXd::Zero(num_coef);
      break;
  }
  Eigen::VectorXd prior_chol_mean = Eigen::VectorXd::Zero(num_lowerchol); // a0 = 0
  Eigen::MatrixXd prior_chol_prec = Eigen::MatrixXd::Identity(num_lowerchol, num_lowerchol); // Va = I
  Eigen::VectorXd prior_sig_shp = 3 * Eigen::VectorXd::Ones(dim); // nu_h = 3 * 1_k
  Eigen::VectorXd prior_sig_scl = .01 * Eigen::VectorXd::Ones(dim); // S_h = .1^2 * 1_k
  Eigen::VectorXd prior_init_mean = Eigen::VectorXd::Ones(dim); // b0 = 1
  Eigen::MatrixXd prior_init_prec = Eigen::MatrixXd::Identity(dim, dim) / 10; // Inverse of B0 = .1 * I
  Eigen::MatrixXd coef_ols = (x.transpose() * x).llt().solve(x.transpose() * y); // LSE
  // record------------------------------------------------
  Eigen::MatrixXd coef_record = Eigen::MatrixXd::Zero(num_iter + 1, num_coef); // alpha in VAR
  Eigen::MatrixXd chol_lower_record = Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol); // a = a21, a31, ..., ak1, ..., ak(k-1)
  Eigen::MatrixXd lvol_sig_record = Eigen::MatrixXd::Zero(num_iter + 1, dim); // sigma_h^2 = (sigma_(h1i)^2, ..., sigma_(hki)^2)
  Eigen::MatrixXd lvol_init_record = Eigen::MatrixXd::Zero(num_iter + 1, dim); // h0 = h10, ..., hk0
  Eigen::MatrixXd lvol_record = Eigen::MatrixXd::Zero(num_design * (num_iter + 1), dim); // time-varying h = (h_1, ..., h_k) with h_j = (h_j1, ..., h_jn): h_ij in each dim-block
  // Eigen::MatrixXd cov_record = Eigen::MatrixXd::Zero(dim * (num_iter + 1), dim * num_design); // sigma_t, t = 1, ..., n
  // Eigen::MatrixXd lvol_record = Eigen::MatrixXd::Zero(num_iter + 1, num_design * dim); // time-varying h = (h_1, ..., h_k) with h_j = (h_j1, ..., h_jn): stack h_j row-wise
  // SSVS--------------
  Eigen::MatrixXd coef_dummy_record(num_iter + 1, num_alpha);
  // HS----------------
  Eigen::MatrixXd local_record(num_iter + 1, num_coef);
  Eigen::VectorXd global_record(num_iter + 1);
  Eigen::MatrixXd shrink_record(num_iter + 1, num_coef);
  // Initialize--------------------------------------------
  Eigen::VectorXd coefvec_ols = vectorize_eigen(coef_ols);
  coef_record.row(0) = coefvec_ols;
  chol_lower_record.row(0) = Eigen::VectorXd::Zero(num_lowerchol); // initialize a as 0
  lvol_init_record.row(0) = (y - x * coef_ols).transpose().array().square().rowwise().mean().log(); // initialize h0 as mean of log((y - x alpha)^T (y - x alpha))
  lvol_record.block(0, 0, num_design, dim) = lvol_init_record.row(0).replicate(num_design, 1);
  // lvol_record.row(0) = lvol_init_record.row(0).replicate(1, num_design);
  lvol_sig_record.row(0) = .1 * Eigen::VectorXd::Ones(dim);
  // SSVS--------------
  coef_dummy_record.row(0) = Eigen::VectorXd::Ones(num_alpha);
  // HS----------------
  local_record.row(0) = init_local;
  global_record[0] = init_global;
  shrink_record.row(0) = 1 / (1 + (init_global * init_local).array().square());
  // Some variables----------------------------------------
  Eigen::MatrixXd coef_mat = unvectorize(coef_record.row(0), dim_design, dim);
  Eigen::MatrixXd chol_lower = Eigen::MatrixXd::Zero(dim, dim); // L in Sig_t^(-1) = L D_t^(-1) LT
  Eigen::MatrixXd latent_innov(num_design, dim); // Z0 = Y0 - X0 A = (eps_p+1, eps_p+2, ..., eps_n+p)^T
  Eigen::MatrixXd reginnov_stack = Eigen::MatrixXd::Zero(num_design * dim, num_lowerchol); // stack t = 1, ..., n => e = E a + eta
  Eigen::MatrixXd innov_prec = Eigen::MatrixXd::Zero(num_design * dim, num_design * dim); // D^(-1) = diag(D_1^(-1), ..., D_n^(-1)) with D_t = diag(exp(h_it))
  Eigen::MatrixXd prec_stack = Eigen::MatrixXd::Zero(num_design * dim, num_design * dim); // sigma^(-1) = diag(sigma_1^(-1), ..., sigma_n^(-1)) with sigma_t^(-1) = L^T D_t^(-1) L
  Eigen::MatrixXd ortho_latent(num_design, dim); // orthogonalized Z0
  int reginnov_id = 0;
  // SSVS--------------
  Eigen::VectorXd prior_sd(num_coef);
  Eigen::VectorXd coef_mixture_mat(num_alpha);
  // HS----------------
  Eigen::VectorXd latent_local(num_coef);
  double latent_global = 0.0;
  // Start Gibbs sampling-----------------------------------
  Progress p(num_iter, display_progress);
  for (int i = 1; i < num_iter + 1; i ++) {
    if (Progress::check_abort()) {
      if (prior_type == 2) {
        return Rcpp::List::create(
          Rcpp::Named("alpha_record") = coef_record,
          Rcpp::Named("h_record") = lvol_record,
          Rcpp::Named("a_record") = chol_lower_record,
          Rcpp::Named("h0_record") = lvol_init_record,
          Rcpp::Named("sigh_record") = lvol_sig_record,
          Rcpp::Named("gamma_record") = coef_dummy_record
        );
      } else if (prior_type == 3) {
        return Rcpp::List::create(
          Rcpp::Named("alpha_record") = coef_record,
          Rcpp::Named("h_record") = lvol_record,
          Rcpp::Named("a_record") = chol_lower_record,
          Rcpp::Named("h0_record") = lvol_init_record,
          Rcpp::Named("sigh_record") = lvol_sig_record,
          Rcpp::Named("lambda_record") = local_record,
          Rcpp::Named("tau_record") = global_record,
          Rcpp::Named("kappa_record") = shrink_record
        );
      }
      return Rcpp::List::create(
        Rcpp::Named("alpha_record") = coef_record,
        Rcpp::Named("h_record") = lvol_record,
        Rcpp::Named("a_record") = chol_lower_record,
        Rcpp::Named("h0_record") = lvol_init_record,
        Rcpp::Named("sigh_record") = lvol_sig_record
      );
    }
    p.increment();
    // 1. alpha----------------------------
    chol_lower = build_inv_lower(dim, chol_lower_record.row(i - 1));
    for (int t = 0; t < num_design; t++) {
      innov_prec.block(t * dim, t * dim, dim, dim).diagonal() = (
        -lvol_record.block(num_design * (i - 1), 0, num_design, dim).row(t)
      ).array().exp();
      prec_stack.block(t * dim, t * dim, dim, dim) = chol_lower.transpose() * innov_prec.block(t * dim, t * dim, dim, dim) * chol_lower;
      // cov_record.block(i * dim, t * dim, dim, dim) = prec_stack.block(t * dim, t * dim, dim, dim).inverse();
    }
    switch(prior_type) {
    case 1:
      coef_record.row(i) = varsv_regression(
        design_mat, response_vec,
        prior_alpha_mean,
        prior_alpha_prec,
        prec_stack
      );
      break;
    case 2:
      // SSVS
      coef_mixture_mat = build_ssvs_sd(coef_spike, coef_slab, coef_dummy_record.row(i - 1));
      if (include_mean) {
        for (int j = 0; j < dim; j++) {
          prior_sd.segment(j * dim_design, num_alpha / dim) = coef_mixture_mat.segment(
            j * num_alpha / dim,
            num_alpha / dim
          );
          prior_sd[j * dim_design + num_alpha / dim] = intercept_sd;
        }
      } else {
        prior_sd = coef_mixture_mat;
      }
      prior_alpha_prec.diagonal() = 1 / prior_sd.array();
      coef_record.row(i) = varsv_regression(
        design_mat, response_vec,
        prior_alpha_mean,
        prior_alpha_prec,
        prec_stack
      );
      coef_mat = unvectorize(coef_record.row(i), dim_design, dim);
      coef_dummy_record.row(i) = ssvs_dummy(
        vectorize_eigen(coef_mat.topRows(num_alpha / dim)),
        coef_slab, coef_spike, coef_slab_weight
      );
      break;
    case 3:
      // HS
      prior_alpha_prec = build_shrink_mat(global_record[i - 1], local_record.row(i - 1));
      coef_record.row(i) = varsv_regression(
        design_mat, response_vec,
        prior_alpha_mean,
        prior_alpha_prec,
        prec_stack
      );
      latent_local = horseshoe_latent_local(local_record.row(i - 1));
      latent_global = horseshoe_latent_global(global_record[i - 1]);
      local_record.row(i) = horseshoe_local_sparsity(
        latent_local, global_record[i - 1],
        coef_record.row(i), 1
      );
      global_record[i] = horseshoe_global_sparsity(
        latent_global, local_record.row(i),
        coef_record.row(i), 1
      );
      shrink_record.row(i) = 1 / (1 + (global_record[i] * local_record.row(i)).array().square());
      break;
    }
    // 2. h---------------------------------
    coef_mat = Eigen::Map<Eigen::MatrixXd>(coef_record.row(i).data(), dim_design, dim);
    latent_innov = y - x * coef_mat;
    ortho_latent = latent_innov * chol_lower.transpose(); // L eps_t <=> Z0 U
    ortho_latent = (ortho_latent.array().square() + .0001).array().log(); // adjustment log(e^2 + c) for some c = 10^(-4) against numerical problems
    for (int t = 0; t < dim; t++) {
      lvol_record.col(t).segment(num_design * i, num_design) = varsv_ht(
        lvol_record.col(t).segment(num_design * (i - 1), num_design),
        lvol_init_record(i - 1, t),
        lvol_sig_record(i - 1, t),
        ortho_latent.col(t), nthreads
      );
    }
    // 3. a---------------------------------
#ifdef _OPENMP
#pragma omp parallel for num_threads(1)
    for (int t = 0; t < num_design; t++) {
      for (int j = 1; j < dim; j++) {
        reginnov_stack.block(t * dim, 0, dim, num_lowerchol).row(j).segment(reginnov_id, j) = -latent_innov.row(t).segment(0, j);
        reginnov_id += j;
      }
      reginnov_id = 0;
    }
#else
    for (int t = 0; t < num_design; t++) {
      for (int j = 1; j < dim; j++) {
        reginnov_stack.block(t * dim, 0, dim, num_lowerchol).row(j).segment(reginnov_id, j) = -latent_innov.row(t).segment(0, j);
        // reginnov_design.row(j).segment(reginnov_id, j) = -latent_innov.row(t).segment(0, j);
        // reginnov_design.block(j, reginnov_id, 1, j) = -latent_innov.row(t).segment(0, j);
        reginnov_id += j;
      }
      reginnov_id = 0;
    }
#endif
    chol_lower_record.row(i) = varsv_regression(
      reginnov_stack,
      vectorize_eigen(latent_innov),
      prior_chol_mean,
      prior_chol_prec,
      innov_prec
    );
    // 4. sigma_h---------------------------
    lvol_sig_record.row(i) = varsv_sigh(
      prior_sig_shp,
      prior_sig_scl,
      lvol_init_record.row(i - 1),
      lvol_record.block(num_design * i, 0, num_design, dim)
    );
    // 5. h0--------------------------------
    lvol_init_record.row(i) = varsv_h0(
      prior_init_mean,
      prior_init_prec,
      lvol_init_record.row(i - 1),
      lvol_record.block(num_design * i, 0, num_design, dim).row(0),
      lvol_sig_record.row(i)
    );
  }
  if (prior_type == 2) {
    return Rcpp::List::create(
      Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("h_record") = lvol_record.bottomRows(num_design * (num_iter - num_burn)),
      Rcpp::Named("a_record") = chol_lower_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("h0_record") = lvol_init_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("sigh_record") = lvol_sig_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("gamma_record") = coef_dummy_record.bottomRows(num_iter - num_burn)
    );
  } else if (prior_type == 3) {
    return Rcpp::List::create(
      Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("h_record") = lvol_record.bottomRows(num_design * (num_iter - num_burn)),
      Rcpp::Named("a_record") = chol_lower_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("h0_record") = lvol_init_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("sigh_record") = lvol_sig_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("lambda_record") = local_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("tau_record") = global_record.tail(num_iter - num_burn),
      Rcpp::Named("kappa_record") = shrink_record.bottomRows(num_iter - num_burn)
    );
  }
  return Rcpp::List::create(
    Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("h_record") = lvol_record.bottomRows(num_design * (num_iter - num_burn)),
    Rcpp::Named("a_record") = chol_lower_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("h0_record") = lvol_init_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("sigh_record") = lvol_sig_record.bottomRows(num_iter - num_burn)
  );
}
