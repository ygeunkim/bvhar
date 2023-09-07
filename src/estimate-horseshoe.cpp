#include <RcppEigen.h>
#include "bvhardraw.h"
#include <progress.hpp>
#include <progress_bar.hpp>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppProgress)]]

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
//' @param mn_id Index for Minnesota lag
//' @param display_progress Progress bar
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_sur_horseshoe(int num_iter, int num_burn,
                                  Eigen::MatrixXd x, Eigen::MatrixXd y,
                                  Eigen::VectorXd init_local,
                                  Eigen::VectorXd init_global,
                                  double init_sigma,
                                  Eigen::VectorXi grp_id,
                                  Eigen::MatrixXd grp_mat,
                                  Eigen::VectorXd mn_id,
                                  int blocked_gibbs,
                                  bool fast,
                                  bool display_progress) {
  int dim = y.cols();
  int dim_design = x.cols(); // dim*p(+1)
  int num_design = y.rows(); // n = T - p
  int num_coef = dim * dim_design;
  int mn_size = mn_id.size(); // If vanilla Horseshoe, same as num_coef
  int ord = (int)dim_design / dim; // p in VAR and 3 in VHAR
  // int glob_len = 1;
  // if (mn_size != num_coef) {
  //   glob_len = mn_size; // glob_len = mn_size = dim
  // }
  int glob_len = init_global.size(); // p + 1 in VAR and 6 in VHAR
  int num_grp = grp_id.size();
  // record------------------------------------------------
  Eigen::MatrixXd coef_record(num_iter + 1, num_coef);
  Eigen::MatrixXd local_record(num_iter + 1, num_coef);
  Eigen::MatrixXd global_record(num_iter + 1, num_grp); // tau1: own-lag, tau2: cross-lag, ...
  Eigen::VectorXd sig_record(num_iter + 1);
  Eigen::MatrixXd shrink_record(num_iter + 1, num_coef);
  local_record.row(0) = init_local;
  global_record.row(0) = init_global;
  sig_record[0] = init_sigma;
  // Some variables----------------------------------------
  Eigen::VectorXd latent_local(num_coef);
  Eigen::VectorXd latent_global(num_grp);
  Eigen::VectorXd global_shrinkage(num_coef);
  Eigen::MatrixXd global_shrinkage_mat = Eigen::MatrixXd::Zero(dim_design, dim);

  Eigen::VectorXd grp_vec = vectorize_eigen(grp_mat);
  Eigen::VectorXi global_id(dim * dim_design); // 0 1 for the group

  // Eigen::VectorXd mn_coef(dim * dim);
  // Eigen::VectorXd mn_local(dim * dim);
  // Eigen::VectorXd mn_latent_global(mn_size); // Latent to global shrinkage in own-lags -> should be fixed
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
    for (int j = 0; j < num_grp; j++) {
      global_shrinkage_mat = (
        grp_mat.array() == grp_id[j]
      ).select(
        global_record.row(i - 1).segment(j, 1).replicate(dim_design, dim),
        global_shrinkage_mat
      );
    }
    global_shrinkage = vectorize_eigen(global_shrinkage_mat);
    lambda_mat = build_shrink_mat(global_shrinkage, init_local);
    shrink_record.row(i - 1) = (Eigen::MatrixXd::Identity(num_coef, num_coef) + lambda_mat).inverse().diagonal();
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
    latent_local = horseshoe_latent(local_record.row(i - 1));
    // 4. xi (global latent)
    latent_global = horseshoe_latent(global_record.row(i - 1));
    // 5. lambdaj (local shrinkage)
    init_local = horseshoe_local_sparsity(
      latent_local,
      global_shrinkage,
      coef_record.row(i),
      sig_record[i]
    );
    local_record.row(i) = init_local;
    // 6. tau (global shrinkage)
    for (int j = 0; j < num_grp; j++) {
      global_id = (grp_vec.array() == grp_id[j]).cast<int>();
      Eigen::VectorXd coef_vec = Eigen::VectorXd::Ones(global_id.sum());
      Eigen::VectorXd local_vec = Eigen::VectorXd::Ones(global_id.sum());
      // for (int k = 0; k < global_id.sum(); k++) {
      //   if (global_id[k] == 1) {
      //     coef_vec[k] = coef_record(i, k);
      //     local_vec[k] = local_record(i, k);
      //   }
      // }
      global_record(i, j) = horseshoe_global_sparsity(
        latent_global[j],
        coef_vec,
        local_vec,
        sig_record[i]
      );
    }

    // for (int j = 0; j < mn_size; j++) {
    //   mn_coef[j] = coef_record(i, mn_id[j]);
    //   mn_local[j] = local_record(i, mn_id[j]);
    // }
    // mn_latent_global = vectorize_eigen(latent_global.replicate(1, mn_size / glob_len));

    // global_record.row(i) = horseshoe_global_sparsity(
    //   mn_latent_global,
    //   mn_local,
    //   mn_coef,
    //   sig_record[i]
    // );

    return Rcpp::List::create(
      Rcpp::Named("test1") = latent_global,
      Rcpp::Named("test2") = global_record.topRows(2)
    );

  }
  shrink_record.row(num_iter) = (Eigen::MatrixXd::Identity(num_coef, num_coef) + lambda_mat).inverse().diagonal();
  return Rcpp::List::create(
    Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("lambda_record") = local_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("tau_record") = global_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("sigma_record") = sig_record.tail(num_iter - num_burn),
    Rcpp::Named("kappa_record") = shrink_record.bottomRows(num_iter - num_burn)
  );
}
