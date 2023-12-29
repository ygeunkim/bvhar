#include <RcppEigen.h>
#include "bvhardraw.h"
#include "bvharprogress.h"
#include "bvharinterrupt.h"

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
//' @param init_local Initial local shrinkage of Horseshoe
//' @param init_global Initial global shrinkage of Horseshoe
//' @param init_contem_local Initial local shrinkage for Cholesky factor in Horseshoe
//' @param init_contem_global Initial global shrinkage for Cholesky factor in Horseshoe
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param prior_sig_shp Inverse-Gamma prior shape of state variance
//' @param prior_sig_scl Inverse-Gamma prior scale of state variance
//' @param prior_init_mean Noraml prior mean of initial state
//' @param prior_init_prec Normal prior precision of initial state
//' @param coef_spike SD of spike normal
//' @param coef_slab_weight SD of slab normal
//' @param chol_spike Standard deviance for cholesky factor Spike normal distribution
//' @param chol_slab Standard deviance for cholesky factor Slab normal distribution
//' @param chol_slab_weight Cholesky factor sparsity proportion
//' @param coef_s1 First shape of prior beta distribution of coefficients slab weight
//' @param coef_s2 Second shape of prior beta distribution of coefficients slab weight
//' @param mean_non Prior mean of unrestricted coefficients
//' @param sd_non SD for unrestricted coefficients
//' @param include_mean Constant term
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
                           Eigen::VectorXd init_local,
                           Eigen::VectorXd init_global,
                           Eigen::VectorXd init_contem_local,
                           Eigen::VectorXd init_contem_global,
                           Eigen::VectorXi grp_id,
                           Eigen::MatrixXd grp_mat,
													 Eigen::VectorXd prior_sig_shp,
													 Eigen::VectorXd prior_sig_scl,
													 Eigen::VectorXd prior_init_mean,
													 Eigen::MatrixXd prior_init_prec,
                           Eigen::VectorXd coef_spike,
                           Eigen::VectorXd coef_slab,
                           Eigen::VectorXd coef_slab_weight,
                           Eigen::VectorXd chol_spike,
                           Eigen::VectorXd chol_slab,
                           Eigen::VectorXd chol_slab_weight,
                           double coef_s1, double coef_s2,
                           double chol_s1, double chol_s2,
                           Eigen::VectorXd mean_non,
                           double sd_non,
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
  int num_grp = grp_id.size();
#ifdef _OPENMP
  Eigen::setNbThreads(nthreads);
  Eigen::initParallel();
#endif
  // Default setting---------------------------------------
  Eigen::VectorXd prior_alpha_mean(num_coef); // prior mean vector of alpha
  Eigen::MatrixXd prior_alpha_prec = Eigen::MatrixXd::Zero(num_coef, num_coef); // prior precision of alpha
  Eigen::VectorXd prior_chol_mean = Eigen::VectorXd::Zero(num_lowerchol); // prior mean vector of a = 0
  Eigen::MatrixXd prior_chol_prec = Eigen::MatrixXd::Identity(num_lowerchol, num_lowerchol); // prior precision of a = I
  switch(prior_type) {
    case 1:
      prior_alpha_mean = vectorize_eigen(prior_coef_mean);
      prior_alpha_prec = kronecker_eigen(prec_diag, prior_coef_prec);
      break;
    case 2:
      if (include_mean) {
        for (int j = 0; j < dim; j++) {
          prior_alpha_mean.segment(j * dim_design, num_alpha / dim) = Eigen::VectorXd::Zero(num_alpha / dim);
          prior_alpha_mean[j * dim_design + num_alpha / dim] = mean_non[j];
        }
      } else {
        prior_alpha_mean = Eigen::VectorXd::Zero(num_alpha);
      }
      break;
    case 3:
      prior_alpha_mean = Eigen::VectorXd::Zero(num_coef);
      break;
  }
  Eigen::MatrixXd coef_mat = (x.transpose() * x).llt().solve(x.transpose() * y); // LSE
  // record------------------------------------------------
  Eigen::MatrixXd coef_record = Eigen::MatrixXd::Zero(num_iter + 1, num_coef); // alpha in VAR
  Eigen::MatrixXd contem_coef_record = Eigen::MatrixXd::Zero(num_iter + 1, num_lowerchol); // a = a21, a31, a32, ..., ak1, ..., ak(k-1)
  Eigen::MatrixXd lvol_sig_record = Eigen::MatrixXd::Zero(num_iter + 1, dim); // sigma_h^2 = (sigma_(h1i)^2, ..., sigma_(hki)^2)
  Eigen::MatrixXd lvol_init_record = Eigen::MatrixXd::Zero(num_iter + 1, dim); // h0 = h10, ..., hk0
  // Eigen::MatrixXd lvol_record = Eigen::MatrixXd::Zero(num_design * (num_iter + 1), dim); // time-varying h = (h_1, ..., h_k) with h_j = (h_j1, ..., h_jn): h_ij in each dim-block
	Eigen::MatrixXd lvol_record = Eigen::MatrixXd::Zero(num_iter + 1, num_design * dim); // time-varying h = (h_1, ..., h_k) with h_j = (h_j1, ..., h_jn), row-binded
  // SSVS--------------
  Eigen::MatrixXd coef_dummy_record(num_iter + 1, num_alpha);
  Eigen::MatrixXd coef_weight_record(num_iter + 1, num_grp);
  Eigen::MatrixXd contem_dummy_record(num_iter + 1, num_lowerchol);
  Eigen::MatrixXd contem_weight_record(num_iter + 1, num_lowerchol);
  // HS----------------
  Eigen::MatrixXd local_record(num_iter + 1, num_coef);
  Eigen::MatrixXd global_record(num_iter + 1, num_grp);
  Eigen::MatrixXd shrink_record(num_iter + 1, num_coef);
  // Initialize--------------------------------------------
  Eigen::VectorXd coefvec_ols = vectorize_eigen(coef_mat);
  coef_record.row(0) = coefvec_ols;
  contem_coef_record.row(0) = Eigen::VectorXd::Zero(num_lowerchol); // initialize a as 0
  lvol_init_record.row(0) = (y - x * coef_mat).transpose().array().square().rowwise().mean().log(); // initialize h0 as mean of log((y - x alpha)^T (y - x alpha))
  // lvol_record.block(0, 0, num_design, dim) = lvol_init_record.row(0).replicate(num_design, 1);
	Eigen::MatrixXd lvol_draw = lvol_init_record.row(0).replicate(num_design, 1); // h_j = (h_j1, ..., h_jn) for MCMC update
  lvol_sig_record.row(0) = .1 * Eigen::VectorXd::Ones(dim);
  // SSVS--------------
  coef_dummy_record.row(0) = Eigen::VectorXd::Ones(num_alpha);
  coef_weight_record.row(0) = coef_slab_weight;
  contem_dummy_record.row(0) = Eigen::VectorXd::Ones(num_lowerchol);
  contem_weight_record.row(0) = chol_slab_weight;
  // HS----------------
  local_record.row(0) = init_local;
  global_record.row(0) = init_global;
  // Some variables----------------------------------------
  Eigen::MatrixXd chol_lower = Eigen::MatrixXd::Zero(dim, dim); // L in Sig_t^(-1) = L D_t^(-1) LT
  Eigen::MatrixXd latent_innov(num_design, dim); // Z0 = Y0 - X0 A = (eps_p+1, eps_p+2, ..., eps_n+p)^T
  Eigen::MatrixXd ortho_latent(num_design, dim); // orthogonalized Z0
  // Eigen::MatrixXd lvol_draw = lvol_record.block(0, 0, num_design, dim);
  // Corrected triangular factorization-------
  Eigen::VectorXd prior_mean_j = Eigen::VectorXd::Zero(dim_design); // Prior mean vector of j-th column of A
  Eigen::MatrixXd prior_prec_j = Eigen::MatrixXd::Identity(dim_design, dim_design); // Prior precision of j-th column of A
  Eigen::MatrixXd coef_j = coef_mat; // j-th column of A = 0: A(-j) = (alpha_1, ..., alpha_(j-1), 0, alpha_(j), ..., alpha_k)
  coef_j.col(0) = Eigen::VectorXd::Zero(dim_design);
  Eigen::VectorXd response_contem(num_design); // j-th column of Z0 = Y0 - X0 * A: n-dim
  Eigen::MatrixXd sqrt_sv(num_design, dim); // stack sqrt of exp(h_t) = (exp(-h_1t / 2), ..., exp(-h_kt / 2)), t = 1, ..., n => n x k
  int contem_id = 0;
  // SSVS--------------
  Eigen::VectorXd prior_sd(num_coef);
  Eigen::VectorXd slab_weight(num_alpha); // pij vector
  Eigen::MatrixXd slab_weight_mat(num_alpha / dim, dim); // pij matrix: (dim*p) x dim
  Eigen::VectorXd coef_mixture_mat(num_alpha);
  Eigen::VectorXd contem_dummy = Eigen::VectorXd::Ones(num_lowerchol);
  // HS----------------
  Eigen::VectorXd latent_local(num_coef);
  Eigen::VectorXd latent_global(num_grp);
  Eigen::VectorXd contem_global(num_lowerchol);
  Eigen::VectorXd latent_contem_local(num_lowerchol);
  Eigen::VectorXd latent_contem_global(1);
  Eigen::VectorXd global_shrinkage(num_coef);
  Eigen::MatrixXd global_shrinkage_mat = Eigen::MatrixXd::Zero(dim_design, dim);
  Eigen::VectorXd grp_vec = vectorize_eigen(grp_mat);
  // Start Gibbs sampling-----------------------------------
	bvharprogress bar(num_iter, display_progress);
	bvharinterrupt();
  for (int i = 1; i < num_iter + 1; i ++) {
		if (bvharinterrupt::is_interrupted()) {
      if (prior_type == 2) {
        return Rcpp::List::create(
          Rcpp::Named("alpha_record") = coef_record,
          Rcpp::Named("h_record") = lvol_record,
          Rcpp::Named("a_record") = contem_coef_record,
          Rcpp::Named("h0_record") = lvol_init_record,
          Rcpp::Named("sigh_record") = lvol_sig_record,
          Rcpp::Named("gamma_record") = coef_dummy_record
        );
      } else if (prior_type == 3) {
        return Rcpp::List::create(
          Rcpp::Named("alpha_record") = coef_record,
          Rcpp::Named("h_record") = lvol_record,
          Rcpp::Named("a_record") = contem_coef_record,
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
        Rcpp::Named("a_record") = contem_coef_record,
        Rcpp::Named("h0_record") = lvol_init_record,
        Rcpp::Named("sigh_record") = lvol_sig_record
      );
    }
		bar.increment();
		if (display_progress) {
			bar.update();
		}
    // 1. alpha----------------------------
    chol_lower = build_inv_lower(dim, contem_coef_record.row(i - 1));
    switch(prior_type) {
    case 2:
      // SSVS
      coef_mixture_mat = build_ssvs_sd(coef_spike, coef_slab, coef_dummy_record.row(i - 1));
      if (include_mean) {
        for (int j = 0; j < dim; j++) {
          prior_sd.segment(j * dim_design, num_alpha / dim) = coef_mixture_mat.segment(
            j * num_alpha / dim,
            num_alpha / dim
          );
          prior_sd[j * dim_design + num_alpha / dim] = sd_non;
        }
      } else {
        prior_sd = coef_mixture_mat;
      }
      prior_alpha_prec.diagonal() = 1 / prior_sd.array().square();
      break;
    case 3:
      // HS
      for (int j = 0; j < num_grp; j++) {
        global_shrinkage_mat = (
          grp_mat.array() == grp_id[j]
        ).select(
          global_record.row(i - 1).segment(j, 1).replicate(dim_design, dim),
          global_shrinkage_mat
        );
      }
      global_shrinkage = vectorize_eigen(global_shrinkage_mat);
      prior_alpha_prec = build_shrink_mat(global_shrinkage, init_local);
			shrink_record.row(i - 1) = 1 / (1 + prior_alpha_prec.diagonal().array());
      break;
    }
    sqrt_sv = (-lvol_draw / 2).array().exp(); // n x k
    for (int j = 0; j < dim; j++) {
      prior_mean_j = prior_alpha_mean.segment(dim_design * j, dim_design);
      prior_prec_j = prior_alpha_prec.block(dim_design * j, dim_design * j, dim_design, dim_design);
      coef_j = coef_mat;
      coef_j.col(j) = Eigen::VectorXd::Zero(dim_design);
      Eigen::MatrixXd chol_lower_j = chol_lower.bottomRows(dim - j); // L_(j:k) = a_jt to a_kt for t = 1, ..., j - 1
      Eigen::MatrixXd sqrt_sv_j = sqrt_sv.rightCols(dim - j); // use h_jt to h_kt for t = 1, .. n => (k - j + 1) x k
      Eigen::MatrixXd design_coef = kronecker_eigen(chol_lower_j.col(j), x).array().colwise() * vectorize_eigen(sqrt_sv_j).array(); // L_(j:k, j) otimes X0 scaled by D_(1:n, j:k): n(k - j + 1) x kp
      Eigen::VectorXd response_j = vectorize_eigen(
        ((y - x * coef_j) * chol_lower_j.transpose()).array() * sqrt_sv_j.array() // Hadamard product between: (Y - X0 A(-j))L_(j:k)^T and D_(1:n, j:k)
      ); // Response vector of j-th column coef equation: n(k - j + 1)-dim
      coef_mat.col(j) = varsv_regression(
        design_coef, response_j,
        prior_mean_j, prior_prec_j
      );
    }
    coef_record.row(i) = vectorize_eigen(coef_mat);
    switch (prior_type) {
    case 2:
      // SSVS
      for (int j = 0; j < num_grp; j++) {
        slab_weight_mat = (
          grp_mat.array() == grp_id[j]
        ).select(
          coef_weight_record.row(i - 1).segment(j, 1).replicate(num_alpha / dim, dim),
          slab_weight_mat
        );
      }
      slab_weight = vectorize_eigen(slab_weight_mat);
      coef_dummy_record.row(i) = ssvs_dummy(
        vectorize_eigen(coef_mat.topRows(num_alpha / dim)),
        coef_slab, coef_spike,
        slab_weight
      );
      // coef_weight_record.row(i) = ssvs_weight(coef_dummy_record.row(i), coef_s1, coef_s2);
      coef_weight_record.row(i) = ssvs_mn_weight(
        grp_vec,
        grp_id,
        coef_dummy_record.row(i),
        coef_s1,
        coef_s2
      );
      break;
    case 3:
      // HS
      latent_local = horseshoe_latent(local_record.row(i - 1));
      latent_global = horseshoe_latent(global_record.row(i - 1));
      init_local = horseshoe_local_sparsity(
        latent_local, global_shrinkage,
        coef_record.row(i), 1
      );
      local_record.row(i) = init_local;
      global_record.row(i) = horseshoe_mn_global_sparsity(
        grp_vec,
        grp_id,
        latent_global,
        init_local,
        coef_record.row(i),
        1
      );
      break;
    default:
      break;
    }
    // 2. h---------------------------------
    latent_innov = y - x * coef_mat;
    ortho_latent = latent_innov * chol_lower.transpose(); // L eps_t <=> Z0 U
    ortho_latent = (ortho_latent.array().square() + .0001).array().log(); // adjustment log(e^2 + c) for some c = 10^(-4) against numerical problems
    for (int t = 0; t < dim; t++) {
      lvol_draw.col(t) = varsv_ht(
        lvol_draw.col(t),
        lvol_init_record(i - 1, t),
        lvol_sig_record(i - 1, t),
        ortho_latent.col(t)
      );
    }
    // lvol_record.block(num_design * i, 0, num_design, dim) = lvol_draw;
		lvol_record.row(i) = vectorize_eigen(lvol_draw.transpose());
    // 3. a---------------------------------
    switch (prior_type) {
    case 2:
      // SSVS
      contem_dummy = ssvs_dummy(
        contem_coef_record.row(i - 1),
        chol_slab,
        chol_spike,
        chol_slab_weight
      );
      contem_dummy_record.row(i) = contem_dummy;
      chol_slab_weight = ssvs_weight(contem_dummy, chol_s1, chol_s2);
      contem_weight_record.row(i) = chol_slab_weight;
      prior_chol_prec.diagonal() = 1 / build_ssvs_sd(chol_spike, chol_slab, contem_dummy).array().square();
      break;
    case 3:
      // HS
      latent_contem_local = horseshoe_latent(init_contem_local);
      latent_contem_global = horseshoe_latent(init_contem_global);
      contem_global = vectorize_eigen(init_contem_global.replicate(1, num_lowerchol));
      init_contem_local = horseshoe_local_sparsity(
        latent_contem_local,
        contem_global,
        contem_coef_record.row(i - 1),
        1
      );
      init_contem_global[0] = horseshoe_global_sparsity(
        latent_contem_global[0],
        latent_contem_local,
        contem_coef_record.row(i - 1),
        1
      );
      prior_chol_prec = build_shrink_mat(contem_global, init_contem_local);
      break;
    default:
      break;
    }
    for (int j = 2; j < dim + 1; j++) {
      response_contem = latent_innov.col(j - 2).array() * sqrt_sv.col(j - 2).array(); // n-dim
      Eigen::MatrixXd design_contem = latent_innov.leftCols(j - 1).array().colwise() * vectorize_eigen(sqrt_sv.col(j - 2)).array(); // n x (j - 1)
      contem_id = (j - 1) * (j - 2) / 2;
      contem_coef_record.block(i, contem_id, 1, j - 1).transpose() = varsv_regression(
        design_contem, response_contem,
        prior_chol_mean.segment(contem_id, j - 1),
        prior_chol_prec.block(contem_id, contem_id, j - 1, j - 1)
      );
    }
    // 4. sigma_h---------------------------
    lvol_sig_record.row(i) = varsv_sigh(
      prior_sig_shp,
      prior_sig_scl,
      lvol_init_record.row(i - 1),
      lvol_draw
    );
    // 5. h0--------------------------------
    lvol_init_record.row(i) = varsv_h0(
      prior_init_mean,
      prior_init_prec,
      lvol_init_record.row(i - 1),
      lvol_draw.row(0),
      lvol_sig_record.row(i)
    );
  }
  if (prior_type == 2) {
    return Rcpp::List::create(
      Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("h_record") = lvol_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("a_record") = contem_coef_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("h0_record") = lvol_init_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("sigh_record") = lvol_sig_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("gamma_record") = coef_dummy_record.bottomRows(num_iter - num_burn)
    );
  } else if (prior_type == 3) {
		shrink_record.row(num_iter) = 1 / (1 + prior_alpha_prec.diagonal().array());
    return Rcpp::List::create(
      Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("h_record") = lvol_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("a_record") = contem_coef_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("h0_record") = lvol_init_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("sigh_record") = lvol_sig_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("lambda_record") = local_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("tau_record") = global_record.bottomRows(num_iter - num_burn),
      Rcpp::Named("kappa_record") = shrink_record.bottomRows(num_iter - num_burn)
    );
  }
  return Rcpp::List::create(
    Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("h_record") = lvol_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("a_record") = contem_coef_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("h0_record") = lvol_init_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("sigh_record") = lvol_sig_record.bottomRows(num_iter - num_burn)
  );
}
