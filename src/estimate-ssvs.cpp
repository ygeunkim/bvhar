#include <RcppEigen.h>
#include "bvhardraw.h"
#include "bvharprogress.h"
#include "bvharinterrupt.h"

//' BVAR(p) SSVS by Gibbs Sampler
//' 
//' This function conducts Gibbs sampling for BVAR SSVS.
//' 
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in (warm-up) for MCMC
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param init_coef Initial k x m coefficient matrix.
//' @param init_chol_diag Inital diagonal cholesky factor
//' @param init_chol_upper Inital upper cholesky factor
//' @param init_coef_dummy Initial indicator vector (0-1) corresponding to each coefficient vector
//' @param init_chol_dummy Initial indicator vector (0-1) corresponding to each upper cholesky factor vector
//' @param coef_spike Standard deviance for Spike normal distribution
//' @param coef_slab Standard deviance for Slab normal distribution
//' @param coef_slab_weight Coefficients vector sparsity proportion
//' @param shape Gamma shape parameters for precision matrix
//' @param rate Gamma rate parameters for precision matrix
//' @param coef_s1 First shape of prior beta distribution of coefficients slab weight
//' @param coef_s2 Second shape of prior beta distribution of coefficients slab weight
//' @param chol_spike Standard deviance for cholesky factor Spike normal distribution
//' @param chol_slab Standard deviance for cholesky factor Slab normal distribution
//' @param chol_slab_weight Cholesky factor sparsity proportion
//' @param chol_s1 First shape of prior beta distribution of cholesky factor slab weight
//' @param chol_s2 Second shape of prior beta distribution of cholesky factor slab weight
//' @param grp_id Unique group id
//' @param grp_mat Group matrix
//' @param mean_non Prior mean of unrestricted coefficients
//' @param sd_non Standard deviance for unrestricted coefficients
//' @param include_mean Add constant term
//' @param init_gibbs Set custom initial values for Gibbs sampler
//' @param display_progress Progress bar
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_ssvs(int num_iter, int num_burn,
                              Eigen::MatrixXd x, Eigen::MatrixXd y, 
                              Eigen::VectorXd init_coef,
                              Eigen::VectorXd init_chol_diag, Eigen::VectorXd init_chol_upper,
                              Eigen::VectorXd init_coef_dummy, Eigen::VectorXd init_chol_dummy,
                              Eigen::VectorXd coef_spike, Eigen::VectorXd coef_slab, Eigen::VectorXd coef_slab_weight,
                              Eigen::VectorXd shape, Eigen::VectorXd rate,
                              double coef_s1, double coef_s2,
                              Eigen::VectorXd chol_spike, Eigen::VectorXd chol_slab, Eigen::VectorXd chol_slab_weight,
                              double chol_s1, double chol_s2,
                              Eigen::VectorXi grp_id,
                              Eigen::MatrixXd grp_mat,
                              Eigen::VectorXd mean_non, double sd_non,
                              bool include_mean,
                              bool init_gibbs,
                              bool display_progress) {
  int dim = y.cols();
  int dim_design = x.cols(); // dim*p(+1)
  int num_design = y.rows(); // n = T - p
  int num_upperchol = chol_slab.size(); // number of upper cholesky = dim (dim - 1) / 2
  int num_grp = grp_id.size();
  // Initialize coefficients vector-------------------------------
  int num_coef = dim * dim_design; // dim^2 p + dim vs dim^2 p (if no constant)
  int num_restrict = num_coef - dim; // number of restricted coefs: dim^2 p vs dim^2 p - dim (if no constant)
  if (!include_mean) {
    num_restrict += dim; // always dim^2 p
  }
  Eigen::VectorXd prior_mean(num_coef);
  Eigen::VectorXd coef_mean = Eigen::VectorXd::Zero(num_restrict); // zero prior mean for restricted coefficient
  if (include_mean) {
    for (int j = 0; j < dim; j++) {
      prior_mean.segment(j * dim_design, num_restrict / dim) = coef_mean.segment(j * num_restrict / dim, num_restrict / dim);
      prior_mean[j * dim_design + num_restrict / dim] = mean_non[j];
    }
  } else {
    prior_mean = coef_mean;
  }
  Eigen::VectorXd prior_sd(num_coef); // M: diagonal matrix = DRD or merge of cI_dim and DRD
  Eigen::VectorXd coef_mixture_mat(num_restrict); // D = diag(hj)
  Eigen::MatrixXd XtX = x.transpose() * x;
  // Eigen::MatrixXd coef_ols = XtX.inverse() * x.transpose() * y;
  Eigen::MatrixXd coef_ols = XtX.llt().solve(x.transpose() * y);
  Eigen::MatrixXd cov_ols = (y - x * coef_ols).transpose() * (y - x * coef_ols) / (num_design - dim_design);
  Eigen::LLT<Eigen::MatrixXd> lltOfscale(cov_ols.inverse());
  Eigen::MatrixXd chol_ols = lltOfscale.matrixU();
  Eigen::VectorXd coefvec_ols = vectorize_eigen(coef_ols);
  // record-------------------------------------------------------
  Eigen::MatrixXd coef_record(num_iter + 1, num_coef);
  Eigen::MatrixXd coef_dummy_record(num_iter + 1, num_restrict);
  Eigen::MatrixXd coef_weight_record(num_iter + 1, num_grp);
  Eigen::MatrixXd chol_diag_record(num_iter + 1, dim);
  Eigen::MatrixXd chol_upper_record(num_iter + 1, num_upperchol);
  Eigen::MatrixXd chol_dummy_record(num_iter + 1, num_upperchol);
  Eigen::MatrixXd chol_weight_record(num_iter + 1, num_upperchol);
  Eigen::MatrixXd chol_factor_record(dim * (num_iter + 1), dim); // 3d matrix alternative
  coef_weight_record.row(0) = coef_slab_weight;
  chol_weight_record.row(0) = chol_slab_weight;
  if (init_gibbs) {
    coef_record.row(0) = init_coef;
    coef_dummy_record.row(0) = init_coef_dummy;
    chol_diag_record.row(0) = init_chol_diag;
    chol_upper_record.row(0) = init_chol_upper;
    chol_dummy_record.row(0) = init_chol_dummy;
    chol_factor_record.topLeftCorner(dim, dim) = build_chol(init_chol_diag, init_chol_upper);
  } else {
    coef_record.row(0) = coefvec_ols;
    coef_dummy_record.row(0) = Eigen::VectorXd::Ones(num_restrict);
    chol_diag_record.row(0) = chol_ols.diagonal();
    for (int i = 1; i < dim; i++) {
      chol_upper_record.block(0, i * (i - 1) / 2, 1, i) = chol_ols.block(0, i, i, 1).transpose();
    }
    chol_dummy_record.row(0) = Eigen::VectorXd::Ones(num_upperchol);
    chol_factor_record.topLeftCorner(dim, dim) = chol_ols;
  }
  // Some variables-----------------------------------------------
  Eigen::MatrixXd coef_mat = unvectorize(coef_record.row(0), dim_design, dim); // coefficient matrix to compute sse_mat
  Eigen::MatrixXd sse_mat = (y - x * coef_mat).transpose() * (y - x * coef_mat);
  Eigen::VectorXd chol_mixture_mat(num_upperchol); // Dj = diag(h1j, ..., h(j-1,j))
  Eigen::VectorXd slab_weight(num_restrict); // pij vector
  Eigen::MatrixXd slab_weight_mat(num_restrict / dim, dim); // pij matrix: (dim*p) x dim
  Eigen::VectorXd grp_vec = vectorize_eigen(grp_mat);
  bvharprogress bar(num_iter, display_progress);
	bvharinterrupt();
  // Start Gibbs sampling-----------------------------------------
  for (int i = 1; i < num_iter + 1; i++) {
    if (bvharinterrupt::is_interrupted()) {
      return Rcpp::List::create(
        Rcpp::Named("alpha_record") = coef_record,
        Rcpp::Named("eta_record") = chol_upper_record,
        Rcpp::Named("psi_record") = chol_diag_record,
        Rcpp::Named("omega_record") = chol_dummy_record,
        Rcpp::Named("gamma_record") = coef_dummy_record,
        Rcpp::Named("chol_record") = chol_factor_record,
        Rcpp::Named("p_record") = coef_weight_record,
        Rcpp::Named("q_record") = chol_weight_record,
        Rcpp::Named("ols_coef") = coef_ols,
        Rcpp::Named("ols_cholesky") = chol_ols
      );
    }
    bar.increment();
		if (display_progress) {
			bar.update();
		}
    // 1. Psi--------------------------
    chol_mixture_mat = build_ssvs_sd(chol_spike, chol_slab, chol_dummy_record.row(i - 1));
    chol_diag_record.row(i) = ssvs_chol_diag(sse_mat, chol_mixture_mat, shape, rate, num_design);
    // 2. eta---------------------------
    chol_upper_record.row(i) = ssvs_chol_off(sse_mat, chol_diag_record.row(i), chol_mixture_mat);
    chol_factor_record.block(i * dim, 0, dim, dim) = build_chol(chol_diag_record.row(i), chol_upper_record.row(i));
    // 3. omega--------------------------
    chol_dummy_record.row(i) = ssvs_dummy(
      chol_upper_record.row(i),
      chol_slab,
      chol_spike,
      chol_weight_record.row(i - 1)
    );
    // qij
    chol_weight_record.row(i) = ssvs_weight(chol_dummy_record.row(i), chol_s1, chol_s2);
    // 4. alpha--------------------------
    coef_mixture_mat = build_ssvs_sd(coef_spike, coef_slab, coef_dummy_record.row(i - 1));
    if (include_mean) {
      for (int j = 0; j < dim; j++) {
        prior_sd.segment(j * dim_design, num_restrict / dim) = coef_mixture_mat.segment(j * num_restrict / dim, num_restrict / dim);
        prior_sd[j * dim_design + num_restrict / dim] = sd_non;
      }
    } else {
      prior_sd = coef_mixture_mat;
    }
    coef_record.row(i) = ssvs_coef(prior_mean, prior_sd, XtX, coefvec_ols, chol_factor_record.block(i * dim, 0, dim, dim));
    coef_mat = unvectorize(coef_record.row(i), dim_design, dim);
    sse_mat = (y - x * coef_mat).transpose() * (y - x * coef_mat);
    // 5. gamma-------------------------
    for (int j = 0; j < num_grp; j++) {
      slab_weight_mat = (
        grp_mat.array() == grp_id[j]
      ).select(
        coef_weight_record.row(i - 1).segment(j, 1).replicate(num_restrict / dim, dim),
        slab_weight_mat
      );
    }
    slab_weight = vectorize_eigen(slab_weight_mat);
    coef_dummy_record.row(i) = ssvs_dummy(
      vectorize_eigen(coef_mat.topRows(num_restrict / dim)),
      coef_slab,
      coef_spike,
      slab_weight
    );
    // p
    coef_weight_record.row(i) = ssvs_mn_weight(
      grp_vec,
      grp_id,
      coef_dummy_record.row(i),
      coef_s1,
      coef_s2
    );
  }
  return Rcpp::List::create(
    Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("eta_record") = chol_upper_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("psi_record") = chol_diag_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("omega_record") = chol_dummy_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("gamma_record") = coef_dummy_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("chol_record") = chol_factor_record.bottomRows(dim * (num_iter - num_burn)),
    Rcpp::Named("p_record") = coef_weight_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("q_record") = chol_weight_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("ols_coef") = coef_ols,
    Rcpp::Named("ols_cholesky") = chol_ols
  );
}
