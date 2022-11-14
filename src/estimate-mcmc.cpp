#ifdef _OPENMP
#include <omp.h>
// [[Rcpp::plugins(openmp)]]
#endif
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

//' BVAR(p) SSVS by Gibbs Sampler
//' 
//' This function conducts Gibbs sampling for BVAR SSVS.
//' 
//' @param num_iter Number of iteration for MCMC
//' @param num_burn Number of burn-in for MCMC
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param init_coef Initial k x m coefficient matrix.
//' @param init_chol_diag Inital diagonal cholesky factor
//' @param init_chol_upper Inital upper cholesky factor
//' @param init_coef_dummy Initial indicator vector (0-1) corresponding to each coefficient vector
//' @param init_chol_dummy Initial indicator vector (0-1) corresponding to each upper cholesky factor vector
//' @param coef_slab_weight Bernoulli parameter for coefficients vector
//' @param coef_spike Standard deviance for Spike normal distribution
//' @param coef_slab Standard deviance for Slab normal distribution
//' @param coef_slab_weight Bernoulli parameter for coefficients sparsity proportion
//' @param shape Gamma shape parameters for precision matrix
//' @param rate Gamma rate parameters for precision matrix
//' @param chol_spike Standard deviance for cholesky factor Spike normal distribution
//' @param chol_slab Standard deviance for cholesky factor Slab normal distribution
//' @param chol_slab_weight Bernoulli parameter for cholesky factor sparsity proportion
//' @param intercept_var Hyperparameter for constant term
//' @param chain The number of MCMC chains.
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_ssvs(int num_iter,
                              int num_burn,
                              Eigen::MatrixXd x, 
                              Eigen::MatrixXd y, 
                              Eigen::VectorXd init_coef,
                              Eigen::VectorXd init_chol_diag,
                              Eigen::VectorXd init_chol_upper,
                              Eigen::VectorXd init_coef_dummy,
                              Eigen::VectorXd init_chol_dummy,
                              Eigen::VectorXd coef_spike,
                              Eigen::VectorXd coef_slab,
                              Eigen::VectorXd coef_slab_weight,
                              Eigen::VectorXd shape,
                              Eigen::VectorXd rate,
                              Eigen::VectorXd chol_spike,
                              Eigen::VectorXd chol_slab,
                              Eigen::VectorXd chol_slab_weight,
                              double intercept_var,
                              int chain) {
  int dim = y.cols();
  int dim_design = x.cols(); // dim*p(+1)
  int num_design = y.rows(); // n = T - p
  int num_upperchol = chol_slab.size(); // number of upper cholesky = dim (dim - 1) / 2
  // Initialize coefficients vector-------------------------------
  int num_coef = dim * dim_design; // dim^2 p + dim vs dim^2 p (if no constant)
  int num_restrict = num_coef - dim; // number of restricted coefs: dim^2 p vs dim^2 p - dim (if no constant)
  int num_non = num_coef - num_restrict; // number of unrestricted coefs (constant vector): dim vs -dim (if no constant)
  if (num_non == -dim) {
    num_restrict += dim; // always dim^2 p
  }
  Eigen::VectorXd prior_mean = Eigen::VectorXd::Zero(num_coef); // zero vector as prior mean
  Eigen::MatrixXd prior_variance = Eigen::MatrixXd::Zero(num_coef, num_coef); // M: diagonal matrix = DRD or merge of cI_dim and DRD
  Eigen::MatrixXd coef_mixture_mat = Eigen::MatrixXd::Zero(num_restrict, num_restrict); // D
  Eigen::MatrixXd DRD = Eigen::MatrixXd::Zero(num_restrict, num_restrict); // DRD
  Eigen::MatrixXd XtX = x.transpose() * x; // X_0^T X_0: k x k
  Eigen::MatrixXd coef_ols = XtX.inverse() * x.transpose() * y;
  Eigen::MatrixXd cov_ols = (y - x * coef_ols).transpose() * (y - x * coef_ols) / (num_design - dim_design);
  Eigen::LLT<Eigen::MatrixXd> lltOfscale(cov_ols);
  Eigen::MatrixXd chol_ols = lltOfscale.matrixU();
  Eigen::VectorXd coefvec_ols = vectorize_eigen(coef_ols);
  // record-------------------------------------------------------
  Eigen::MatrixXd coef_record = Eigen::MatrixXd::Zero(num_iter, num_coef * chain);
  coef_record.row(0) = init_coef;
  Eigen::MatrixXd coef_dummy_record = Eigen::MatrixXd::Zero(num_iter, num_restrict * chain);
  coef_dummy_record.row(0) = init_coef_dummy;
  Eigen::MatrixXd chol_diag_record = Eigen::MatrixXd::Zero(num_iter, dim * chain);
  chol_diag_record.row(0) = init_chol_diag;
  Eigen::MatrixXd chol_upper_record = Eigen::MatrixXd::Zero(num_iter, num_upperchol * chain);
  chol_upper_record.row(0) = init_chol_upper;
  Eigen::MatrixXd chol_dummy_record = Eigen::MatrixXd::Zero(num_iter, num_upperchol * chain);
  chol_dummy_record.row(0) = init_chol_dummy;
  Eigen::MatrixXd chol_factor_record = Eigen::MatrixXd::Zero(dim * num_iter, dim * chain); // 3d matrix alternative
  // Some variables-----------------------------------------------
  Eigen::MatrixXd coef_mat = unvectorize(init_coef, dim_design, dim); // coefficient matrix to compute sse_mat
  Eigen::MatrixXd sse_mat = (y - x * coef_mat).transpose() * (y - x * coef_mat);
  Eigen::MatrixXd chol_mixture_mat(num_upperchol, num_upperchol); // Dj = diag(h1j, ..., h(j-1,j))
  Eigen::MatrixXd chol_prior_prec = Eigen::MatrixXd::Zero(num_upperchol, num_upperchol); // DjRjDj^(-1)
  // Start Gibbs sampling-----------------------------------------
  #ifdef _OPENMP
  Rcpp::Rcout << "Use parallel" << std::endl;
  #pragma \
  omp \
    parallel \
    for \
      num_threads(chain) \
      shared(prior_mean, XtX, coefvec_ols, dim, dim_design, num_restrict, num_non, num_design, num_upperchol,
             coef_spike, coef_slab, coef_slab_weight, shape, rate, chol_spike, chol_slab, chol_slab_weight, intercept_var)
  for (int b = 0; b < chain; b++) {
    for (int i = 1; i < num_iter; i++) {
      // 1. Psi--------------------------
      chol_mixture_mat = build_ssvs_sd(
        chol_spike,
        chol_slab,
        chol_dummy_record.block(i - 1, b * num_upperchol, 1, num_upperchol)
      );
      chol_prior_prec = (chol_mixture_mat * chol_mixture_mat).inverse();
      chol_diag_record.block(i, b * dim, 1, dim) = ssvs_chol_diag(sse_mat, chol_prior_prec, shape, rate, num_design);
      // 2. eta---------------------------
      chol_upper_record.block(i, b * num_upperchol, 1, num_upperchol) = ssvs_chol_off(sse_mat, chol_diag_record.block(i, b * dim, 1, dim), chol_prior_prec);
      chol_factor_record.block(i * dim, b * dim, dim, dim) = build_chol(chol_diag_record.block(i, b * dim, 1, dim), chol_upper_record.block(i, b * num_upperchol, 1, num_upperchol));
      // 3. omega--------------------------
      chol_dummy_record.block(i, b * num_upperchol, 1, num_upperchol) = ssvs_chol_dummy(chol_upper_record.block(i, b * num_upperchol, 1, num_upperchol), chol_spike, chol_slab, chol_slab_weight);
      // 4. alpha--------------------------
      coef_mixture_mat = build_ssvs_sd(coef_spike, coef_slab, chol_dummy_record.block(i, b * num_upperchol, 1, num_upperchol));
      DRD = coef_mixture_mat * coef_mixture_mat;
      if (num_non == dim) {
        // constant case
        for (int j = 0; j < dim; j++) {
          prior_variance.block(j * dim_design, j * dim_design, num_restrict / dim, num_restrict / dim) =
            DRD.block(j * num_restrict / dim, j * num_restrict / dim, num_restrict / dim, num_restrict / dim); // kp x kp
          prior_variance(j * dim_design + num_restrict / dim, j * dim_design + num_restrict / dim) = intercept_var;
        }
      } else if (num_non == -dim) {
        // no constant term
        prior_variance = DRD;
      }
      coef_record.block(i, b * num_coef, 1, num_coef) = ssvs_coef(
        prior_mean, 
        prior_variance.inverse(), 
        XtX, 
        coefvec_ols, 
        chol_factor_record.block(i * dim, b * dim, dim, dim)
      );
      coef_mat = unvectorize(coef_record.block(i, b * num_coef, 1, num_coef), dim_design, dim);
      sse_mat = (y - x * coef_mat).transpose() * (y - x * coef_mat);
      // 5. gamma-------------------------
      coef_dummy_record.block(i, b * num_restrict, 1, num_restrict) = ssvs_coef_dummy(
        vectorize_eigen(coef_mat.topRows(num_restrict / dim)), 
        coef_spike, 
        coef_slab, 
        coef_slab_weight
      );
    }
  }
  #else
  for (int i = 1; i < num_iter; i++) {
    // 1. Psi--------------------------
    chol_mixture_mat = build_ssvs_sd(
      chol_spike,
      chol_slab,
      chol_dummy_record.row(i - 1)
    );
    chol_prior_prec = (chol_mixture_mat * chol_mixture_mat).inverse();
    chol_diag_record.row(i) = ssvs_chol_diag(sse_mat, chol_prior_prec, shape, rate, num_design);
    // 2. eta---------------------------
    chol_upper_record.row(i) = ssvs_chol_off(sse_mat, chol_diag_record.row(i), chol_prior_prec);
    chol_factor_record.block(i * dim, 0, dim, dim) = build_chol(chol_diag_record.row(i), chol_upper_record.row(i));
    // 3. omega--------------------------
    chol_dummy_record.row(i) = ssvs_chol_dummy(chol_upper_record.row(i), chol_spike, chol_slab, chol_slab_weight);
    // 4. alpha--------------------------
    coef_mixture_mat = build_ssvs_sd(coef_spike, coef_slab, coef_dummy_record.row(i - 1));
    DRD = coef_mixture_mat * coef_mixture_mat;
    if (num_non == dim) {
      // constant case
      for (int j = 0; j < dim; j++) {
        prior_variance.block(j * dim_design, j * dim_design, num_restrict / dim, num_restrict / dim) =
          DRD.block(j * num_restrict / dim, j * num_restrict / dim, num_restrict / dim, num_restrict / dim); // kp x kp
        prior_variance(j * dim_design + num_restrict / dim, j * dim_design + num_restrict / dim) = intercept_var;
      }
    } else if (num_non == -dim) {
      // no constant term
      prior_variance = DRD;
    }
    coef_record.row(i) = ssvs_coef(
      prior_mean,
      prior_variance.inverse(),
      XtX,
      coefvec_ols,
      chol_factor_record.block(i * dim, 0, dim, dim)
    );
    coef_mat = unvectorize(coef_record.row(i), dim_design, dim);
    sse_mat = (y - x * coef_mat).transpose() * (y - x * coef_mat);
    // 5. gamma-------------------------
    coef_dummy_record.row(i) = ssvs_coef_dummy(
      vectorize_eigen(coef_mat.topRows(num_restrict / dim)), 
      coef_spike, 
      coef_slab, 
      coef_slab_weight
    );
  }
  #endif
  return Rcpp::List::create(
    Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("eta_record") = chol_upper_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("psi_record") = chol_diag_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("omega_record") = chol_dummy_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("gamma_record") = coef_dummy_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("alpha_posterior") = coef_record.bottomRows(1),
    Rcpp::Named("omega_posterior") = chol_dummy_record.bottomRows(1),
    Rcpp::Named("gamma_posterior") = coef_dummy_record.bottomRows(1),
    Rcpp::Named("chol_record") = chol_factor_record,
    Rcpp::Named("chol_posterior") = chol_factor_record.bottomRows(dim),
    Rcpp::Named("sse") = sse_mat,
    Rcpp::Named("coefficients") = coef_ols,
    Rcpp::Named("choleskyols") = chol_ols,
    Rcpp::Named("chain") = chain
  );
}
