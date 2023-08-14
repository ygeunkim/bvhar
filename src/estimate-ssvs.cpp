#include "bvharomp.h"
#include <RcppEigen.h>
#include "bvharmisc.h"
#include "bvharprob.h"
#include <progress.hpp>
#include <progress_bar.hpp>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppProgress)]]

//' Building Spike-and-slab SD Diagonal Matrix
//' 
//' In MCMC process of SSVS, this function computes diagonal matrix \eqn{D} or \eqn{D_j} defined by spike-and-slab sd.
//' 
//' @param spike_sd Standard deviance for Spike normal distribution
//' @param slab_sd Standard deviance for Slab normal distribution
//' @param mixture_dummy Indicator vector (0-1) corresponding to each element
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd build_ssvs_sd(Eigen::VectorXd spike_sd, Eigen::VectorXd slab_sd, Eigen::VectorXd mixture_dummy) {
  Eigen::VectorXd res(spike_sd.size());
  res.array() = (1 - mixture_dummy.array()) * spike_sd.array() + mixture_dummy.array() * slab_sd.array(); // diagonal term = spike_sd if mixture_dummy = 0 while slab_sd if mixture_dummy = 1
  return res;
}

//' Generating the Diagonal Component of Cholesky Factor in SSVS Gibbs Sampler
//' 
//' In MCMC process of SSVS, this function generates the diagonal component \eqn{\Psi} from variance matrix
//' 
//' @param sse_mat The result of \eqn{Z_0^T Z_0 = (Y_0 - X_0 \hat{A})^T (Y_0 - X_0 \hat{A})}
//' @param DRD Inverse of matrix product between \eqn{D_j} and correlation matrix \eqn{R_j}
//' @param shape Gamma shape parameters for precision matrix
//' @param rate Gamma rate parameters for precision matrix
//' @param num_design The number of sample used, \eqn{n = T - p}
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_chol_diag(Eigen::MatrixXd sse_mat, Eigen::VectorXd DRD, Eigen::VectorXd shape, Eigen::VectorXd rate, int num_design) {
  int dim = sse_mat.cols();
  int num_param = DRD.size();
  Eigen::VectorXd res(dim);
  Eigen::MatrixXd inv_DRD = Eigen::MatrixXd::Zero(num_param, num_param);
  inv_DRD.diagonal() = 1 / DRD.array().square();
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
        (sse_mat.topLeftCorner(j, j) + inv_DRD.block(block_id, block_id, j, j)).llt().solve(Eigen::MatrixXd::Identity(j, j)) * 
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
//' @param DRD Inverse of matrix product between \eqn{D_j} and correlation matrix \eqn{R_j}
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_chol_off(Eigen::MatrixXd sse_mat, Eigen::VectorXd chol_diag, Eigen::VectorXd DRD) {
  int dim = sse_mat.cols();
  int num_param = DRD.size();
  Eigen::MatrixXd normal_variance(dim - 1, dim - 1);
  Eigen::VectorXd sse_colvec(dim - 1); // sj = (s1j, ..., s(j-1, j)) from SSE
  Eigen::VectorXd normal_mean(dim - 1);
  Eigen::VectorXd res(num_param);
  Eigen::MatrixXd inv_DRD = Eigen::MatrixXd::Zero(num_param, num_param);
  inv_DRD.diagonal() = 1 / DRD.array().square();
  int block_id = 0;
  for (int j = 1; j < dim; j++) {
    sse_colvec.segment(0, j) = sse_mat.block(0, j, j, 1);
    normal_variance.topLeftCorner(j, j) = (sse_mat.topLeftCorner(j, j) + inv_DRD.block(block_id, block_id, j, j)).llt().solve(Eigen::MatrixXd::Identity(j, j));
    normal_mean.segment(0, j) = -chol_diag[j] * normal_variance.topLeftCorner(j, j) * sse_colvec.segment(0, j);
    res.segment(block_id, j) = vectorize_eigen(sim_mgaussian_chol(1, normal_mean.segment(0, j), normal_variance.topLeftCorner(j, j)));
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

//' Generating Coefficient Vector in SSVS Gibbs Sampler
//' 
//' In MCMC process of SSVS, this function generates \eqn{\alpha_j} conditional posterior.
//' 
//' @param prior_mean The prior mean vector of the VAR coefficient vector
//' @param prior_sd Diagonal prior sd matrix of the VAR coefficient vector
//' @param XtX The result of design matrix arithmetic \eqn{X_0^T X_0}
//' @param coef_ols OLS (MLE) estimator of the VAR coefficient
//' @param chol_factor Cholesky factor of variance matrix
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_coef(Eigen::VectorXd prior_mean, Eigen::VectorXd prior_sd, Eigen::MatrixXd XtX, Eigen::VectorXd coef_ols, Eigen::MatrixXd chol_factor) {
  int num_coef = prior_sd.size();
  Eigen::MatrixXd scaled_xtx = kronecker_eigen(chol_factor * chol_factor.transpose(), XtX); // Sigma^(-1) = chol * chol^T
  // Eigen::MatrixXd scaled_xtx = Eigen::kroneckerProduct(chol_factor * chol_factor.transpose(), XtX).eval(); // Sigma^(-1) = chol * chol^T
  Eigen::MatrixXd prior_prec = Eigen::MatrixXd::Zero(num_coef, num_coef);
  prior_prec.diagonal() = 1 / prior_sd.array().square();
  Eigen::MatrixXd normal_variance = (scaled_xtx + prior_prec).llt().solve(Eigen::MatrixXd::Identity(num_coef, num_coef)); // Delta
  Eigen::VectorXd normal_mean = normal_variance * (scaled_xtx * coef_ols + prior_prec * prior_mean); // mu
  return vectorize_eigen(sim_mgaussian_chol(1, normal_mean, normal_variance));
}

//' Generating Dummy Vector for Parameters in SSVS Gibbs Sampler
//' 
//' In MCMC process of SSVS, this function generates latent \eqn{\gamma_j} or \eqn{\omega_{ij}} conditional posterior.
//' 
//' @param param_obs Realized parameters vector
//' @param sd_numer Standard deviance for Slab normal distribution, which will be used for numerator.
//' @param sd_denom Standard deviance for Spike normal distribution, which will be used for denominator.
//' @param slab_weight Proportion of nonzero coefficients
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_dummy(Eigen::VectorXd param_obs, Eigen::VectorXd sd_numer, Eigen::VectorXd sd_denom, Eigen::VectorXd slab_weight) {
  int num_latent = slab_weight.size();
  Eigen::VectorXd bernoulli_param_u1 = slab_weight.array() * (-param_obs.array().square() / (2 * sd_numer.array().square())).exp() / sd_numer.array();
  Eigen::VectorXd bernoulli_param_u2 = (1 - slab_weight.array()) * (-param_obs.array().square() / (2 * sd_denom.array().square())).exp() / sd_denom.array();
  Eigen::VectorXd res(num_latent); // latentj | Y0, -latentj ~ Bernoulli(u1 / (u1 + u2))
  for (int i = 0; i < num_latent; i++) {
    res[i] = binom_rand(1, bernoulli_param_u1[i] / (bernoulli_param_u1[i] + bernoulli_param_u2[i]));
  }
  return res;
}

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
//' @param chol_spike Standard deviance for cholesky factor Spike normal distribution
//' @param chol_slab Standard deviance for cholesky factor Slab normal distribution
//' @param chol_slab_weight Cholesky factor sparsity proportion
//' @param intercept_mean Prior mean of unrestricted coefficients
//' @param intercept_sd Standard deviance for unrestricted coefficients
//' @param include_mean Add constant term
//' @param init_gibbs Set custom initial values for Gibbs sampler
//' @param display_progress Progress bar
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_ssvs(int num_iter, int num_burn,
                              Eigen::MatrixXd x, Eigen::MatrixXd y, 
                              Eigen::VectorXd init_coef, Eigen::VectorXd init_chol_diag, Eigen::VectorXd init_chol_upper, Eigen::VectorXd init_coef_dummy, Eigen::VectorXd init_chol_dummy,
                              Eigen::VectorXd coef_spike, Eigen::VectorXd coef_slab, Eigen::VectorXd coef_slab_weight,
                              Eigen::VectorXd shape, Eigen::VectorXd rate, Eigen::VectorXd chol_spike, Eigen::VectorXd chol_slab, Eigen::VectorXd chol_slab_weight,
                              Eigen::VectorXd intercept_mean, double intercept_sd,
                              bool include_mean,
                              bool init_gibbs,
                              bool display_progress) {
  int dim = y.cols();
  int dim_design = x.cols(); // dim*p(+1)
  int num_design = y.rows(); // n = T - p
  int num_upperchol = chol_slab.size(); // number of upper cholesky = dim (dim - 1) / 2
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
      prior_mean[j * dim_design + num_restrict / dim] = intercept_mean[j];
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
  Eigen::MatrixXd chol_diag_record(num_iter + 1, dim);
  Eigen::MatrixXd chol_upper_record(num_iter + 1, num_upperchol);
  Eigen::MatrixXd chol_dummy_record(num_iter + 1, num_upperchol);
  Eigen::MatrixXd chol_factor_record(dim * (num_iter + 1), dim); // 3d matrix alternative
  if (init_gibbs) {
    coef_record.row(0) = init_coef;
    coef_dummy_record.row(0) = init_coef_dummy;
    chol_diag_record.row(0) = init_chol_diag;
    chol_upper_record.row(0) = init_chol_upper;
    chol_dummy_record.row(0) = init_chol_dummy;
    chol_factor_record.topLeftCorner(dim, dim) = build_chol(init_chol_diag, init_chol_upper);
  } else {
    coef_record.row(0) = coefvec_ols;
    coef_dummy_record.row(0) = Eigen::MatrixXd::Identity(num_restrict, num_restrict).diagonal();
    chol_diag_record.row(0) = chol_ols.diagonal();
    for (int i = 1; i < dim; i++) {
      chol_upper_record.block(0, i * (i - 1) / 2, 1, i) = chol_ols.block(0, i, i, 1).transpose();
    }
    chol_dummy_record.row(0) = Eigen::MatrixXd::Identity(num_upperchol, num_upperchol).diagonal();
    chol_factor_record.topLeftCorner(dim, dim) = chol_ols;
  }
  // Some variables-----------------------------------------------
  Eigen::MatrixXd coef_mat = unvectorize(coef_record.row(0), dim_design, dim); // coefficient matrix to compute sse_mat
  Eigen::MatrixXd sse_mat = (y - x * coef_mat).transpose() * (y - x * coef_mat);
  Eigen::VectorXd chol_mixture_mat(num_upperchol); // Dj = diag(h1j, ..., h(j-1,j))
  Progress p(num_iter, display_progress);
  // Start Gibbs sampling-----------------------------------------
  for (int i = 1; i < num_iter + 1; i++) {
    if (Progress::check_abort()) {
      return Rcpp::List::create(
        Rcpp::Named("alpha_record") = coef_record,
        Rcpp::Named("eta_record") = chol_upper_record,
        Rcpp::Named("psi_record") = chol_diag_record,
        Rcpp::Named("omega_record") = chol_dummy_record,
        Rcpp::Named("gamma_record") = coef_dummy_record,
        Rcpp::Named("chol_record") = chol_factor_record,
        Rcpp::Named("ols_coef") = coef_ols,
        Rcpp::Named("ols_cholesky") = chol_ols
      );
    }
    p.increment();
    // 1. Psi--------------------------
    chol_mixture_mat = build_ssvs_sd(chol_spike, chol_slab, chol_dummy_record.row(i - 1));
    chol_diag_record.row(i) = ssvs_chol_diag(sse_mat, chol_mixture_mat, shape, rate, num_design);
    // 2. eta---------------------------
    chol_upper_record.row(i) = ssvs_chol_off(sse_mat, chol_diag_record.row(i), chol_mixture_mat);
    chol_factor_record.block(i * dim, 0, dim, dim) = build_chol(chol_diag_record.row(i), chol_upper_record.row(i));
    // 3. omega--------------------------
    chol_dummy_record.row(i) = ssvs_dummy(chol_upper_record.row(i), chol_slab, chol_spike, chol_slab_weight);
    // 4. alpha--------------------------
    coef_mixture_mat = build_ssvs_sd(coef_spike, coef_slab, coef_dummy_record.row(i - 1));
    if (include_mean) {
      for (int j = 0; j < dim; j++) {
        prior_sd.segment(j * dim_design, num_restrict / dim) = coef_mixture_mat.segment(j * num_restrict / dim, num_restrict / dim);
        prior_sd[j * dim_design + num_restrict / dim] = intercept_sd;
      }
    } else {
      prior_sd = coef_mixture_mat;
    }
    coef_record.row(i) = ssvs_coef(prior_mean, prior_sd, XtX, coefvec_ols, chol_factor_record.block(i * dim, 0, dim, dim));
    coef_mat = unvectorize(coef_record.row(i), dim_design, dim);
    sse_mat = (y - x * coef_mat).transpose() * (y - x * coef_mat);
    // 5. gamma-------------------------
    coef_dummy_record.row(i) = ssvs_dummy(vectorize_eigen(coef_mat.topRows(num_restrict / dim)), coef_slab, coef_spike, coef_slab_weight);
  }
  return Rcpp::List::create(
    Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("eta_record") = chol_upper_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("psi_record") = chol_diag_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("omega_record") = chol_dummy_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("gamma_record") = coef_dummy_record.bottomRows(num_iter - num_burn),
    Rcpp::Named("chol_record") = chol_factor_record.bottomRows(dim * (num_iter - num_burn)),
    Rcpp::Named("ols_coef") = coef_ols,
    Rcpp::Named("ols_cholesky") = chol_ols
  );
}
