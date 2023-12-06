#include <RcppEigen.h>
#include "bvharmisc.h"
#include "bvharprob.h"

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

//' Building Lower Triangular Matrix
//' 
//' In MCMC, this function builds \eqn{L} given \eqn{a} vector.
//' 
//' @param dim Dimension (dim x dim) of L
//' @param lower_vec Vector a
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd build_inv_lower(int dim, Eigen::VectorXd lower_vec) {
  Eigen::MatrixXd res = Eigen::MatrixXd::Identity(dim, dim);
  int id = 0;
  for (int i = 1; i < dim; i++) {
    res.col(i - 1).segment(i, dim - i) = lower_vec.segment(id, dim - i);
    id += dim - i;
  }
  return res;
}

//' Generating the Lower diagonal of LDLT Factor or Coefficients Vector
//' 
//' @param x Design matrix in SUR or stacked E_t
//' @param y Response vector in SUR or stacked e_t
//' @param prior_mean Prior mean vector
//' @param prior_prec Prior precision matrix
//' @param innov_prec Stacked precision matrix of innovation
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd varsv_regression(Eigen::MatrixXd x, Eigen::VectorXd y,
                                 Eigen::VectorXd prior_mean, Eigen::MatrixXd prior_prec,
                                 Eigen::MatrixXd innov_prec) {
  Eigen::MatrixXd post_prec = (prior_prec + x.transpose() * innov_prec * x).llt().solve(Eigen::MatrixXd::Identity(x.cols(), x.cols()));
  return vectorize_eigen(sim_mgaussian_chol(1, post_prec * (prior_prec * prior_mean + x.transpose() * innov_prec * y), post_prec));
}

//' Generating log-volatilities in MCMC
//' 
//' In MCMC, this function samples log-volatilities \eqn{h_{it}} vector using auxiliary mixture sampling
//' 
//' @param sv_vec log-volatilities vector
//' @param init_sv Initial log-volatility
//' @param sv_sig Variance of log-volatilities
//' @param latent_vec Auxiliary residual vector
//' @param nthreads Number of threads for openmp
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd varsv_ht(Eigen::VectorXd pj,
                         Eigen::VectorXd muj, Eigen::VectorXd sigj,
                         Eigen::VectorXd sv_vec, double init_sv,
                         double sv_sig, Eigen::VectorXd latent_vec, int nthreads) {
  int num_design = sv_vec.size(); // h_i1, ..., h_in for i = 1, .., k
  Eigen::VectorXd sdj = sigj.cwiseSqrt();
  Eigen::VectorXi binom_latent(num_design);
  Eigen::VectorXd ds(num_design); // (mu_st - 1.2704)
  Eigen::MatrixXd inv_sig_s = Eigen::MatrixXd::Zero(num_design, num_design); // diag(1 / sig_st^2)
  Eigen::VectorXd inv_method(num_design); // inverse transform method
  Eigen::MatrixXd mixture_pdf(num_design, 7);
  Eigen::MatrixXd mixture_cumsum = Eigen::MatrixXd::Zero(num_design, 7);
  for (int i = 0; i < num_design; i++) {
    inv_method[i] = unif_rand(0, 1);
  }
  for (int i = 0; i < 7; i++) {
    mixture_pdf.col(i) = (-((latent_vec.array() - sv_vec.array() - muj[i]).array() / sdj[i]).array().square() / 2).exp() * pj[i] / (sdj[i] * sqrt(2 * M_PI));
  }
  Eigen::VectorXd ct = mixture_pdf.rowwise().sum().array();
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < num_design; i++) {
    mixture_pdf.row(i).array() = mixture_pdf.row(i).array() / ct[i];
  }
#else
  for (int i = 0; i < num_design; i++) {
    mixture_pdf.row(i).array() = mixture_pdf.row(i).array() / ct[i];
  }
#endif
  for (int i = 0; i < 7; i++) {
    mixture_cumsum.block(0, i, num_design, 7 - i) += mixture_pdf.col(i).rowwise().replicate(7 - i);
  }
  binom_latent.array() = 7 - (inv_method.rowwise().replicate(7).array() < mixture_cumsum.array()).cast<int>().rowwise().sum().array(); // 0 to 6 for indexing
  Eigen::MatrixXd diff_mat = Eigen::MatrixXd::Identity(num_design, num_design);
  for (int i = 0; i < num_design - 1; i++) {
    ds[i] = muj[binom_latent[i]];
    inv_sig_s(i, i) = 1 / sigj[binom_latent[i]];
    diff_mat(i + 1, i) = -1;
  }
  ds[num_design - 1] = muj[binom_latent[num_design - 1]];
  inv_sig_s(num_design - 1, num_design - 1) = 1 / sigj[binom_latent[num_design - 1]];
  Eigen::MatrixXd HtH = diff_mat.transpose() * diff_mat;
  Eigen::MatrixXd post_prec = (HtH / sv_sig + inv_sig_s).llt().solve(Eigen::MatrixXd::Identity(num_design, num_design));
  return vectorize_eigen(sim_mgaussian_chol(1, post_prec * (HtH * init_sv * Eigen::VectorXd::Ones(num_design) / sv_sig + inv_sig_s * (latent_vec - ds)), post_prec));
}

//' Generating sig_h in MCMC
//' 
//' In MCMC, this function samples \eqn{\sigma_h^2} in VAR-SV.
//' 
//' @param shp Prior shape of sigma
//' @param scl Prior scale of sigma
//' @param init_sv Initial log volatility
//' @param h1 Time-varying h1 matrix
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd varsv_sigh(Eigen::VectorXd shp, Eigen::VectorXd scl, Eigen::VectorXd init_sv, Eigen::MatrixXd h1) {
  int dim = init_sv.size();
  int num_design = h1.rows();
  Eigen::VectorXd res(dim);
  Eigen::MatrixXd h_slide(num_design, dim); // h_ij, j = 0, ..., n - 1
  h_slide.row(0) = init_sv;
  // h_slide.bottomRows(num_design - 1) = lvol_record.block(num_design * i, 0, num_design - 1, dim);
  h_slide.bottomRows(num_design - 1) = h1.topRows(num_design - 1);
  for (int i = 0; i < dim; i++) {
    res[i] = 1 / gamma_rand(
      shp[i] + num_design / 2,
      1 / (scl[i] + (h1.array() - h_slide.array()).square().sum() / 2)
    );
  }
  return res;
}

//' Generating h0 in MCMC
//' 
//' In MCMC, this function samples h0 in VAR-SV.
//' 
//' @param prior_mean Prior mean vector of h0.
//' @param prior_prec Prior precision matrix of h0.
//' @param init_sv Initial log volatility
//' @param h1 h1
//' @param sv_sig Variance of log volatility
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd varsv_h0(Eigen::VectorXd prior_mean, Eigen::MatrixXd prior_prec,
                         Eigen::VectorXd init_sv, Eigen::VectorXd h1,
                         Eigen::VectorXd sv_sig) {
  int dim = init_sv.size();
  Eigen::MatrixXd post_h0_prec(dim, dim); // k_h0
  Eigen::MatrixXd h_diagprec = Eigen::MatrixXd::Zero(dim, dim); // diag(1 / sigma_h^2)
  h_diagprec.diagonal() = 1 / sv_sig.array();
  post_h0_prec = (prior_prec + h_diagprec).llt().solve(Eigen::MatrixXd::Identity(dim, dim));
  return vectorize_eigen(sim_mgaussian_chol(1, post_h0_prec * (prior_prec * prior_mean + h_diagprec * h1), post_h0_prec));
}
