#include <RcppEigen.h>
#include "bvharmisc.h"
#include "bvharprob.h"

// [[Rcpp::depends(RcppEigen)]]

//' Semiautomatic Approach to Select Coefficient Mixture Hyperparameters
//' 
//' @param spike_automatic
//' @param slab_automatic
//' 
//' 
//' @references 
//' George, E. I., & McCulloch, R. E. (1997). *APPROACHES FOR BAYESIAN VARIABLE SELECTION*. Statistica Sinica, 7(2), 339–373.
//' 
//' George, E. I., & McCulloch, R. E. (2012). *Variable Selection via Gibbs Sampling*. Journal of the American Statistical Association, 88(423), 881–889. doi:[10.1080/01621459.1993.10476353](https://www.tandfonline.com/doi/abs/10.1080/01621459.1993.10476353)
//' @noRd






//' Building Spike-and-slab SD Diagonal Matrix
//' 
//' In MCMC process of SSVS, compute diagonal matrix \eqn{D} or \eqn{D_j} defined by spike-and-slab sd.
//' 
//' @param spike_sd Standard deviance for Spike normal distribution
//' @param slab_sd Standard deviance for Slab normal distribution
//' @param mixture_dummy Indicator vector (0-1) corresponding to each element
//' @details
//' Let \eqn{(\gamma_1, \ldots, \gamma_k)^\intercal} be dummy variables restricting coefficients vector.
//' Then
//' \deqn{
//'   h_i = \begin{cases}
//'     \tau_{0i} & \text{if } \gamma_i = 0 \\
//'     \tau_{1i} & \text{if } \gamma_i = 1
//'   \end{cases}
//' }
//' In turn, \eqn{D = diag(h_1, \ldots, h_k)}.
//' Let \eqn{\omega_j = (\omega_{1j}, \ldots, \omega_{j - 1, j})^\intercal} be dummy variables restricting covariance matrix.
//' Then
//' \deqn{
//'   h_{ij} = \begin{cases}
//'     \kappa_{0ij} & \text{if } \omega_{ij} = 0 \\
//'     \kappa_{1ij} & \text{if } \omega_{ij} = 1
//'   \end{cases}
//' }
//' In turn, \eqn{D_j = diag(h_{1j}, \ldots, h_{j-1, j})}.
//' @references
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
//' 
//' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd build_ssvs_sd(Eigen::VectorXd spike_sd,
                              Eigen::VectorXd slab_sd,
                              Eigen::VectorXd mixture_dummy) {
  int num_param = spike_sd.size();
  // if (spike_sd.size() != num_coef_vec) {
  //   Rcpp::stop("The length of 'spike_sd' and 'slab_sd' should be the same.");
  // }
  // check outside of the function
  if (mixture_dummy.size() != num_param) {
    Rcpp::stop("The length of 'mixture_dummy' should be the same as the coefficient vector.");
  }
  Eigen::MatrixXd diag_sparse = Eigen::MatrixXd::Zero(num_param, num_param);
  for (int i = 0; i < num_param; i++) {
    diag_sparse(i, i) = mixture_dummy(i) * spike_sd(i) + (1 - mixture_dummy(i)) * slab_sd(i); // spike_sd if mixture_dummy=1 while slab_sd if mixture_dummy=0
  }
  return diag_sparse;
}

//' Generating Coefficient Vector in SSVS Gibbs Sampler
//' 
//' In MCMC process of SSVS, generate \eqn{\alpha_j} conditional posterior.
//' 
//' @param prior_mean The prior mean vector of the VAR coefficient vector
//' @param XtX The result of design matrix arithmetic \eqn{X_0^T X_0}
//' @param coef_ols OLS (MLE) estimator of the VAR coefficient
//' @param chol_factor Cholesky factor of variance matrix
//' @param inv_DRD Inverse of matrix product between \eqn{D} and correlation matrix \eqn{R}
//' @details
//' After sampling \eqn{\psi_{jj}, \psi_{ij}}, and \eqn{\omega_{ij}}, generate \eqn{\alpha} by
//' \deqn{\alpha \mid \gamma, \eta, \omega, \psi, Y_0 \sim N_{k^2 p} (\mu, \Delta)}
//' The dimension \eqn{k^2 p} is when non-constant term.
//' When there is constant term, it is \eqn{k (kp + 1)}.
//' Here,
//' \deqn{
//'   \mu = (
//'     (\Psi \Psi^\intercal) \otimes (X_0 X_0^\intercal) + (DRD)^{-1}
//'   )^{-1} (
//'     ( (\Psi \Psi^\intercal) \otimes (X_0 X_0^\intercal) ) \hat{\alpha}^{MLE} + (DRD)^{-1} \alpha_0
//'   )
//' }
//' where \eqn{\alpha_0} is the prior mean for \eqn{\alpha}.
//' In regression, MLE is the same as OLS.
//' \deqn{
//'   \Delta = ((\Psi \Psi^\intercal) \otimes (X_0 X_0^\intercal) + (DRD)^{-1})^{-1}
//' }
//' After this step, we move to generating Bernoulli \eqn{\gamma_j}.
//' @references
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
//' 
//' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_coef(Eigen::VectorXd prior_mean,
                          Eigen::MatrixXd XtX,
                          Eigen::VectorXd coef_ols,
                          Eigen::MatrixXd chol_factor,
                          Eigen::MatrixXd inv_DRD) {
  Eigen::MatrixXd prec_mat = chol_factor * chol_factor.transpose(); // Sigma^(-1) = chol * chol^T
  Eigen::MatrixXd lhs_kronecker = kronecker_eigen(prec_mat, XtX); // Sigma^(-1) otimes (X_0^T X_0)
  Eigen::MatrixXd normal_variance = (lhs_kronecker + inv_DRD).inverse(); // Delta
  Eigen::VectorXd normal_mean = normal_variance * (lhs_kronecker * coef_ols + inv_DRD * prior_mean); // mu
  return vectorize_eigen(sim_mgaussian(1, normal_mean, normal_variance));
}

//' Generating Dummy Vector for Parameters in SSVS Gibbs Sampler
//' 
//' In MCMC process of SSVS, generate latent \eqn{\gamma_j} or \eqn{\omegam_{ij}} conditional posterior.
//' 
//' @param param_obs Realized parameters vector
//' @param spike_sd Standard deviance for Spike normal distribution
//' @param slab_sd Standard deviance for Slab normal distribution
//' @param mixture_dummy Indicator vector (0-1) corresponding to each element
//' @param slab_weight Proportion of nonzero coefficients
//' @details
//' We draw \eqn{\omega_{ij}} and \eqn{\gamma_j} from Bernoulli distribution.
//' \deqn{
//'   \omega_{ij} \mid \eta_j, \psi_j, \alpha, \gamma, \omega_{j}^{(previous)} \mid Bernoulli(\frac{u_{ij1}}{u_{ij1} + u_{ij2}})
//' }
//' If \eqn{R_j = I_{j - 1}},
//' \deqn{
//'   u_{ij1} = \frac{1}{\kappa_{0ij} \exp(- \frac{\psi_{ij}^2}{2 \kappa_{0ij}^2}) \exp(- \frac{\psi_{1ij}^2}{2 \kappa_{0ij}^2}) q_{ij},
//'   u_{ij2} = \frac{1}{\kappa_{1ij} \exp(- \frac{\psi_{ij}^2}{2 \kappa_{1ij}^2}) \exp(- \frac{\psi_{0ij}^2}{2 \kappa_{1ij}^2}) (1 - q_{ij})
//' }
//' Otherwise, see George et al. (2008).
//' Also,
//' \deqn{
//'   \gamma_j \mid \alpha, \psi, \eta, \omega, Y_0 \sim Bernoulli(\frac{u_{i1}}{u_{j1} + u_{j2}})
//' }
//' Similarly, if \eqn{R = I_{k^2 p}},
//' \deqn{
//'   u_{j1} = \frac{1}{\tau_{0j}} \exp(- \frac{\alpha_j^2}{2 \tau_{0j}^2})p_i,
//'   u_{j2} = \frac{1}{\tau_{1j}} \exp(- \frac{\alpha_j^2}{2 \tau_{1j}^2})(1 - p_i)
//' }
//' @references
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
//' 
//' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_dummy(Eigen::VectorXd param_obs, 
                           Eigen::VectorXd spike_sd,
                           Eigen::VectorXd slab_sd,
                           Eigen::VectorXd mixture_dummy,
                           Eigen::VectorXd slab_weight) {
  double bernoulli_param_spike;
  double bernoulli_param_slab;
  int num_latent = mixture_dummy.size();
  Eigen::VectorXd latent_prop(num_latent); // latentj | Y0, -latentj ~ Bernoulli(u1 / (u1 + u2))
  for (int i = 0; i < num_latent; i ++) {
    bernoulli_param_spike = slab_weight(i) * exp(-pow(param_obs(i) / (2 * spike_sd(i)), 2.0)) / spike_sd(i);
    bernoulli_param_slab = (1.0 - slab_weight(i)) * exp(-pow(param_obs(i) / (2 * slab_sd(i)), 2.0)) / slab_sd(i);
    latent_prop(i) = binom_rand(1, bernoulli_param_spike / (bernoulli_param_spike + bernoulli_param_slab)); // qj-bar
  }
  return latent_prop;
}

//' Generating the Diagonal Component of Cholesky Factor in SSVS Gibbs Sampler
//' 
//' In MCMC process of SSVS, generate the diagonal component \eqn{\Psi} from variance matrix
//' 
//' @param sse_mat The result of \eqn{Z_0^T Z_0 = (Y_0 - X_0 \hat{A})^T (Y_0 - X_0 \hat{A})}
//' @param inv_DRD Inverse of matrix product between \eqn{D_j} and correlation matrix \eqn{R_j}
//' @param shape Gamma shape parameters for precision matrix
//' @param rate Gamma rate parameters for precision matrix
//' @param num_design The number of sample used, \eqn{n = T - p}
//' @details
//' Let SSE matrix be \eqn{S(\hat{A}) = (Y_0 - X_0 \hat{A})^\intercal (Y_0 - X_0 \hat{A}) \in \mathbb{R}^{k \times k}},
//' let \eqn{S_j} be the upper-left j x j block matrix of \eqn{S(\hat{A})},
//' and let \eqn{s_j = (s_{1j}, \ldots, s_{j - 1, j})^\intercal}.
//' For specified shape and rate of Gamma distribuion \eqn{a_j} and \eqn{b_j},
//' \deqn{
//'   \psi_{jj}^2 \mid \alpha, \gamma, \omega, Y_0 \sim \gamma(a_i + n / 2, B_i)
//' }
//' where
//' \deqn{
//'   B_i = \begin{cases}
//'     b_1 + s_{11} / 2 & \text{if } i = 1 \\
//'     b_i + (s_{ii} - s_i^\intercal ( S_{i - 1} + (D_i R_i D_i)^(-1) )^(-1) s_i) & \text{if } i = 2, \ldots, k
//'   \end{cases}
//' }
//' , and \eqn{D_i = diag(h_{1j}, \ldots, h_{i - 1, i}) \in \mathbb{R}^{(j - 1) \times (j - 1)}}
//' is the one made by upper diagonal element of \eqn{\Psi} matrix.
//' @references
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
//' 
//' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_chol_diag(Eigen::MatrixXd sse_mat,
                               Eigen::MatrixXd inv_DRD,
                               Eigen::VectorXd shape,
                               Eigen::VectorXd rate,
                               int num_design) {
  int dim = sse_mat.cols(); // m
  Eigen::VectorXd res(dim);
  // double chol_diag;
  // if (col_index == 0) {
  //   rate += ZtZ(0, 0) / 2; // b[1] + v11
  //   chol_diag = gamma_rand(shape, 1 / rate); // psi[jj]^2 ~ Gamma
  //   return sqrt(chol_diag);
  // }
  // Eigen::MatrixXd z_j = sse_mat.block(0, 0, col_index, col_index); // V(j - 1)
  // Eigen::MatrixXd diag_block = diag_sparse.block(0, 0, col_index, col_index); // Fj
  // Eigen::MatrixXd large_mat(col_index, col_index);
  // large_mat = z_j.transpose() * (z_j + (diag_block * diag_block).inverse()).inverse() * z_j;
  // rate += (ZtZ(col_index, col_index) - large_mat(col_index, col_index)) / 2;
  // chol_diag = gamma_rand(shape, 1 / rate); // psi[jj]^2 ~ Gamma
  Eigen::MatrixXd sse_block(dim - 1, dim - 1);
  Eigen::VectorXd sse_colvec(dim - 1);
  shape.array() += num_design / 2;
  rate[0] += sse_mat(0, 0) / 2;
  for (int i = 0; i < dim; i++) {
    sse_block = sse_mat.block(0, 0, i - 1, i - 1); // upper left (i - 1) x (i - 1)
    sse_colvec = sse_mat.block(0, i, i - 1, 1); // (s1i, ..., si-1,i)
    rate[i] += (sse_mat(i, i) - sse_colvec.transpose() * (sse_block + inv_DRD).inverse() * sse_colvec) / 2;
    res[i] = gamma_rand(shape[i], 1 / rate[i]); // psi[jj]^2 ~ Gamma(shape, rate)
  }
  return res.sqrt();
}

//' Generating the Off-Diagonal Component of Cholesky Factor in SSVS Gibbs Sampler
//' 
//' In MCMC process of SSVS, generate the off-diagonal component \eqn{\Psi} of variance matrix
//' 
//' @param col_index Choose the column index of cholesky factor
//' @param sse_mat The result of \eqn{Z_0^T Z_0 = (Y_0 - X_0 \hat{A})^T (Y_0 - X_0 \hat{A})}
//' @param chol_diag Diagonal element of the cholesky factor
//' @param inv_DRD Inverse of matrix product between \eqn{D_j} and correlation matrix \eqn{R_j}
//' @details
//' After drawing \eqn{\psi_{jj}}, generate upper elements by
//' \deqn{
//'   \eta_j \mid \alpha, \gamma, \omega, \psi, Y_0 \sim N_{j - 1} (\mu_j, \Delta_j)
//' }
//' where
//' \deqn{
//'   \mu_j = -\psi_{jj} (S_{j - 1} (D_j R_j D_j)^{-1})^{-1} s_j,
//'   \Delta_j = (S_{j - 1} (D_j R_j D_j)^{-1})^{-1}
//' }
//' @references
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
//' 
//' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_chol_off(int col_index, 
                              Eigen::MatrixXd sse_mat, 
                              Eigen::VectorXd chol_diag, 
                              Eigen::MatrixXd inv_DRD) {
  int dim = sse_mat.cols(); // m
  if (chol_diag.size() != dim) {
    Rcpp::stop("Wrong length of 'chol_diag'.");
  }
  Eigen::MatrixXd normal_variance = (sse_mat.block(0, 0, col_index - 1, col_index - 1) + inv_DRD).inverse();
  Eigen::VectorXd sse_colvec = sse_mat.block(0, col_index, col_index - 1, 1);
  Eigen::VectorXd normal_mean = -chol_diag(col_index, col_index) * normal_variance * sse_colvec;
  return vectorize_eigen(sim_mgaussian(1, normal_mean, normal_variance));
}

//' Filling Cholesky Factor Upper Triangular Matrix
//' 
//' Build a cholesky factor matrix \eqn{\Psi} (upper triangular)
//' using diagonal component vector and off-diaognal component vector
//' 
//' @param diag_vec Diagonal components
//' @param off_diagvec Off-diagonal components
//' @details
//' Consider \eqn{\Sigma_e^{-1} = \Psi \Psi^\intercal} where upper triangular \eqn{\Psi = [\psi_{ij}]}.
//' Column vector for upper off-diagonal element is denoted by
//' \deqn{
//'   \eta_j = (\psi_{12}, \ldots, \psi_{j-1, j})
//' }
//' for \eqn{j = 2, \ldots, k}.
//' @references
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
//' 
//' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd build_chol(Eigen::VectorXd diag_vec, Eigen::VectorXd off_diagvec) {
  int dim = diag_vec.size();
  if (off_diagvec.size() != dim * (dim - 1) / 2) {
    Rcpp::stop("Wrong length of 'off_diagvec'.");
  }
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim, dim);
  res.diagonal() = diag_vec; // psi_ii
  int id = 0; // length of eta = m(m-1)/2
  // should assign eta (off_diagvec) column-wise
  for (int j = 1; j < dim; j++) {
    // res(j, j) = diag_vec(j);
    for (int i = 0; i < j; i++) {
      res(i, j) = off_diagvec[id + i];
    }
    id += j;
  }
  return res;
}

//' BVAR(p) Point Estimates based on SSVS Prior
//' 
//' Compute MCMC for SSVS prior
//' 
//' @param num_iter Number of iteration for MCMC
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param init_coef Initial k x m coefficient matrix.
//' @param init_coef_sparse Indicator vector (0-1) corresponding to each coefficient vector
//' @param init_cov Initial m x m variance matrix.
//' @param init_cov_sparse Indicator vector (0-1) corresponding to each covariance component
//' @param coef_spike Standard deviance for Spike normal distribution
//' @param coef_slab Standard deviance for Slab normal distribution
//' @param coef_slab_weight Bernoulli parameter for coefficients sparsity proportion
//' @param shape Gamma shape parameters for precision matrix
//' @param rate Gamma rate parameters for precision matrix
//' @param chol_spike Standard deviance for cholesky factor Spike normal distribution
//' @param chol_slab Standard deviance for cholesky factor Slab normal distribution
//' @param chol_slab_weight Bernoulli parameter for cholesky factor sparsity proportion
//' @details
//' Gibbs sampling:
//' 1. Diagonal elements of \eqn{\Psi}
//' 2. Off-diagonal elements of \eqn{\Psi}
//' 3. Dummy vector for cholesky factor \eqn{\psi_{ij}}
//' 4. Coefficient vector \eqn{\alpha}
//' 5. Dummy vector for coefficient vector \eqn{\gamma_j}
//' @references
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
//' 
//' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_ssvs(int num_iter,
                              Eigen::MatrixXd x, 
                              Eigen::MatrixXd y, 
                              Eigen::MatrixXd init_coef,
                              Eigen::MatrixXd init_coef_sparse,
                              Eigen::Map<Eigen::MatrixXd> init_cov,
                              Eigen::MatrixXd init_cov_sparse,
                              Eigen::VectorXd coef_spike,
                              Eigen::VectorXd coef_slab,
                              Eigen::VectorXd coef_slab_weight,
                              Eigen::VectorXd shape,
                              Eigen::VectorXd rate,
                              Eigen::VectorXd chol_spike,
                              Eigen::VectorXd chol_slab,
                              Eigen::VectorXd chol_slab_weight) {
  int dim = y.cols(); // dim = k
  int dim_design = x.cols(); // kp(+1)
  int num_design = y.rows(); // n = T - p
  // record-------------------------------------------------------
  
  
  
  
  
  
  Eigen::MatrixXd coef_mat(dim_design, dim); // A: kp(+1) x k
  Eigen::MatrixXd XtX = x.transpose() * x; // X_0^T X_0: k x k
  coef_mat = XtX.inverse() * x.transpose() * y; // Ahat
  Eigen::VectorXd coef_lse = vectorize_eigen(coef_mat); // alphahat = vec(Ahat)
  // initial for coefficient-----------------------------------------------------------
  if (init_coef.rows() != dim_design) {
    Rcpp::stop("Invalid number of rows of 'init_coef'.");
  }
  if (init_coef.cols() != dim) {
    Rcpp::stop("Invalid number of colums of 'init_coef'.");
  }
  if (init_coef_sparse.rows() != dim_design) {
    Rcpp::stop("Invalid number of rows of 'init_coef_sparse'.");
  }
  if (init_coef_sparse.cols() != dim) {
    Rcpp::stop("Invalid number of columns of 'init_coef_sparse'.");
  }
  Eigen::MatrixXd resid_mat = y - x * init_coef; // Z = Y0 - X0 B
  Eigen::MatrixXd ZtZ = resid_mat.transpose() * resid_mat; // (Y0 - X0 B)^T (Y0 - X0 B)
  Eigen::VectorXd init_coef_vec = vectorize_eigen(init_coef); // initial for vec(A)
  Eigen::VectorXd init_coef_sparse_vec = vectorize_eigen(init_coef_sparse); // initial for gamma[j]
  Eigen::MatrixXd diag_coef_sparse(init_coef_sparse_vec.size(), init_coef_sparse_vec.size()); // D: mk x mk
  Eigen::VectorXd init_cov_sparse_vec = vectorize_eigen(init_cov_sparse); // initial for w[ij]
  Eigen::MatrixXd diag_cov_sparse(dim - 1, dim - 1); // large matrix to assign diag_sparse of cov
  // initial for covariance matrix: Sigma_e^(-1) = Psi * Psi^T--------------------------
  Eigen::LLT<Eigen::MatrixXd> lltOfcov(init_cov); // Sigma = LL^T
  if (lltOfcov.info() == Eigen::NumericalIssue) {
    Rcpp::stop("'init_cov' should be positive definite matrix."); // Sigma is pd
  }
  Eigen::MatrixXd chol_upper = lltOfcov.matrixL().transpose(); // Psi = (L^(-1))^T: upper
  // Trace for MCMC---------------------------------------------------------------------
  Eigen::MatrixXd coef_trace(num_iter + 1, dim * dim_design); // record alpha in MCMC
  coef_trace.row(0) = init_coef_vec;
  Eigen::MatrixXd coef_prop_trace(num_iter + 1, dim * dim_design); // record gamma[j] in MCMC
  coef_prop_trace.row(0) = init_coef_sparse_vec;
  Eigen::MatrixXd diag_trace(num_iter + 1, dim); // record diagonal component in MCMC
  for (int i = 0; i < dim; i++) {
    diag_trace(0, i) = chol_upper(i, i);
  }
  Eigen::MatrixXd off_trace(num_iter + 1, dim * (dim - 1) / 2); // record off-diagonal component in MCMC
  int id = 0;
  for (int j = 1; j < dim; j++) { // from 2nd column
    for (int i = 0; i < j; i++) { // upper triangular: exclude diagonal
      off_trace(0, id) = chol_upper(i, j);
      id++;
    }
  }
  Eigen::MatrixXd cov_prop_trace(num_iter + 1, dim * (dim - 1) / 2); // record w[ij] in MCMC
  cov_prop_trace.row(0) = init_cov_sparse_vec;
  // Gibbs sampler iteration------------------------------------------------------------
  for (int t = 0; t < num_iter; t++) {
    // 1. Coefficient
    diag_coef_sparse = ssvs_coef_prop(coef_spike, coef_slab, init_coef_sparse_vec); // D
    init_coef_vec = ssvs_coef(XtX, coef_lse, chol_upper, diag_coef_sparse); // Normal
    coef_trace.row(t + 1) = init_coef_vec; // record
    // 2. Sparsity proportion of coefficient
    init_coef_sparse_vec = ssvs_coef_latent(init_coef_vec, coef_spike, coef_slab, coef_slab_weight); // Bernoulli
    coef_prop_trace.row(t + 1) = init_coef_sparse_vec; // record
    // 3. Diagonal components of cholesky factor
    chol_upper(0, 0) = ssvs_cov_diag(0, ZtZ, diag_cov_sparse, shape, rate); // sqrt of Gamma
    diag_trace(t + 1, 0) = chol_upper(0, 0); // record
    for (int i = 1; i < dim; i++) {
      diag_cov_sparse.block(0, 0, i, i) = ssvs_cov_prop(i, chol_spike, chol_slab, init_cov_sparse_vec); // Fj
      chol_upper(i, i) = ssvs_cov_diag(i, ZtZ, diag_cov_sparse, shape, rate); // sqrt of Gamma
      diag_trace(t + 1, i) = chol_upper(i, i); // record
    }
    // 4. Off-diagonal components of cholesky factor
    id = 0;
    for (int j = 1; j < dim; j++) {
      chol_upper.block(0, j, j - 1, 1) = ssvs_cov_off(j, ZtZ, chol_upper, diag_cov_sparse);
      off_trace.block(num_iter, id, j - 1, 1) = chol_upper.block(0, j, j - 1, 1);
      id += j - 1; // next column index
    }
    // 5. Sparsity proportion of variance matrix
    init_cov_sparse_vec = ssvs_cov_latent(chol_upper, chol_spike, chol_slab, chol_slab_weight); // Bernoulli
    cov_prop_trace.row(t + 1) = init_cov_sparse_vec; // record
  }
  return Rcpp::List::create(
    Rcpp::Named("coef") = init_coef_vec,
    Rcpp::Named("cov") = chol_upper,
    Rcpp::Named("coef_prop") = init_coef_sparse_vec,
    Rcpp::Named("cov_prop") = init_cov_sparse_vec,
    Rcpp::Named("coef_record") = coef_trace,
    Rcpp::Named("diag_record") = diag_trace,
    Rcpp::Named("off_record") = off_trace,
    Rcpp::Named("coef_prop_record") = coef_prop_trace,
    Rcpp::Named("cov_prop_record") = cov_prop_trace
  );
}
