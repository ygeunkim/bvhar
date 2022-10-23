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
//' @param prior_prec The prior precision matrix of the VAR coefficient vector
//' @param XtX The result of design matrix arithmetic \eqn{X_0^T X_0}
//' @param coef_ols OLS (MLE) estimator of the VAR coefficient
//' @param chol_factor Cholesky factor of variance matrix
//' @details
//' After sampling \eqn{\psi_{jj}, \psi_{ij}}, and \eqn{\omega_{ij}}, generate \eqn{\alpha} by
//' combining non-restricted constant term and potentially restricted coefficients vector term.
//' This process is done outside of the function and gives each prior mean and prior precision.
//' In turn,
//' \deqn{\alpha \mid \gamma, \eta, \omega, \psi, Y_0 \sim N_{k^2 p} (\mu, \Delta)}
//' The dimension \eqn{k^2 p} is when non-constant term.
//' When there is constant term, it is \eqn{k (kp + 1)}.
//' Here,
//' \deqn{
//'   \mu = (
//'     (\Psi \Psi^\intercal) \otimes (X_0 X_0^\intercal) + M^{-1}
//'   )^{-1} (
//'     ( (\Psi \Psi^\intercal) \otimes (X_0 X_0^\intercal) ) \hat{\alpha}^{MLE} + M^{-1} \alpha_0
//'   )
//' }
//' where \eqn{\alpha_0} is the prior mean for \eqn{\alpha}.
//' In regression, MLE is the same as OLS.
//' \deqn{
//'   \Delta = ((\Psi \Psi^\intercal) \otimes (X_0 X_0^\intercal) + M^{-1})^{-1}
//' }
//' After this step, we move to generating Bernoulli \eqn{\gamma_j}.
//' @references
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
//' 
//' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_coef(Eigen::VectorXd prior_mean,
                          Eigen::MatrixXd prior_prec,
                          Eigen::MatrixXd XtX,
                          Eigen::VectorXd coef_ols,
                          Eigen::MatrixXd chol_factor) {
  Eigen::MatrixXd prec_mat = chol_factor * chol_factor.transpose(); // Sigma^(-1) = chol * chol^T
  Eigen::MatrixXd lhs_kronecker = kronecker_eigen(prec_mat, XtX); // Sigma^(-1) otimes (X_0^T X_0)
  Eigen::MatrixXd normal_variance = (lhs_kronecker + prior_prec).inverse(); // Delta
  Eigen::VectorXd normal_mean = normal_variance * (lhs_kronecker * coef_ols + prior_prec * prior_mean); // mu
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
//'   \mu_j = -\psi_{jj} (S_{j - 1} + (D_j R_j D_j)^{-1})^{-1} s_j,
//'   \Delta_j = (S_{j - 1} + (D_j R_j D_j)^{-1})^{-1}
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
  Eigen::MatrixXd normal_variance = (sse_mat.block(0, 0, col_index, col_index) + inv_DRD).inverse();
  Eigen::VectorXd sse_colvec = sse_mat.block(0, col_index, col_index, 1);
  Eigen::VectorXd normal_mean = -chol_diag[col_index] * normal_variance * sse_colvec;
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
//' @param spike_automatic Tuning parameter (small number, e.g. .1) corresponding to standard deviation of spike normal distribution
//' @param slab_automatic Tuning parameter (large number, e.g. 10) corresponding to standard deviation of slab normal distribution
//' @param intercept_var Hyperparameter for constant term
//' @details
//' Data: \eqn{X_0}, \eqn{Y_0}
//' 
//' Input:
//' * VAR order p
//' * MCMC iteration number
//' * Weight of each slab: Bernoulli distribution parameters
//'     * \eqn{p_j}: of coefficients
//'     * \eqn{q_{ij}}: of cholesky factor
//' * Gamma distribution parameters for cholesky factor diagonal elements \eqn{\psi_{jj}}
//'     * \eqn{a_j}: shape
//'     * \eqn{b_j}: rate
//' * Correlation matrix of coefficient vector: \eqn{R = I_{k^2p}}
//' * Correlation matrix to restrict cholesky factor (of \eqn{\eta_j}): \eqn{R_j = I_{j - 1}}
//' * Tuning parameters for spike-and-slab sd semi-automatic approach
//'     * \eqn{c_0}: small value (0.1)
//'     * \eqn{c_1}: large value (10)
//' * Constant to reduce prior influence on constant term: \eqn{c}
//' 
//' Gibbs sampling:
//' 1. Initialize \eqn{\Psi}, \eqn{\omega}, \eqn{\alpha}, \eqn{\gamma}
//' 2. Iterate
//'     1. Diagonal elements of cholesky factor: \eqn{\psi^{(t)} \mid \alpha^{(t - 1)}, \gamma^{(t - 1)}, \omega^{(t - 1)}, Y_0}
//'     2. Off-diagonal elements of cholesky factor: \eqn{\eta^{(t)} \mid \psi^{(t)} \alpha^{(t - 1)}, \gamma^{(t - 1)}, \omega^{(t - 1)}, Y_0}
//'     3. Dummy vector for cholesky factor: \eqn{\omega^{(t)} \mid \eta^{(t)}, \psi^{(t)} \alpha^{(t - 1)}, \gamma^{(t - 1)}, \omega^{(t - 1)}, Y_0}
//'     4. Coefficient vector: \eqn{\alpha^{(t)} \mid \gamma^{(t - 1)}, \Sigma^{(t)}, \omega^{(t)}, Y_0}
//'     5. Dummy vector for coefficient vector: \eqn{\gamma^{(t)} \mid \alpha^{(t)}, \psi^{(t)}, \eta^{(t)}, \omega^{(t)}, Y_0}
//' @references
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
//' 
//' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_ssvs(int num_iter,
                              int num_burn,
                              Eigen::MatrixXd x, 
                              Eigen::MatrixXd y, 
                              Eigen::MatrixXd init_coef,
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
                              double spike_automatic,
                              double slab_automatic,
                              double intercept_var) {
  int dim = y.cols(); // dim = k
  int dim_design = x.cols(); // kp(+1)
  int num_design = y.rows(); // n = T - p
  int num_upperchol = init_chol_upper.size(); // number of upper cholesky = dim (dim - 1) / 2
  if (!(spike_automatic <= 0 && slab_automatic <= 0 && intercept_var && 0)) {
    Rcpp::stop("'spike_automatic', 'slab_automatic', and 'intercept_var' should be positive.");
  }
  if (!(init_chol_diag.size() == dim && shape.size() == dim && rate.size() == dim)) {
    Rcpp::stop("Size of 'init_chol_diag', 'shape', and 'rate' vector should be the same as the time series dimension.");
  }
  if (!(init_chol_upper.size() == dim * (dim - 1) / 2 && 
      init_chol_dummy.size() == init_chol_upper.size() &&
      chol_spike.size() == init_chol_upper.size() && 
      chol_slab.size() == init_chol_upper.size() &&
      chol_slab_weight.size() == init_chol_upper.size())) {
    Rcpp::stop("Size of 'init_chol_upper', 'init_chol_dummy', 'chol_spike', 'chol_slab', and 'chol_slab_weight' vector should be the same as dim * (dim - 1) / 2.");
  }
  // Initialize coefficients vector-------------------------------
  int num_coef = dim * dim_design; // dim * (kp(+1))
  int num_restrict = num_coef - dim; // number of restricted coefs: k^2p
  int num_non = num_coef - num_restrict; // number of unrestricted coefs (constant vector): k
  if (!(init_coef.cols() == dim && init_coef.rows() == dim_design)) {
    Rcpp::stop("Dimension of 'init_coef' should be (dim * p) x dim.");
  }
  if (!(init_coef_dummy.size() == num_restrict && 
      coef_spike.size() == num_restrict && 
      coef_slab.size() == num_restrict && 
      coef_slab_weight.size() == num_restrict)) {
    Rcpp::stop("Size of 'init_coef_dummy', 'coef_spike', 'coef_slab', and 'coef_slab_weight' vector should be the same as dim^2 * p.");
  }
  // Eigen::VectorXd coef_vec(num_coef);
  Eigen::VectorXd coef_vec = vectorize_eigen(init_coef);
  Eigen::VectorXd prior_mean = Eigen::VectorXd::Zero(num_coef); // zero vector as prior mean
  Eigen::MatrixXd prior_variance = Eigen::MatrixXd::Zero(num_coef, num_coef); // M: diagonal matrix = DRD or merge of cI and DRD
  Eigen::MatrixXd coef_corr = Eigen::MatrixXd::Identity(num_restrict, num_restrict); // R
  Eigen::MatrixXd coef_mixture_mat = Eigen::MatrixXd::Zero(num_restrict, num_restrict); // D
  Eigen::MatrixXd DRD = Eigen::MatrixXd::Zero(num_restrict, num_restrict); // DRD
  Eigen::MatrixXd XtX = x.transpose() * x; // X_0^T X_0: k x k
  Eigen::MatrixXd coef_ols = XtX.inverse() * x.transpose() * y;
  // Eigen::VectorXd coefvec_ols(num_coef);
  Eigen::VectorXd coefvec_ols = vectorize_eigen(coef_ols);
  // record-------------------------------------------------------
  int num_mcmc = num_iter + num_burn;
  Eigen::MatrixXd coef_record(num_restrict, num_mcmc);
  coef_record.row(0) = coef_vec;
  Eigen::MatrixXd coef_dummy_record(num_restrict, num_mcmc);
  coef_dummy_record.row(0) = init_coef_dummy;
  Eigen::MatrixXd chol_diag_record(dim, num_mcmc);
  chol_diag_record.row(0) = init_chol_diag;
  Eigen::MatrixXd chol_upper_record(num_upperchol, num_mcmc);
  chol_upper_record.row(0) = init_chol_upper;
  Eigen::MatrixXd chol_dummy_record(num_upperchol, num_mcmc);
  chol_dummy_record.row(0) = init_chol_dummy;
  // 3d array?
  // Some variables-----------------------------------------------
  // Eigen::MatrixXd fitted_mat = x * coef_ols;
  Eigen::MatrixXd sse_mat = (y - x * coef_ols).transpose() * (y - x * coef_ols);
  Eigen::MatrixXd chol_factor(dim, dim); // Psi = upper triangular matrix
  Eigen::MatrixXd sse_block = sse_mat.block(0, 0, dim - 1, dim - 1); // Sj = upper-left j x j submatrix of SSE
  Eigen::VectorXd sse_colvec = sse_mat.block(0, dim - 1, dim - 1, 1); // sj = (s1j, ..., s(j-1, j)) from SSE
  Eigen::MatrixXd chol_corr = Eigen::MatrixXd::Identity(dim - 1, dim - 1); // Rj = I(j - 1)
  Eigen::MatrixXd chol_mixture_mat(dim - 1, dim - 1); // Dj = diag(h1j, ..., h(j-1,j))
  // Eigen::MatrixXd chol_mixture_mat(num_upperchol, num_upperchol); // Dj = diag(h1j, ..., h(j-1,j))
  Eigen::MatrixXd chol_prior_prec = Eigen::MatrixXd::Zero(dim - 1, dim - 1); // DjRjDj^(-1)
  // Eigen::VectorXd posterior_shape(dim);
  // Eigen::VectorXd posterior_rate(dim);
  // Semi-automatic approach--------------------------------------
  // Eigen::VectorXd ols_sd = sse_mat.diagonal().sqrt() / (num_design - dim_design); // OLS sd
  // Eigen::VectorXd coef_spike = spike_automatic * ols_sd; // sd of coef spike normal distribution (tau0j = c0 sd(alphaj-hat))
  // Eigen::VectorXd coef_slab = slab_automatic * ols_sd; // sd of coef slab normal distribution (tau1j = c1 sd(alphaj-hat))
  // Outside of function? Preliminary MCMC -> get sd of alpha-hat
  // Start Gibbs sampling-----------------------------------------
  int id_diag = 0; // id for diagonal psi vector
  int id_upper = 0; // id for upper psi vector
  for (int i = 1; i < num_mcmc; i++) {
    // 1. Psi
    chol_mixture_mat.block(0, 0, i - 1, i - 1) = build_ssvs_sd(chol_spike.segment(id_diag, id_diag + i - 1), chol_slab.segment(id_diag, id_diag + i - 1), chol_dummy_record.row(i - 1));
    id_diag += i;
    if (i > 0) {
      chol_prior_prec.block(0, 0, i - 1, i - 1) = (chol_mixture_mat.block(0, 0, i - 1, i - 1) * 
        chol_corr.block(0, 0, i - 1, i - 1) * 
        chol_mixture_mat.block(0, 0, i - 1, i - 1)).inverse();
    }
    chol_diag_record.row(i) = ssvs_chol_diag(sse_mat, chol_prior_prec.block(0, 0, i - 1, i - 1), shape, rate, num_design);
    // 2. eta
    for (int j = 1; j < dim; j++) {
      // for each col_index (eta2 to etak)
      chol_upper_record.block(i, id_upper, 1, j) = ssvs_chol_off(j, sse_mat, chol_diag_record.row(i), chol_prior_prec.block(0, 0, i - 1, i - 1));
      id_upper += j;
    }
    id_upper = 0;
    chol_factor = build_chol(chol_diag_record.row(i), chol_upper_record.row(i));
    // 3. omega
    chol_dummy_record.row(i) = ssvs_dummy(chol_upper_record.row(i), chol_spike, chol_slab, chol_dummy_record.row(i - 1), chol_slab_weight);
    // 4. alpha
    coef_mixture_mat = build_ssvs_sd(coef_spike, coef_slab, coef_dummy_record.row(i - 1));
    DRD = coef_mixture_mat * coef_corr * coef_mixture_mat;
    if (num_non == dim) {
      // constant case
      for (int j = 0; j < dim; j++) {
        prior_variance.block(j * dim_design, j * dim_design, num_restrict / dim, num_restrict / dim) = 
          DRD.block(j, j, num_restrict / dim, num_restrict / dim);
        prior_variance(j * dim_design + num_restrict / dim, j * dim_design + num_restrict / dim) = intercept_var;
      }
    } else if (num_non == -dim) {
      // no constant term
      prior_variance = DRD;
    }
    coef_record.row(i) = ssvs_coef(prior_mean, prior_variance.inverse(), XtX, coefvec_ols, chol_factor);
    // 5. gamma
    chol_dummy_record.row(i) = ssvs_dummy(coef_record.row(i), coef_spike, coef_slab, coef_dummy_record.row(i - 1), coef_slab_weight);
  }
  return Rcpp::List::create(
    Rcpp::Named("alpha_record") = coef_record.block(0, num_burn - 1, num_restrict, num_mcmc),
    Rcpp::Named("psi_ij_record") = chol_upper_record.block(0, num_burn - 1, num_upperchol, num_mcmc),
    Rcpp::Named("psi_jj_record") = chol_diag_record.block(0, num_burn - 1, dim, num_mcmc),
    Rcpp::Named("omega_ij_record") = chol_dummy_record.block(0, num_burn - 1, num_upperchol, num_mcmc),
    Rcpp::Named("tau_record") = coef_dummy_record.block(0, num_burn - 1, num_restrict, num_mcmc),
    Rcpp::Named("psi") = chol_factor,
    Rcpp::Named("sse") = sse_mat,
    Rcpp::Named("coef_vec") = coefvec_ols,
    Rcpp::Named("coefficients") = coef_ols
  );
}
