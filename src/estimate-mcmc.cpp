#ifdef _OPENMP
#include <omp.h>
#endif
#include <RcppEigen.h>
#include "bvharmisc.h"
#include "bvharprob.h"

// [[Rcpp::depends(RcppEigen)]]

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
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(num_param, num_param);
  // double sd_val;
  for (int i = 0; i < num_param; i++) {
    // spike_sd if mixture_dummy = 0 while slab_sd if mixture_dummy = 1
    res(i, i) = (1.0 - mixture_dummy[i]) * spike_sd[i] + mixture_dummy[i] * slab_sd[i];
    // sd_val = mixture_dummy[i] * spike_sd[i] + (1.0 - mixture_dummy[i]) * slab_sd[i];
    // res(i, i) = pow(1 / sd_val, 2.0);
  }
  return res;
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
  return vectorize_eigen(sim_mgaussian_chol(1, normal_mean, normal_variance));
}

//' Generating Dummy Vector for Parameters in SSVS Gibbs Sampler
//' 
//' In MCMC process of SSVS, generate latent \eqn{\gamma_j} or \eqn{\omega_{ij}} conditional posterior.
//' 
//' @param param_obs Realized parameters vector
//' @param spike_sd Standard deviance for Spike normal distribution
//' @param slab_sd Standard deviance for Slab normal distribution
//' @param slab_weight Proportion of nonzero coefficients
//' @details
//' We draw \eqn{\omega_{ij}} and \eqn{\gamma_j} from Bernoulli distribution.
//' \deqn{
//'   \omega_{ij} \mid \eta_j, \psi_j, \alpha, \gamma, \omega_{j}^{(previous)} \sim Bernoulli(\frac{u_{ij1}}{u_{ij1} + u_{ij2}})
//' }
//' If \eqn{R_j = I_{j - 1}},
//' \deqn{
//'   u_{ij1} = \frac{1}{\kappa_{1ij} \exp(- \frac{\psi_{ij}^2}{2 \kappa_{1ij}^2}) q_{ij},
//'   u_{ij2} = \frac{1}{\kappa_{0ij} \exp(- \frac{\psi_{ij}^2}{2 \kappa_{0ij}^2}) (1 - q_{ij})
//' }
//' Otherwise, see George et al. (2008).
//' Also,
//' \deqn{
//'   \gamma_j \mid \alpha, \psi, \eta, \omega, Y_0 \sim Bernoulli(\frac{u_{i1}}{u_{j1} + u_{j2}})
//' }
//' Similarly, if \eqn{R = I_{k^2 p}},
//' \deqn{
//'   u_{j1} = \frac{1}{\tau_{1j}} \exp(- \frac{\alpha_j^2}{2 \tau_{1j}^2})p_i,
//'   u_{j2} = \frac{1}{\tau_{0j}} \exp(- \frac{\alpha_j^2}{2 \tau_{0j}^2})(1 - p_i)
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
                           Eigen::VectorXd slab_weight) {
  double bernoulli_param_spike;
  double bernoulli_param_slab;
  int num_latent = slab_weight.size();
  Eigen::VectorXd res(num_latent); // latentj | Y0, -latentj ~ Bernoulli(u1 / (u1 + u2))
  for (int i = 0; i < num_latent; i++) {
    bernoulli_param_slab = slab_weight[i] * exp(- pow(param_obs[i], 2.0) / (2 * pow(slab_sd[i], 2.0)) ) / slab_sd[i];
    bernoulli_param_spike = (1.0 - slab_weight[i]) * exp(- pow(param_obs[i], 2.0) / (2 * pow(spike_sd[i], 2.0)) ) / spike_sd[i];
    res[i] = binom_rand(1.0, bernoulli_param_slab / (bernoulli_param_slab + bernoulli_param_spike)); // qj-bar
  }
  return res;
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
  int dim = sse_mat.cols();
  Eigen::VectorXd res(dim);
  Eigen::VectorXd sse_colvec(dim - 1); // sj = (s1j, ..., s(j-1, j)) from SSE
  // shape.array() += num_design / 2.f;
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
//' In MCMC process of SSVS, generate the off-diagonal component \eqn{\Psi} of variance matrix
//' 
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
    res.segment(block_id, j) = vectorize_eigen(sim_mgaussian_chol(1, normal_mean, normal_variance));
    block_id += j;
  }
  return res;
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
  for (int j = 1; j < dim; j++) {
    for (int i = 0; i < j; i++) {
      // should assign eta (off_diagvec) column-wise
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
//' @param intercept_var Hyperparameter for constant term
//' @param chain The number of MCMC chains.
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
  int num_upperchol = init_chol_upper.size(); // number of upper cholesky = dim (dim - 1) / 2
  // Initialize coefficients vector-------------------------------
  int num_coef = dim * dim_design; // dim^2 p + dim vs dim^2 p (if no constant)
  int num_restrict = num_coef - dim; // number of restricted coefs: dim^2 p vs dim^2 p - dim (if no constant)
  int num_non = num_coef - num_restrict; // number of unrestricted coefs (constant vector): dim vs -dim (if no constant)
  // Eigen::VectorXd coef_vec = vectorize_eigen(init_coef);
  Eigen::VectorXd prior_mean = Eigen::VectorXd::Zero(num_coef); // zero vector as prior mean
  Eigen::MatrixXd prior_variance = Eigen::MatrixXd::Zero(num_coef, num_coef); // M: diagonal matrix = DRD or merge of cI_dim and DRD
  // Eigen::MatrixXd prior_precision = Eigen::MatrixXd::Zero(num_coef, num_coef);
  Eigen::MatrixXd coef_mixture_mat = Eigen::MatrixXd::Zero(num_restrict, num_restrict); // D
  Eigen::MatrixXd DRD = Eigen::MatrixXd::Zero(num_restrict, num_restrict); // DRD
  // Eigen::MatrixXd DRD_inv = Eigen::MatrixXd::Zero(num_restrict, num_restrict);
  Eigen::MatrixXd XtX = x.transpose() * x; // X_0^T X_0: k x k
  Eigen::MatrixXd coef_ols = XtX.inverse() * x.transpose() * y;
  Eigen::VectorXd coefvec_ols = vectorize_eigen(coef_ols);
  // record-------------------------------------------------------
  int num_mcmc = num_iter + num_burn;
  Eigen::MatrixXd coef_record(num_mcmc, num_coef * chain);
  coef_record.row(0) = init_coef;
  Eigen::MatrixXd coef_dummy_record(num_mcmc, num_restrict * chain);
  coef_dummy_record.row(0) = init_coef_dummy;
  Eigen::MatrixXd chol_diag_record(num_mcmc, dim * chain);
  chol_diag_record.row(0) = init_chol_diag;
  Eigen::MatrixXd chol_upper_record(num_mcmc, num_upperchol * chain);
  chol_upper_record.row(0) = init_chol_upper;
  Eigen::MatrixXd chol_dummy_record(num_mcmc, num_upperchol * chain);
  chol_dummy_record.row(0) = init_chol_dummy;
  Eigen::MatrixXd chol_factor_record(dim * num_mcmc, dim * chain); // 3d matrix alternative
  // Some variables-----------------------------------------------
  Eigen::MatrixXd sse_mat = (y - x * coef_ols).transpose() * (y - x * coef_ols);
  // Eigen::MatrixXd chol_factor(dim, dim); // Psi = upper triangular matrix
  Eigen::MatrixXd chol_mixture_mat(num_upperchol, num_upperchol); // Dj = diag(h1j, ..., h(j-1,j))
  Eigen::MatrixXd chol_prior_prec = Eigen::MatrixXd::Zero(num_upperchol, num_upperchol); // DjRjDj^(-1)
  // Start Gibbs sampling-----------------------------------------
  #ifdef _OPENMP
  #pragma \
  omp \
    parallel \
    for \
      num_threads(chain) \
      shared(prior_mean, XtX, coefvec_ols, sse_mat, dim, dim_design, num_restrict, num_non, num_design, num_upperchol,
             coef_spike, coef_slab, coef_slab_weight, shape, rate, chol_spike, chol_slab, chol_slab_weight, intercept_var)
  for (int b = 0; b < chain; b++) {
    for (int i = 1; i < num_mcmc; i++) {
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
      chol_dummy_record.block(i, b * num_upperchol, 1, num_upperchol) = ssvs_dummy(chol_upper_record.block(i, b * num_upperchol, 1, num_upperchol), chol_spike, chol_slab, chol_slab_weight);
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
      // 5. gamma-------------------------
      coef_dummy_record.block(i, b * num_restrict, 1, num_restrict) = ssvs_dummy(coef_record.block(i, b * num_coef, 1, num_coef).head(num_restrict), coef_spike, coef_slab, coef_slab_weight);
    }
  }
  #else
  for (int i = 1; i < num_mcmc; i++) {
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
    // chol_factor = build_chol(chol_diag_record.row(i), chol_upper_record.row(i));
    // chol_factor_record.block(i * dim, 0, dim, dim) = chol_factor;
    chol_factor_record.block(i * dim, 0, dim, dim) = build_chol(chol_diag_record.row(i), chol_upper_record.row(i));
    // 3. omega--------------------------
    chol_dummy_record.row(i) = ssvs_dummy(chol_upper_record.row(i), chol_spike, chol_slab, chol_slab_weight);
    // 4. alpha--------------------------
    coef_mixture_mat = build_ssvs_sd(coef_spike, coef_slab, coef_dummy_record.row(i - 1));
    DRD = coef_mixture_mat * coef_mixture_mat;
    // DRD_inv = build_ssvs_sd(coef_spike, coef_slab, coef_dummy_record.row(i - 1));
    if (num_non == dim) {
      // constant case
      for (int j = 0; j < dim; j++) {
        prior_variance.block(j * dim_design, j * dim_design, num_restrict / dim, num_restrict / dim) =
          DRD.block(j * num_restrict / dim, j * num_restrict / dim, num_restrict / dim, num_restrict / dim); // kp x kp
        prior_variance(j * dim_design + num_restrict / dim, j * dim_design + num_restrict / dim) = intercept_var;
        // prior_precision.block(j * dim_design, j * dim_design, num_restrict / dim, num_restrict / dim) =
        //   DRD_inv.block(j, j, num_restrict / dim, num_restrict / dim);
        // prior_precision(j * dim_design + num_restrict / dim, j * dim_design + num_restrict / dim) = 1 / intercept_var;
      }
    } else if (num_non == -dim) {
      // no constant term
      prior_variance = DRD;
      // prior_precision = DRD_inv;
    }
    coef_record.row(i) = ssvs_coef(
      prior_mean, 
      prior_variance.inverse(), 
      XtX, 
      coefvec_ols, 
      chol_factor_record.block(i * dim, 0, dim, dim)
    );
    // coef_record.row(i) = ssvs_coef(prior_mean, prior_precision, XtX, coefvec_ols, chol_factor);
    // 5. gamma-------------------------
    coef_dummy_record.row(i) = ssvs_dummy(coef_record.row(i).head(num_restrict), coef_spike, coef_slab, coef_slab_weight); // exclude constant term
  }
  #endif
  return Rcpp::List::create(
    Rcpp::Named("alpha_record") = coef_record.bottomRows(num_iter),
    Rcpp::Named("psi_ij_record") = chol_upper_record.bottomRows(num_iter),
    Rcpp::Named("psi_jj_record") = chol_diag_record.bottomRows(num_iter),
    Rcpp::Named("omega_ij_record") = chol_dummy_record.bottomRows(num_iter),
    Rcpp::Named("tau_record") = coef_dummy_record.bottomRows(num_iter),
    Rcpp::Named("psi_record") = chol_factor_record,
    Rcpp::Named("sse") = sse_mat,
    Rcpp::Named("coefficients") = coef_ols
  );
}
