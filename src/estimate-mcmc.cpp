#include <RcppEigen.h>
#include "bvharmisc.h"
#include "bvharprob.h"

// [[Rcpp::depends(RcppEigen)]]

//' Generating Sparse Covariance Proportions Diagonal Matrix
//' 
//' In MCMC process of SSVS, generate diagonal matrix \eqn{F_j} (given j) defined by spike-and-slab sd.
//' 
//' @param col_index Choose the column index of cholesky factor
//' @param cov_spike Standard deviance for Spike normal distribution, for covariance prior
//' @param cov_slab Standard deviance for Slab normal distribution, for covariance prior
//' @param prop_sparse Indicator vector corresponding to each component
//' 
//' @references
//' Jochmann, M., Koop, G., & Strachan, R. W. (2010). *Bayesian forecasting using stochastic search variable selection in a VAR subject to breaks*. International Journal of Forecasting, 26(2), 326–347. doi:[10.1016/j.ijforecast.2009.11.002](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001782?via%3Dihub)
//' 
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001753?via%3Dihub)
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd ssvs_cov_prop(int col_index,
                              Eigen::VectorXd cov_spike,
                              Eigen::VectorXd cov_slab,
                              Eigen::VectorXd prop_sparse) {
  if (col_index == 0) {
    Rcpp::stop("'col_index' should be larger than 0.");
  }
  if (prop_sparse.size() != col_index) {
    Rcpp::stop("The length of 'prop_sparse' should be 'col_index'."); // w1j, ..., w(j-1,j)
  }
  Eigen::MatrixXd diag_sparse = Eigen::MatrixXd::Zero(col_index, col_index); // (j - 1) x (j - 1)
  int id = col_index * (col_index + 1) / 2;
  for (int i = 0; i < col_index - 1; i++) {
    diag_sparse(i, i) = prop_sparse(i) * cov_spike(id + i) + (1 - prop_sparse(i)) * cov_slab(id + i); // kappa[0,ij] if w[ij] = 0, kappa[1,ij] if w[ij] = 1
  }
  return diag_sparse;
}

//' Generating the Diagonal Component of Cholesky Factor
//' 
//' In MCMC process of SSVS, generate the diagonal component \eqn{\psi} of variance matrix
//' 
//' @param col_index Choose the column index of cholesky factor, starting from zero
//' @param ZtZ The result of \eqn{Z_0^T Z_0 = (Y_0 - X_0 A)^T (Y_0 - X_0 A)}
//' @param diag_sparse Generated sparse covariance proportions diagonal matrix
//' @param cov_shape Gamma shape parameters for precision matrix
//' @param cov_rate Gamma rate parameters for precision matrix
//' 
//' @references
//' Jochmann, M., Koop, G., & Strachan, R. W. (2010). *Bayesian forecasting using stochastic search variable selection in a VAR subject to breaks*. International Journal of Forecasting, 26(2), 326–347. doi:[10.1016/j.ijforecast.2009.11.002](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001782?via%3Dihub)
//' 
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001753?via%3Dihub)
//' 
//' @noRd
// [[Rcpp::export]]
double ssvs_cov_diag(int col_index, 
                     Eigen::MatrixXd ZtZ, 
                     Eigen::MatrixXd diag_sparse,
                     Eigen::VectorXd cov_shape,
                     Eigen::VectorXd cov_rate) {
  int dim = ZtZ.cols(); // m
  if (col_index > dim) {
    Rcpp::stop("Invalid 'col_index' argument.");
  }
  int num_design = ZtZ.rows(); // s = n - p
  double shape = cov_shape(col_index) + num_design / 2; // a[j] + s / 2
  double rate = cov_rate(col_index); // b[j] + something
  double chol_diag;
  if (col_index == 0) {
    rate += ZtZ(0, 0) / 2; // b[1] + v11
    chol_diag = gamma_rand(shape, 1 / rate); // psi[jj]^2 ~ Gamma
  } else {
    Eigen::MatrixXd z_j(col_index, col_index); // V(j - 1)
    Eigen::MatrixXd large_mat(col_index, col_index);
    z_j = ZtZ.block(0, 0, col_index, col_index);
    large_mat = z_j.transpose() * (z_j + (diag_sparse * diag_sparse).inverse()).inverse() * z_j;
    rate += (ZtZ(col_index, col_index) - large_mat(col_index, col_index)) / 2;
    chol_diag = gamma_rand(shape, 1 / rate); // psi[jj]^2 ~ Gamma
  }
  return sqrt(chol_diag);
}

//' Generating the Off-Diagonal Component of Cholesky Factor
//' 
//' In MCMC process of SSVS, generate the off-diagonal component \eqn{\psi} of variance matrix
//' 
//' @param col_index Choose the column index of cholesky factor
//' @param ZtZ The result of \eqn{(Y_0 - X_0 A)^T (Y_0 - X_0 A)}
//' @param chol_factor Cholesky factor of variance matrix
//' @param diag_sparse Generated sparse covariance proportions diagonal matrix
//' 
//' @references
//' Jochmann, M., Koop, G., & Strachan, R. W. (2010). *Bayesian forecasting using stochastic search variable selection in a VAR subject to breaks*. International Journal of Forecasting, 26(2), 326–347. doi:[10.1016/j.ijforecast.2009.11.002](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001782?via%3Dihub)
//' 
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001753?via%3Dihub)
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_cov_off(int col_index, 
                             Eigen::MatrixXd ZtZ, 
                             Eigen::MatrixXd chol_factor, 
                             Eigen::MatrixXd diag_sparse) {
  Eigen::MatrixXd normal_variance = ZtZ.block(0, 0, col_index, col_index) + (diag_sparse * diag_sparse).inverse();
  Eigen::VectorXd normal_mean = -chol_factor(col_index, col_index) * normal_variance * ZtZ.block(0, col_index, col_index, 1);
  return vectorize_eigen(sim_mgaussian(1, normal_mean, normal_variance));
}

//' Symmetric Matrix from Diagonal and Off-diagonal Components
//' 
//' Build a matrix using diagonal component vector and off-diaognal component vector
//' 
//' @param diag_vec Diagonal components
//' @param off_diagvec Off-diagonal components
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd build_symmat(Eigen::VectorXd diag_vec, Eigen::VectorXd off_diagvec) {
  int dim = diag_vec.size();
  Eigen::MatrixXd res(dim, dim);
  int id;
  res(0, 0) = diag_vec(0);
  for (int j = 1; j < dim; j++) {
    res(j, j) = diag_vec(j);
    id = j * (j - 1) / 2;
    for (int i = 0; i < j; i++) {
      res(i, j) = off_diagvec(id + i);
      res(j, i) = off_diagvec(id + i);
    }
  }
  return res;
}

//' Generating Latent Vector for Spike-and-Slab Coefficient
//' 
//' In MCMC process of SSVS, generate latent \eqn{w_{ij}} conditional posterior.
//' 
//' @param chol_factor Cholesky factor of variance matrix
//' @param cov_spike Standard deviance for Spike normal distribution, for covariance prior
//' @param cov_slab Standard deviance for Slab normal distribution, for covariance prior
//' @param cov_sparse Bernoulli parameter for sparsity proportion, for covariance prior
//' 
//' @references
//' Jochmann, M., Koop, G., & Strachan, R. W. (2010). *Bayesian forecasting using stochastic search variable selection in a VAR subject to breaks*. International Journal of Forecasting, 26(2), 326–347. doi:[10.1016/j.ijforecast.2009.11.002](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001782?via%3Dihub)
//' 
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001753?via%3Dihub)
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_cov_latent(Eigen::MatrixXd chol_factor,
                                Eigen::VectorXd cov_spike,
                                Eigen::VectorXd cov_slab,
                                Eigen::VectorXd cov_sparse) {
  double bernoulli_param_spike;
  double bernoulli_param_slab;
  int num_latent = cov_sparse.size(); // upper triangular (w[12], w[13], w[23], w[14], ..., w[34], w[1m], ..., w[m-1, m])
  Eigen::VectorXd latent_prop(num_latent); // w[ij] | Y0, -w[ij] ~ Bernoulli(q[ij]-bar)
  int dim = chol_factor.cols(); // m
  int id;
  for (int j = 1; j < dim; j++) {
    id = j * (j - 1) / 2;
    for (int i = 0; i < j; i++) {
      bernoulli_param_spike = cov_sparse(id + i) * exp(-pow(chol_factor(i, j) / (2 * cov_spike(id + i)), 2.0)) / cov_spike(id + i);
      bernoulli_param_slab = cov_sparse(id + i) * exp(-pow(chol_factor(i, j) / (2 * cov_slab(id + i)), 2.0)) / cov_slab(id + i);
    }
    latent_prop(j) = binom_rand(1, bernoulli_param_slab / (bernoulli_param_spike + bernoulli_param_slab)); // qij-bar
  }
  return latent_prop;
}

//' Generating Nonzero Coefficients Proportions Diagonal Matrix
//' 
//' In MCMC process of SSVS, generate diagonal matrix \eqn{D} defined by spike-and-slab sd.
//' 
//' @param coef_spike Standard deviance for Spike normal distribution
//' @param coef_slab Standard deviance for Slab normal distribution
//' @param prop_sparse Indicator vector corresponding to each coefficient
//' 
//' @references
//' Jochmann, M., Koop, G., & Strachan, R. W. (2010). *Bayesian forecasting using stochastic search variable selection in a VAR subject to breaks*. International Journal of Forecasting, 26(2), 326–347. doi:[10.1016/j.ijforecast.2009.11.002](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001782?via%3Dihub)
//' 
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001753?via%3Dihub)
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::MatrixXd ssvs_coef_prop(Eigen::VectorXd coef_spike,
                               Eigen::VectorXd coef_slab,
                               Eigen::VectorXd prop_sparse) {
  int num_coef_vec = coef_spike.size();
  if (prop_sparse.size() != num_coef_vec) {
    Rcpp::stop("The length of 'prop_sparse' should be the same as the coefficient vector.");
  }
  Eigen::MatrixXd diag_sparse = Eigen::MatrixXd::Zero(num_coef_vec, num_coef_vec);
  // Eigen::VectorXd prop_sparse(num_coef_vec);
  for (int i = 0; i < num_coef_vec; i++) {
    // prop_sparse(i) = binom_rand(1, coef_sparse(i)); // gamma[j] ~ bernoulli(q[j])
    diag_sparse(i, i) = prop_sparse(i) * coef_spike(i) + (1 - prop_sparse(i)) * coef_slab(i); // kappa[0j] if gamma[j] = 0, kappa[1j] if gamma[j] = 1
  }
  return diag_sparse;
}

//' Generating Coefficient Vector in SSVS Gibbs Sampler
//' 
//' In MCMC process of SSVS, generate \eqn{\alpha_j} conditional posterior.
//' 
//' @param XtX The result of design matrix arithmetic \eqn{X_0^T X_0}
//' @param coef_lse LSE estimator of the VAR coefficient
//' @param chol_factor Cholesky factor of variance matrix
//' @param diag_sparse Generated sparse coefficient proportions diagonal matrix
//' 
//' @references
//' Jochmann, M., Koop, G., & Strachan, R. W. (2010). *Bayesian forecasting using stochastic search variable selection in a VAR subject to breaks*. International Journal of Forecasting, 26(2), 326–347. doi:[10.1016/j.ijforecast.2009.11.002](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001782?via%3Dihub)
//' 
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001753?via%3Dihub)
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_coef(Eigen::MatrixXd XtX,
                          Eigen::VectorXd coef_lse,
                          Eigen::MatrixXd chol_factor,
                          Eigen::MatrixXd diag_sparse) {
  Eigen::MatrixXd prec_mat = chol_factor * chol_factor.transpose();
  Eigen::MatrixXd lhs_kronecker = kronecker_eigen(prec_mat, XtX); // (Sigma)^(-1) otimes (X_0^T X_0)
  Eigen::MatrixXd normal_variance = (lhs_kronecker + (diag_sparse * diag_sparse).inverse()).inverse(); // v-bar
  Eigen::VectorXd normal_mean = normal_variance * lhs_kronecker * coef_lse; // alpha-bar
  return vectorize_eigen(sim_mgaussian(1, normal_mean, normal_variance));
}

//' Generating Latent Vector for Spike-and-Slab Coefficient
//' 
//' In MCMC process of SSVS, generate latent \eqn{\gamma_j} conditional posterior.
//' 
//' @param coef_vec Coefficient vector
//' @param coef_spike Standard deviance for Spike normal distribution
//' @param coef_slab Standard deviance for Slab normal distribution
//' @param coef_sparse Bernoulli parameter for sparsity proportion
//' 
//' @references
//' Jochmann, M., Koop, G., & Strachan, R. W. (2010). *Bayesian forecasting using stochastic search variable selection in a VAR subject to breaks*. International Journal of Forecasting, 26(2), 326–347. doi:[10.1016/j.ijforecast.2009.11.002](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001782?via%3Dihub)
//' 
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001753?via%3Dihub)
//' 
//' @noRd
// [[Rcpp::export]]
Eigen::VectorXd ssvs_coef_latent(Eigen::VectorXd coef_vec, 
                                 Eigen::VectorXd coef_spike,
                                 Eigen::VectorXd coef_slab,
                                 Eigen::VectorXd coef_sparse) {
  double bernoulli_param_spike;
  double bernoulli_param_slab;
  int num_latent = coef_sparse.size();
  Eigen::VectorXd latent_prop(num_latent); // gammaj | Y0, -gammaj ~ Bernoulli(qj-bar)
  for (int i = 0; i < num_latent; i ++) {
    bernoulli_param_spike = coef_sparse(i) * exp(-pow(coef_vec(i) / (2 * coef_spike(i)), 2.0)) / coef_spike(i);
    bernoulli_param_slab = coef_sparse(i) * exp(-pow(coef_vec(i) / (2 * coef_slab(i)), 2.0)) / coef_slab(i);
    latent_prop(i) = binom_rand(1, bernoulli_param_slab / (bernoulli_param_spike + bernoulli_param_slab)); // qj-bar
  }
  return latent_prop;
}

//' BVAR(p) Point Estimates based on SSVS Prior
//' 
//' Compute MCMC for SSVS prior
//' 
//' @param x Design matrix X0
//' @param y Response matrix Y0
//' @param init_coef Initial k x m coefficient matrix.
//' @param init_cov Initial m x m variance matrix.
//' @param coef_spike Standard deviance for Spike normal distribution
//' @param coef_slab Standard deviance for Slab normal distribution
//' @param coef_sparse Bernoulli parameter for sparsity proportion
//' @param cov_shape Gamma shape parameters for precision matrix
//' @param cov_rate Gamma rate parameters for precision matrix
//' @param cov_spike Standard deviance for Spike normal distribution, for covariance prior
//' @param cov_slab Standard deviance for Slab normal distribution, for covariance prior
//' @param cov_sparse Bernoulli parameter for sparsity proportion, for covariance prior
//' @details
//' 1. Diagonal components of cholesky factor from Gamma distribution
//' 2. Off-diagonal components from Normal distribution
//' 3. Proportion of covariance sparsity from Bernoulli
//' 4. Coefficient from spike-and-slab based on the above simulated covariance matrix
//' 5. Proportion of nonzero coefficient from Bernoulli
//' 
//' @references
//' Jochmann, M., Koop, G., & Strachan, R. W. (2010). *Bayesian forecasting using stochastic search variable selection in a VAR subject to breaks*. International Journal of Forecasting, 26(2), 326–347. doi:[10.1016/j.ijforecast.2009.11.002](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001782?via%3Dihub)
//' 
//' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001753?via%3Dihub)
//' 
//' @noRd
// [[Rcpp::export]]
Rcpp::List estimate_bvar_ssvs(int num_iter,
                              Eigen::MatrixXd x, 
                              Eigen::MatrixXd y, 
                              Eigen::MatrixXd init_coef,
                              Eigen::Map<Eigen::MatrixXd> init_cov,
                              Eigen::VectorXd coef_spike,
                              Eigen::VectorXd coef_slab,
                              Eigen::VectorXd coef_sparse,
                              Eigen::VectorXd cov_shape,
                              Eigen::VectorXd cov_rate,
                              Eigen::VectorXd cov_spike,
                              Eigen::VectorXd cov_slab,
                              Eigen::VectorXd cov_sparse) {
  int dim = y.cols(); // m
  int dim_design = x.cols(); // k = mp (+ 1)
  int num_design = y.rows(); // s = n - p
  // Eigen::MatrixXd coef_mat(dim_design, dim); // A: k x m
  Eigen::MatrixXd XtX = x.transpose() * x; // X_0^T X_0: k x k
  // initial for coefficient-----------------------------------------------------------
  if (init_coef.rows() != dim_design) {
    Rcpp::stop("Invalid number of nrow of 'init_coef'.");
  }
  if (init_coef.cols() != dim) {
    Rcpp::stop("Invalid number of ncol of 'init_coef'.");
  }
  Eigen::MatrixXd resid_mat = y - x * init_coef; // Z = Y0 - X0 B
  Eigen::MatrixXd ZtZ = resid_mat.transpose() * resid_mat; // (Y0 - X0 B)^T (Y0 - X0 B)
  Eigen::VectorXd coef_vec = vectorize_eigen(init_coef); // alphahat = vec(A)
  Eigen::MatrixXd diag_coef_sparse(coef_sparse.size(), coef_sparse.size()); // D: mk x mk
  Eigen::MatrixXd diag_cov_sparse(dim - 1, dim - 1); // large matrix to assign diag_sparse of cov
  // initial for covariance matrix: Sigma_e^(-1) = Psi * Psi^T--------------------------
  Eigen::LLT<Eigen::MatrixXd> lltOfcov(init_cov); // Sigma = LL^T
  if (lltOfcov.info() == Eigen::NumericalIssue) {
    Rcpp::stop("'init_cov' should be positive definite matrix."); // Sigma is pd
  }
  Eigen::MatrixXd chol_upper = lltOfcov.matrixL().transpose(); // Psi = (L^(-1))^T: upper
  
  // Gibbs sampler iteration------------------------------------------------------------
  for (int t = 0; t < num_iter; t++) {
    // 1. diagonal components of cholesky factor
    chol_upper(0, 0) = ssvs_cov_diag(0, ZtZ, diag_cov_sparse, cov_shape, cov_rate); // Gamma
    for (int i = 1; i < dim; i++) {
      diag_cov_sparse.block(0, 0, i, i) = ssvs_cov_prop(i, cov_spike, cov_slab, cov_sparse); // Fj
      chol_upper(i, i) = ssvs_cov_diag(i, ZtZ, diag_cov_sparse.block(0, 0, i, i), cov_shape, cov_rate); // sqrt of Gamma
    }
    // 2. Off-diagonal components
    for (int j = 1; j < dim; j++) {
      diag_cov_sparse.block(0, 0, j, j) = ssvs_cov_prop(j, cov_spike, cov_slab, cov_sparse); // Fj
      chol_upper.block(0, j, j - 1, 1) = ssvs_cov_off(j, ZtZ, chol_upper, diag_cov_sparse.block(0, 0, j, j));
    }
    // 3. Sparsity proportion of variance matrix
    
    // 4. Coefficient
    
    // 5. Sparsity proportion of coefficient
  }
  
  
  
  
  return Rcpp::List::create(coef_vec, diag_coef_sparse); // temporary
}
