#' Spike and Slab Hyperparameter for VAR Coefficient
#' 
#' Set Hyperparameters for VAR coefficient matrix.
#' 
#' @param spike_sd Standard deviance for Spike normal distribution (See Details).
#' @param slab_sd Standard deviance for Slab normal distribution (See Details).
#' @param prop_sparse Bernoulli parameter for sparsity proportion (See Details).
#' @details 
#' Let \eqn{\alpha} be the vectorized coefficient,
#' \deqn{\alpha = vec(A)}
#' Spike-slab prior is given using two normal distributions.
#' \deqn{\alpha_j \mid \gamma_j \sim (1 - \gamma_j) N(0, \kappa_{0j}^2) + \gamma_j N(0, \kappa_{1j}^2)}
#' As spike-slab prior itself suggests, set \eqn{\kappa_{0j}} small (point mass at zero: spike distribution)
#' and set \eqn{\kappa_{1j}} large (symmetric by zero: slab distribution).
#' 
#' \eqn{\gamma_j} is the proportion of the nonzero coefficients and it follows
#' \deqn{\gamma_j \sim Bernoulli(q_j)}
#' 
#' * `spike_sd`: \eqn{\kappa_{0j}}
#' * `slab_sd`: \eqn{\kappa_{1j}}
#' * `prop_sparse`: \eqn{q_j}
#' * \eqn{j = 1, \ldots, mk}: vectorized format corresponding to coefficient matrix
#' 
#' @references 
#' Jochmann, M., Koop, G., & Strachan, R. W. (2010). *Bayesian forecasting using stochastic search variable selection in a VAR subject to breaks*. International Journal of Forecasting, 26(2), 326–347. doi:[10.1016/j.ijforecast.2009.11.002](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001782?via%3Dihub)
#' 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001753?via%3Dihub)
#' 
#' @order 1
#' @export
set_spikeslab_coef <- function(spike_sd = NULL, slab_sd = NULL, prop_sparse = NULL) {
  if (!is.vector(spike_sd)) {
    stop("'spike_sd' should be vectorized.")
  }
  if (!is.vector(slab_sd)) {
    stop("'slab_sd' should be vectorized.")
  }
  if (!is.vector(prop_sparse)) {
    stop("'prop_sparse' should be vectorized.")
  }
  if (length(spike_sd) != length(slab_sd)) {
    stop("The length of 'spike_sd' and 'spike_sd' should be the same.")
  }
  coef_param <- list(
    coef_spike = spike_sd,
    coef_slab = slab_sd,
    coef_mixture = prop_sparse
  )
  class(coef_param) <- "bvharss_coef"
  coef_param
}

#' Spike and Slab Hyperparameter for VAR Covariance
#' 
#' Set Hyperparameters for VAR covariance matrix (precision matrix).
#' 
#' @param shape Gamma shape parameters for precision matrix
#' @param rate Gamma rate parameters for precision matrix
#' @param spike_sd Standard deviance for Spike normal distribution, in the precision prior.
#' @param slab_sd Standard deviance for Slab normal distribution, in the precision prior.
#' @param prop_sparse Bernoulli parameter for sparsity proportion, in the precision prior.
#' @details 
#' For precision matrix \eqn{\Sigma_e^{-1}}, SSVS applies Cholesky decomposition.
#' \deqn{\Sigma_e^{-1} = \Psi \Psi^T}
#' where \eqn{\Psi = \{\psi_{ij}\}} is upper triangular.
#' 
#' Diagonal components follow the gamma distribution.
#' \deqn{\psi_{jj}^2 \sim Gamma(shape = a_j, rate = b_j)}
#' For each row of off-diagonal (upper-triangular) components, we apply spike-slab prior again.
#' \deqn{\psi_{ij} \mid w_{ij} \sim (1 - w_{ij}) N(0, \kappa_{0,ij}^2) + w_{ij} N(0, \kappa_{1,ij}^2)}
#' \deqn{w_{ij} \sim Bernoulli(q_{ij})}
#' 
#' * `shape`: \eqn{a_j}
#' * `rate`: \eqn{b_j}
#' * `spike_sd`: \eqn{\kappa_{0,ij}}
#' * `slab_sd`: \eqn{\kappa_{1,ij}}
#' * `prop_sparse`: \eqn{q_{ij}}
#' * \eqn{j = 1, \ldots, mk}: vectorized format corresponding to coefficient matrix
#' * \eqn{i = 1, \ldots, j - 1} and \eqn{j = 2, \ldots, m}: \eqn{\eta = (\psi_{12}, \psi_{13}, \psi_{23}, \psi_{14}, \ldots, \psi_{34}, \ldots, \psi_{1m}, \ldots, \psi_{m - 1, m})^T}
#' 
#' @references 
#' Jochmann, M., Koop, G., & Strachan, R. W. (2010). *Bayesian forecasting using stochastic search variable selection in a VAR subject to breaks*. International Journal of Forecasting, 26(2), 326–347. doi:[10.1016/j.ijforecast.2009.11.002](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001782?via%3Dihub)
#' 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001753?via%3Dihub)
#' 
#' @order 1
#' @export
set_spikeslab_cov <- function(shape = NULL,
                              rate = NULL,
                              spike_sd = NULL,
                              slab_sd = NULL,
                              prop_sparse = NULL) {
  if (!is.vector(shape)) {
    stop("'shape' should be vectorized.")
  }
  if (!is.vector(rate)) {
    stop("'rate' should be vectorized.")
  }
  if (length(shape) != length(rate)) {
    stop("The length of 'shape' and 'rate' should be the same.")
  }
  if (!is.vector(spike_sd)) {
    stop("'spike_sd' should be vectorized.")
  }
  if (!is.vector(slab_sd)) {
    stop("'slab_sd' should be vectorized.")
  }
  if (!is.vector(prop_sparse)) {
    stop("'prop_sparse' should be vectorized.")
  }
  if (length(spike_sd) != length(slab_sd)) {
    stop("The length of 'spike_sd' and 'spike_sd' should be the same.")
  }
  cov_param <- list(
    cov_shape = shape,
    cov_rate = rate,
    cov_spike = spike_sd,
    cov_slab = slab_sd,
    cov_mixture = prop_sparse
  )
  class(cov_param) <- "bvharss_sig"
  cov_param
}

#' Hyperparameters for SSVS Model
#' 
#' Set hyperparameters of stochastic search variable selection for Bayesian VAR model.
#' 
#' @param coef_ss Spike and slab specification for vectorized coefficient, using [set_spikeslab_coef()].
#' @param sig_ss Spike and slab specification for covariance matrix, using [set_spikeslab_cov()].
#' @details 
#' Get the default SSVS setting for given VAR model.
#' 
#' @references 
#' Jochmann, M., Koop, G., & Strachan, R. W. (2010). *Bayesian forecasting using stochastic search variable selection in a VAR subject to breaks*. International Journal of Forecasting, 26(2), 326–347. doi:[10.1016/j.ijforecast.2009.11.002](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001782?via%3Dihub)
#' 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001753?via%3Dihub)
#' 
#' @seealso 
#' * [set_spikeslab_coef()]: Set spike and slab prior hyperparameters for coefficient vector
#' * [set_spikeslab_cov()]: Set spike and slab prior hyperparameters for covariance matrix
#' 
#' @order 1
#' @export
set_ssvs <- function(coef_ss = set_spikeslab_coef(), sig_ss = set_spikeslab_cov()) {
  if (!is.bvharss_coef(coef_ss)) {
    stop("Invalid 'coef_ss'.")
  }
  if (!is.bvharss_sig(sig_ss)) {
    stop("Invalid 'sig_ss'.")
  }
  ssvs_param <- append(coef_ss, sig_ss)
  class(ssvs_param) <- "bvharss_spec"
  ssvs_param
}
