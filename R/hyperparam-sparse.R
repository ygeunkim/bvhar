#' Spike and Slab Hyperparameter for VAR Coefficient
#' 
#' Set Hyperparameters for VAR coefficient matrix.
#' 
#' @param spike_sd Standard deviance for Spike normal distribution (See Details).
#' @param slab_sd Standard deviance for Slab normal distribution (See Details).
#' @param slab_weight Bernoulli parameter for sparsity proportion (See Details).
#' @details 
#' Let \eqn{\alpha} be the vectorized coefficient, \eqn{\alpha = vec(A)}.
#' Spike-slab prior is given using two normal distributions.
#' \deqn{\alpha_j \mid \gamma_j \sim (1 - \gamma_j) N(0, \tau_{0j}^2) + \gamma_j N(0, \tau_{1j}^2)}
#' As spike-slab prior itself suggests, set \eqn{\tau_{0j}} small (point mass at zero: spike distribution)
#' and set \eqn{\tau_{1j}} large (symmetric by zero: slab distribution).
#' 
#' \eqn{\gamma_j} is the proportion of the nonzero coefficients and it follows
#' \deqn{\gamma_j \sim Bernoulli(p_j)}
#' 
#' * `spike_sd`: \eqn{\tau_{0j}}
#' * `slab_sd`: \eqn{\tau_{1j}}
#' * `slab_weight`: \eqn{p_j}
#' * \eqn{j = 1, \ldots, mk}: vectorized format corresponding to coefficient matrix
#' @references 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
#' 
#' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
#' @order 1
#' @export
set_spikeslab_coef <- function(spike_sd = NULL, slab_sd = NULL, slab_weight = NULL) {
  if (is.vector(spike_sd) && is.null(spike_sd)) {
    stop("'spike_sd' should be vectorized.")
  }
  if (is.vector(slab_sd) && is.null(slab_sd)) {
    stop("'slab_sd' should be vectorized.")
  }
  if (is.vector(slab_weight) && is.null(slab_weight)) {
    stop("'slab_weight' should be vectorized.")
  }
  if ((length(spike_sd) != length(slab_sd)) || (length(slab_weight) != length(spike_sd))) {
    stop("The length of 'spike_sd', 'spike_sd', and 'slab_weight' should be the same.") # mk
  }
  coef_param <- list(
    coef_spike = spike_sd,
    coef_slab = slab_sd,
    coef_mixture = slab_weight
  )
  class(coef_param) <- "bvharss_coef"
  coef_param
}

#' Spike and Slab Hyperparameter for VAR Covariance
#' 
#' Set Hyperparameters for cholesky factor of VAR precision matrix.
#' 
#' @param shape Gamma shape parameters for precision matrix
#' @param rate Gamma rate parameters for precision matrix
#' @param spike_sd Standard deviance for Spike normal distribution, in the precision prior.
#' @param slab_sd Standard deviance for Slab normal distribution, in the precision prior.
#' @param slab_weight Bernoulli parameter for sparsity proportion, in the precision prior.
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
#' * `slab_weight`: \eqn{q_{ij}}
#' * \eqn{j = 1, \ldots, mk}: vectorized format corresponding to coefficient matrix
#' * \eqn{i = 1, \ldots, j - 1} and \eqn{j = 2, \ldots, m}: \eqn{\eta = (\psi_{12}, \psi_{13}, \psi_{23}, \psi_{14}, \ldots, \psi_{34}, \ldots, \psi_{1m}, \ldots, \psi_{m - 1, m})^T}
#' 
#' @references
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
#' 
#' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
#' @order 1
#' @export
set_spikeslab_chol <- function(shape = NULL,
                               rate = NULL,
                               spike_sd = NULL,
                               slab_sd = NULL,
                               slab_weight = NULL) {
  if (is.vector(shape) && is.null(shape)) {
    stop("'shape' should be vectorized.")
  }
  if (is.vector(rate) && is.null(rate)) {
    stop("'rate' should be vectorized.")
  }
  if (length(shape) != length(rate)) {
    stop("The length of 'shape' and 'rate' should be the same.")
  }
  if (is.vector(spike_sd) && is.null(spike_sd)) {
    stop("'spike_sd' should be vectorized.")
  }
  if (is.vector(slab_sd) && is.null(slab_sd)) {
    stop("'slab_sd' should be vectorized.")
  }
  if (is.vector(slab_weight) && is.null(slab_weight)) {
    stop("'slab_weight' should be vectorized.")
  }
  if ((length(spike_sd) != length(slab_sd)) || (length(slab_weight) != length(slab_weight))) {
    stop("The length of 'spike_sd', 'spike_sd', and 'slab_weight' should be the same.") # upper triangular
  }
  chol_param <- list(
    chol_shape = shape,
    chol_rate = rate,
    chol_spike = spike_sd,
    chol_slab = slab_sd,
    chol_mixture = slab_weight
  )
  class(chol_param) <- "bvharss_sig"
  chol_param
}

#' Initial parameters and Hyperparameters for SSVS Model
#' 
#' Set initial parameters and hyperparameters of stochastic search variable selection for Bayesian VAR model.
#' 
#' @param init_coef Initial k x m coefficient matrix.
#' @param init_coef_dummy Initial k x m indicator matrix (1-0) corresponding to each component of coefficient.
#' @param init_chol Initial m x m variance matrix.
#' @param init_chol_dummy Initial m x m indicator matrix (1-0) corresponding to each component of variance matrix.
#' @param coef_ss Spike and slab specification for vectorized coefficient, using [set_spikeslab_coef()].
#' @param chol_ss Spike and slab specification for cholesky factor of precision matrix, using [set_spikeslab_chol()].
#' @details 
#' Get the default SSVS setting for given VAR model.
#' 
#' @references
#' Jochmann, M., Koop, G., & Strachan, R. W. (2010). *Bayesian forecasting using stochastic search variable selection in a VAR subject to breaks*. International Journal of Forecasting, 26(2), 326–347. doi:[10.1016/j.ijforecast.2009.11.002](https://doi.org/10.1016/j.ijforecast.2009.11.002)
#' 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
#' @seealso 
#' * [set_spikeslab_coef()]: Set spike and slab prior hyperparameters for coefficient vector
#' * [set_spikeslab_chol()]: Set spike and slab prior hyperparameters for cholesky factor
#' 
#' @order 1
#' @export
set_ssvs <- function(init_coef,
                     init_coef_dummy,
                     init_chol,
                     init_chol_dummy,
                     coef_ss = set_spikeslab_coef(), 
                     chol_ss = set_spikeslab_chol()) {
  if (!is.bvharss_coef(coef_ss)) {
    stop("Invalid 'coef_ss'.")
  }
  if (!is.bvharss_chol(chol_ss)) {
    stop("Invalid 'chol_ss'.")
  }
  # Dimensions of parameters---------------------------
  dim_design <- nrow(init_coef) # kp + 1
  dim_data <- ncol(init_coef) # dim
  if (!(nrow(init_coef_dummy) == dim_design && ncol(init_coef_dummy) == dim_data)) {
    stop("Invalid dimension of 'init_coef_dummy'.")
  }
  if (!(nrow(init_chol) != dim_data && ncol(init_chol) != dim_data)) {
    stop("Invalid dimension of 'init_chol'.")
  }
  if (!(nrow(init_chol_dummy) != dim_data && ncol(init_chol_dummy) != dim_data)) {
    stop("Invalid dimension of 'init_chol_sparse'.")
  }
  # Initial values if NULL-----------------------------
  if (is.null(coef_ss$coef_spike)) {
    coef_ss$coef_spike <- rep(NA, dim_data * dim_design) # NA vector of length mk - compute in bvar_ssvs
  }
  if (is.null(coef_ss$coef_slab)) {
    coef_ss$coef_slab <- rep(NA, dim_data * dim_design) # NA vector of length mk - compute in bvar_ssvs
  }
  if (is.null(coef_ss$coef_mixture)) {
    coef_ss$coef_mixture <- rep(.5, dim_data * dim_design) # natural default choice
  }
  if (is.null(chol_ss$cov_shape)) {
    chol_ss$chol_shape <- rep(2.2, dim_data) # non-informative choice
  }
  if (is.null(chol_ss$cov_rate)) {
    chol_ss$chol_rate <- rep(.24, dim_data) # non-informative choice
  }
  if (is.null(chol_ss$cov_spike)) {
    chol_ss$chol_spike <- rep(NA, dim_data * (dim_data - 1) / 2) # NA vector of length m(m-1) / 2 - compute in bvar_ssvs
  }
  if (is.null(chol_ss$chol_slab)) {
    chol_ss$chol_slab <- rep(NA, dim_data * (dim_data - 1) / 2) # NA vector of length m(m-1) / 2 - compute in bvar_ssvs
  }
  if (is.null(chol_ss$chol_mixture)) {
    chol_ss$chol_mixture <- rep(.5, dim_data * (dim_data - 1) / 2)
  }
  # Dimensions of hyperparameters----------------------
  if (length(coef_ss$coef_mixture) != dim_data * dim_design) {
    stop("Invalid length of Coefficients spike-and-slab hyperparameters.") # mk
  }
  if (length(chol_ss$chol_shape) != dim_data) {
    stop("Invalid length of Gamma hyperparameters.") # m
  }
  if (length(chol_ss$chol_mixture) != dim_data * (dim_data - 1) / 2) {
    stop("Invalid length of Variance spike-and-slab hyperparameters.") # m * (m - 1) / 2
  }
  # return--------------------------------------------
  ssvs_param <- append(coef_ss, sig_ss)
  ssvs_param$init_coef <- init_coef
  ssvs_param$init_coef_dummy <- init_coef_dummy
  ssvs_param$init_chol <- init_chol
  ssvs_param$init_chol_dummy <- init_chol_dummy
  class(ssvs_param) <- "bvharss_spec"
  ssvs_param
}
