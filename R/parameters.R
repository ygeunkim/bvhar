#' Initialize Parameters Randomly using Uniform Distribution
#'
#' `r lifecycle::badge("experimental")` Used in initializer functions, this function can initialize a parameter randomly.
#'
#' @param min Lower limit of uniform distribution
#' @param max Upper limit of uniform distribution
#' #' @return `unifinit` class.
#' @order 1
#' @export
init_unif <- function(min = -1, max = 1) {
  res <- list(
    type = "unif",
    param = c(min, max)
  )
  class(res) <- c("unifinit", "paraminit")
  res
}

#' Initialize Parameters as Constant Values
#'
#' `r lifecycle::badge("experimental")` Used in initializer functions, this function can initialize a parameter by one value.
#'
#' @param scl Initialize by `scl * 1`.
#' @return `sclinit` class.
#' @order 1
#' @export
init_const <- function(scl = 1) {
  res <- list(
    type = "const",
    param = scl
  )
  class(res) <- c("sclinit", "paraminit")
  res
}

#' Initial Parameters of Stochastic Search Variable Selection (SSVS) Model
#'
#' Set initial parameters before starting Gibbs sampler for SSVS.
#'
#' @param init_coef Initial coefficient matrix. Initialize with an array or list for multiple chains.
#' @param init_coef_dummy Initial indicator matrix (1-0) corresponding to each component of coefficient. Initialize with an array or list for multiple chains.
#' @param init_chol Initial cholesky factor (upper triangular). Initialize with an array or list for multiple chains.
#' @param init_chol_dummy Initial indicator matrix (1-0) corresponding to each component of cholesky factor. Initialize with an array or list for multiple chains.
#' @param type `r lifecycle::badge("experimental")` Type to choose initial values. One of `"user"` (User-given) and `"auto"` (OLS for coefficients and 1 for dummy).
#' @details
#' Set SSVS initialization for the VAR model.
#'
#' * `init_coef`: (kp + 1) x m \eqn{A} coefficient matrix.
#' * `init_coef_dummy`: kp x m \eqn{\Gamma} dummy matrix to restrict the coefficients.
#' * `init_chol`: k x k \eqn{\Psi} upper triangular cholesky factor, which \eqn{\Psi \Psi^\intercal = \Sigma_e^{-1}}.
#' * `init_chol_dummy`: k x k \eqn{\Omega} upper triangular dummy matrix to restrict the cholesky factor.
#'
#' Denote that `init_chol` and `init_chol_dummy` should be upper_triangular or the function gives error.
#'
#' For parallel chain initialization, assign three-dimensional array or three-length list.
#' @return `ssvsinit` object
#' @references
#' George, E. I., & McCulloch, R. E. (1993). *Variable Selection via Gibbs Sampling*. Journal of the American Statistical Association, 88(423), 881–889.
#'
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580.
#'
#' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358.
#' @order 1
#' @export
init_ssvs <- function(init_coef,
                      init_coef_dummy,
                      init_chol,
                      init_chol_dummy,
                      type = c("user", "auto")) {
  type <- match.arg(type)
  if (type == "auto") {
    init_coef <- NULL
    init_coef_dummy <- NULL
    init_chol <- NULL
    init_chol_dummy <- NULL
    num_chain <- 1
  } else {
    num_chain <- 1
    coef_mat <- init_coef
    coef_dummy <- init_coef_dummy
    chol_mat <- init_chol
    chol_dummy <- init_chol_dummy
    # Check dimension validity-----------------------------
    if (!(is.paraminit(coef_mat) &&
      is.paraminit(coef_dummy) &&
      is.paraminit(chol_mat) &&
      is.paraminit(chol_dummy))) {
      dim_design <- nrow(coef_mat) # kp(+1)
      dim_data <- ncol(coef_mat) # k = dim
      if (!(nrow(coef_dummy) == dim_design && ncol(coef_dummy) == dim_data)) {
        if (!(nrow(coef_dummy) == dim_design - 1 && ncol(coef_dummy) == dim_data)) {
          stop("Invalid dimension of 'init_coef_dummy'.")
        }
      }
      if (!(nrow(chol_mat) == dim_data && ncol(chol_mat) == dim_data)) {
        stop("Invalid dimension of 'init_chol'.")
      }
      if (any(chol_mat[lower.tri(chol_mat, diag = FALSE)] != 0)) {
        stop("'init_chol' should be upper triangular matrix.")
      }
      if (!(nrow(chol_dummy) == dim_data || ncol(chol_dummy) == dim_data)) {
        stop("Invalid dimension of 'init_chol_dummy'.")
      }
    }
  }
  res <- list(
    process = "VAR",
    prior = "SSVS",
    chain = num_chain,
    init_coef = init_coef,
    init_coef_dummy = init_coef_dummy,
    init_chol = init_chol,
    init_chol_dummy = init_chol_dummy,
    type = type
  )
  class(res) <- "ssvsinit"
  res
}

#' Horseshoe Prior Specification
#'
#' Set initial hyperparameters and parameter before starting Gibbs sampler for Horseshoe prior.
#'
#' @param local_sparsity Initial local shrinkage hyperparameters
#' @param global_sparsity Initial global shrinkage hyperparameter
#' @details
#' Set horseshoe prior initialization for VAR family.
#'
#' * `local_sparsity`: Local shrinkage for each row of coefficients matrix.
#' * `global_sparsity`: (Initial) global shrinkage.
#' * `init_cov`: Initial covariance matrix.
#'
#' In this package, horseshoe prior model is estimated by Gibbs sampling,
#' initial means initial values for that gibbs sampler.
#' @references
#' Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe estimator for sparse signals. Biometrika, 97(2), 465–480.
#'
#' Makalic, E., & Schmidt, D. F. (2016). *A Simple Sampler for the Horseshoe Estimator*. IEEE Signal Processing Letters, 23(1), 179–182.
#' @order 1
#' @export
set_horseshoe <- function(local_sparsity = 1, global_sparsity = 1) {
  if (!is.vector(local_sparsity)) {
    stop("'local_sparsity' should be a vector.")
  }
  # if (length(local_sparsity) > 1) {
  #   warning("Scalar 'local_sparsity' works.")
  # }
  # if (!is.matrix(init_cov)) {
  #   stop("'init_cov' should be a matrix.")
  # }
  # if (ncol(init_cov) != nrow(init_cov)) {
  #   stop("'init_cov' should be a square matrix.")
  # }
  if (length(global_sparsity) > 1) {
    stop("'global_sparsity' should be a scalar.")
  }
  res <- list(
    process = "VAR",
    prior = "Horseshoe",
    local_sparsity = local_sparsity,
    global_sparsity = global_sparsity # ,init_cov = init_cov
  )
  class(res) <- c("horseshoespec", "bvharspec")
  res
}

#' Initialize Stochastic Volatility Parameters
#' 
#' `r lifecycle::badge("experimental")` This function initializes stochastic volatility parameters in Gibbs sampler.
#' 
#' @param lvol Time-varying log-volatility.
#' The length is same with data dimension times sample size (\eqn{n = T - p}).
#' @param lvol_init Initial state of log-volatility.
#' The length is same with data dimension.
#' @param lvol_sig Variance of log-volatility.
#' The length is same with data dimension.
#' @param type `r lifecycle::badge("experimental")` Type to choose initial values. One of `"user"` (User-given) and `"auto"` (OLS for coefficients and 1 for dummy).
#' @references
#' Carriero, A., Chan, J., Clark, T. E., & Marcellino, M. (2022). *Corrigendum to “Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors” \[J. Econometrics 212 (1)(2019) 137–154\]*. Journal of Econometrics, 227(2), 506-512.
#'
#' Chan, J., Koop, G., Poirier, D., & Tobias, J. (2019). *Bayesian Econometric Methods (2nd ed., Econometric Exercises)*. Cambridge: Cambridge University Press.
#' @order 1
#' @export 
init_sv <- function(lvol = 0, lvol_init = .1, lvol_sig = .1, type = c("user", "auto")) {
  type <- match.arg(type)
  if (type == "auto") {
    lvol <- NULL
    lvol_init <- NULL
    lvol_sig <- NULL
  } else {
    if (!(is.paraminit(lvol) &&
      is.paraminit(lvol_init) &&
      is.paraminit(lvol_sig))) {
      if (!(is.vector(lvol_init) && is.vector(lvol_sig))) {
        stop("'lvol_init' and 'lvol_sig' should be a vector.")
      }
      if (is.matrix(lvol) && length(lvol_init) > 1) {
        if (ncol(lvol) != length(lvol_init)) {
          stop("Invalid size of 'lvol' or 'lvol_init'. 'lvol' should be n x dim.")
        }
      }
      if (is.matrix(lvol) && length(lvol_sig) > 1) {
        if (ncol(lvol) != length(lvol_sig)) {
          stop("Invalid size of 'lvol' or 'lvol_sig'. 'lvol' should be n x dim.")
        }
      }
      if (length(lvol) > 1 && length(lvol_init) > 1) {
        if (length(lvol) == length(lvol_init)) {
          stop("Invalid length of 'lvol' or 'lvol_init'.")
        }
      }
      if (length(lvol) > 1 && length(lvol_sig) > 1) {
        if (length(lvol) == length(lvol_sig)) {
          stop("Invalid length of 'lvol' or 'lvol_sig'.")
        }
      }
      if (length(lvol_init) > 1 && length(lvol_sig) > 1) {
        if (length(lvol_init) != length(lvol_sig)) {
          stop("Invalid length of 'lvol_init' or 'lvol_sig'. They should be the same as data dimension when length > 1.")
        }
      }
    }
  }
  res <- list(
    process = "SV",
    prior = "Cholesky",
    lvol = lvol,
    lvol_init = lvol_init,
    lvol_sig = lvol_sig,
    type = type
  )
  class(res) <- "svinit"
  res
}
