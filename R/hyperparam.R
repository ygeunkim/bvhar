#' Hyperparameters for Bayesian Models
#' 
#' Set hyperparameters of Bayesian VAR and VHAR models.
#' 
#' @param sigma Standard error vector for each variable (Default: sd of each variable)
#' @param lambda Tightness of the prior around a random walk or white noise (Default: .1)
#' @param delta Persistence (Litterman sets 1 = random walk prior (default: rep(1, number of variables)), White noise prior = 0)
#' @param eps Very small number (Default: 1e-04)
#' @details 
#' * Missing arguments will be set to be default values in each model function mentioned above.
#' * `set_bvar()` sets hyperparameters for [bvar_minnesota()].
#' * Each `delta` (vector), `lambda` (length of 1), `sigma` (vector), `eps` (vector) corresponds to \eqn{\delta_j}, \eqn{\lambda}, \eqn{\delta_j}, \eqn{\epsilon}.
#' 
#' \eqn{\delta_i} are related to the belief to random walk.
#' 
#' * If \eqn{\delta_i = 1} for all i, random walk prior
#' * If \eqn{\delta_i = 0} for all i, white noise prior
#' 
#' \eqn{\lambda} controls the overall tightness of the prior around these two prior beliefs.
#' 
#' * If \eqn{\lambda = 0}, the posterior is equivalent to prior and the data do not influence the estimates.
#' * If \eqn{\lambda = \infty}, the posterior mean becomes OLS estimates (VAR).
#' 
#' \eqn{\sigma_i^2 / \sigma_j^2} in Minnesota moments explain the data scales.
#' @return Every function returns `bvharspec` [class].
#' It is the list of which the components are the same as the arguments provided.
#' If the argument is not specified, `NULL` is assigned here.
#' The default values mentioned above will be considered in each fitting function.
#' \describe{
#'   \item{process}{Model name: `BVAR`, `BVHAR`}
#'   \item{prior}{
#'   Prior name: `Minnesota` (Minnesota prior for BVAR),
#'   `Hierarchical` (Hierarchical prior for BVAR),
#'   `MN_VAR` (BVHAR-S),
#'   `MN_VHAR` (BVHAR-L),
#'   `Flat` (Flat prior for BVAR)
#'   }
#'   \item{sigma}{Vector value (or `bvharpriorspec` class) assigned for sigma}
#'   \item{lambda}{Value (or `bvharpriorspec` class) assigned for lambda}
#'   \item{delta}{Vector value assigned for delta}
#'   \item{eps}{Value assigned for epsilon}
#' }
#' @note 
#' By using [set_psi()] and [set_lambda()] each, hierarchical modeling is available.
#' @references 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1).
#' 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25.
#' @examples 
#' # Minnesota BVAR specification------------------------
#' bvar_spec <- set_bvar(
#'   sigma = c(.03, .02, .01), # Sigma = diag(.03^2, .02^2, .01^2)
#'   lambda = .2, # lambda = .2
#'   delta = rep(.1, 3), # delta1 = .1, delta2 = .1, delta3 = .1
#'   eps = 1e-04 # eps = 1e-04
#' )
#' class(bvar_spec)
#' str(bvar_spec)
#' @seealso 
#' * lambda hyperprior specification [set_lambda()]
#' * sigma hyperprior specification [set_psi()]
#' @order 1
#' @export
set_bvar <- function(sigma, lambda = .1, delta, eps = 1e-04) {
  if (missing(sigma)) {
    sigma <- NULL
  }
  if (missing(delta)) {
    delta <- NULL
  }
  hiearchical <- is.bvharpriorspec(sigma)
  if (hiearchical) {
    if (!all(is.bvharpriorspec(sigma) & is.bvharpriorspec(lambda))) {
      stop("When using hiearchical model, each 'sigma' and 'lambda' should be 'bvharpriorspec'.")
    }
    prior_type <- "MN_Hierarchical"
  } else {
    if (lambda <= 0) {
      stop("'lambda' should be larger than 0.")
    }
    if (length(sigma) > 0 & any(sigma <= 0)) {
      stop("'sigma' should be larger than 0.")
    }
    if (length(delta) > 0 & any(delta < 0)) {
      stop("'delta' should not be smaller than 0.")
    }
    if (length(sigma) > 0 & length(delta) > 0) {
      if (length(sigma) != length(delta)) {
        stop("Length of 'sigma' and 'delta' must be the same as the dimension of the time series.")
      }
    }
    prior_type <- "Minnesota"
  }
  bvar_param <- list(
    process = "BVAR",
    prior = prior_type,
    sigma = sigma,
    lambda = lambda,
    delta = delta,
    eps = eps
  )
  class(bvar_param) <- "bvharspec"
  bvar_param
}

#' @rdname set_bvar
#' @param U Positive definite matrix. By default, identity matrix of dimension ncol(X0)
#' @details 
#' * `set_bvar_flat` sets hyperparameters for [bvar_flat()].
#' @examples 
#' # Flat BVAR specification-------------------------
#' # 3-dim
#' # p = 5 with constant term
#' # U = 500 * I(mp + 1)
#' bvar_flat_spec <- set_bvar_flat(U = 500 * diag(16))
#' class(bvar_flat_spec)
#' str(bvar_flat_spec)
#' @references Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526).
#' @order 1
#' @export
set_bvar_flat <- function(U) {
  if (missing(U)) {
    U <- NULL
  }
  bvar_param <- list(
    process = "BVAR",
    prior = "Flat",
    U = U
  )
  class(bvar_param) <- "bvharspec"
  bvar_param
}

#' @rdname set_bvar
#' @param sigma Standard error vector for each variable (Default: sd)
#' @param lambda Tightness of the prior around a random walk or white noise (Default: .1)
#' @param delta Persistence (Default: Litterman sets 1 = random walk prior, White noise prior = 0)
#' @param eps Very small number (Default: 1e-04)
#' @details 
#' * `set_bvhar()` sets hyperparameters for [bvhar_minnesota()] with VAR-type Minnesota prior, i.e. BVHAR-S model.
#' @examples 
#' # BVHAR-S specification-----------------------
#' bvhar_var_spec <- set_bvhar(
#'   sigma = c(.03, .02, .01), # Sigma = diag(.03^2, .02^2, .01^2)
#'   lambda = .2, # lambda = .2
#'   delta = rep(.1, 3), # delta1 = .1, delta2 = .1, delta3 = .1
#'   eps = 1e-04 # eps = 1e-04
#' )
#' class(bvhar_var_spec)
#' str(bvhar_var_spec)
#' @references Kim, Y. G., and Baek, C. (2023+). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation.
#' @order 1
#' @export
set_bvhar <- function(sigma, lambda = .1, delta, eps = 1e-04) {
  if (missing(sigma)) {
    sigma <- NULL
  }
  if (missing(delta)) {
    delta <- NULL
  }
  if (length(sigma) > 0 & length(delta) > 0) {
    if (length(sigma) != length(delta)) {
      stop("Length of 'sigma' and 'delta' must be the same as the dimension of the time series.")
    }
  }
  bvhar_param <- list(
    process = "BVHAR",
    prior = "MN_VAR",
    sigma = sigma,
    lambda = lambda,
    delta = delta,
    eps = eps
  )
  class(bvhar_param) <- "bvharspec"
  bvhar_param
}

#' @rdname set_bvar
#' @param sigma Standard error vector for each variable (Default: sd)
#' @param lambda Tightness of the prior around a random walk or white noise (Default: .1)
#' @param eps Very small number (Default: 1e-04)
#' @param daily Same as delta in VHAR type (Default: 1 as Litterman)
#' @param weekly Fill the second part in the first block (Default: 1)
#' @param monthly Fill the third part in the first block (Default: 1)
#' @details 
#' * `set_weight_bvhar()` sets hyperparameters for [bvhar_minnesota()] with VHAR-type Minnesota prior, i.e. BVHAR-L model.
#' @return `set_weight_bvhar()` has different component with `delta` due to its different construction.
#' \describe{
#'   \item{daily}{Vector value assigned for daily weight}
#'   \item{weekly}{Vector value assigned for weekly weight}
#'   \item{monthly}{Vector value assigned for monthly weight}
#' }
#' @references Kim, Y. G., and Baek, C. (2023+). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation.
#' @examples 
#' # BVHAR-L specification---------------------------
#' bvhar_vhar_spec <- set_weight_bvhar(
#'   sigma = c(.03, .02, .01), # Sigma = diag(.03^2, .02^2, .01^2)
#'   lambda = .2, # lambda = .2
#'   eps = 1e-04, # eps = 1e-04
#'   daily = rep(.2, 3), # daily1 = .2, daily2 = .2, daily3 = .2
#'   weekly = rep(.1, 3), # weekly1 = .1, weekly2 = .1, weekly3 = .1
#'   monthly = rep(.05, 3) # monthly1 = .05, monthly2 = .05, monthly3 = .05
#' )
#' class(bvhar_vhar_spec)
#' str(bvhar_vhar_spec)
#' @order 1
#' @export
set_weight_bvhar <- function(sigma,
                             lambda = .1,
                             eps = 1e-04,
                             daily,
                             weekly,
                             monthly) {
  if (missing(sigma)) {
    sigma <- NULL
  }
  if (missing(daily)) {
    daily <- NULL
  }
  if (missing(weekly)) {
    weekly <- NULL
  }
  if (missing(monthly)) {
    monthly <- NULL
  }
  if (length(sigma) > 0) {
    if (length(daily) > 0) {
      if (length(sigma) != length(daily)) {
        stop("Length of 'sigma' and 'daily' must be the same as the dimension of the time series.")
      }
    }
    if (length(weekly) > 0) {
      if (length(sigma) != length(weekly)) {
        stop("Length of 'sigma' and 'weekly' must be the same as the dimension of the time series.")
      }
    }
    if (length(monthly) > 0) {
      if (length(sigma) != length(monthly)) {
        stop("Length of 'sigma' and 'monthly' must be the same as the dimension of the time series.")
      }
    }
  }
  bvhar_param <- list(
    process = "BVHAR",
    prior = "MN_VHAR",
    sigma = sigma,
    lambda = lambda,
    eps = eps,
    daily = daily,
    weekly = weekly,
    monthly = monthly
  )
  class(bvhar_param) <- "bvharspec"
  bvhar_param
}

#' Stochastic Search Variable Selection (SSVS) Hyperparameter for Coefficients Matrix and Cholesky Factor
#' 
#' Set SSVS hyperparameters for VAR or VHAR coefficient matrix and Cholesky factor.
#' 
#' @param coef_spike Standard deviance for Spike normal distribution (See Details).
#' @param coef_slab Standard deviance for Slab normal distribution (See Details).
#' @param coef_mixture Bernoulli parameter for sparsity proportion (See Details).
#' @param mean_non Prior mean of unrestricted coefficients
#' @param sd_non Standard deviance for unrestricted coefficients
#' @param shape Gamma shape parameters for precision matrix (See Details).
#' @param rate Gamma rate parameters for precision matrix (See Details).
#' @param chol_spike Standard deviance for Spike normal distribution, in the cholesky factor (See Details).
#' @param chol_slab Standard deviance for Slab normal distribution, in the cholesky factor (See Details).
#' @param chol_mixture Bernoulli parameter for sparsity proportion, in the cholesky factor (See Details).
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
#' * `coef_spike`: \eqn{\tau_{0j}}
#' * `coef_slab`: \eqn{\tau_{1j}}
#' * `coef_mixture`: \eqn{p_j}
#' * \eqn{j = 1, \ldots, mk}: vectorized format corresponding to coefficient matrix
#' * If one value is provided, model function will read it by replicated value.
#' * `coef_non`: vectorized constant term is given prior Normal distribution with variance \eqn{cI}. Here, `coef_non` is \eqn{\sqrt{c}}.
#' 
#' Next for precision matrix \eqn{\Sigma_e^{-1}}, SSVS applies Cholesky decomposition.
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
#' * `chol_spike`: \eqn{\kappa_{0,ij}}
#' * `chol_slab`: \eqn{\kappa_{1,ij}}
#' * `chol_mixture`: \eqn{q_{ij}}
#' * \eqn{j = 1, \ldots, mk}: vectorized format corresponding to coefficient matrix
#' * \eqn{i = 1, \ldots, j - 1} and \eqn{j = 2, \ldots, m}: \eqn{\eta = (\psi_{12}, \psi_{13}, \psi_{23}, \psi_{14}, \ldots, \psi_{34}, \ldots, \psi_{1m}, \ldots, \psi_{m - 1, m})^T}
#' * `chol_` arguments can be one value for replication, vector, or upper triangular matrix.
#' @return `ssvsinput` object
#' @references 
#' George, E. I., & McCulloch, R. E. (1993). *Variable Selection via Gibbs Sampling*. Journal of the American Statistical Association, 88(423), 881–889.
#' 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580.
#' 
#' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358.
#' @order 1
#' @export
set_ssvs <- function(coef_spike = .1, 
                     coef_slab = 5, 
                     coef_mixture = .5,
                     mean_non = 0,
                     sd_non = .1,
                     shape = .01,
                     rate = .01,
                     chol_spike = .1,
                     chol_slab = 5,
                     chol_mixture = .5) {
  if (!(is.vector(coef_spike) && 
        is.vector(coef_slab) && 
        is.vector(coef_mixture) &&
        is.vector(shape) &&
        is.vector(rate) &&
        is.vector(mean_non))) {
    stop("'coef_spike', 'coef_slab', 'coef_mixture', 'shape', 'rate', and 'mean_non' be a vector.")
  }
  if (length(sd_non) != 1) {
    stop("'sd_non' should be length 1 numeric.")
  }
  if (sd_non < 0) {
    stop("'sd_non' should be positive.")
  }
  if (!(is.numeric(chol_spike) ||
        is.vector(chol_spike) || 
        is.matrix(chol_spike) ||
        is.numeric(chol_slab) ||
        is.vector(chol_slab) ||
        is.matrix(chol_slab) ||
        is.numeric(chol_mixture) ||
        is.vector(chol_mixture) ||
        is.matrix(chol_mixture))) {
    stop("'chol_spike', 'chol_slab', and 'chol_mixture' should be a vector or upper triangular matrix.")
  }
  # coefficients---------------------
  coef_param <- list(
    coef_spike = coef_spike,
    coef_slab = coef_slab,
    coef_mixture = coef_mixture
  )
  non_param <- list(
    mean_non = mean_non,
    sd_non = sd_non
  )
  len_param <- sapply(coef_param, length)
  if (length(unique(len_param[len_param != 1])) > 1) {
    stop("The length of 'coef_spike', 'coef_slab', and 'coef_mixture' should be the same.")
  }
  res <- append(coef_param, non_param)
  # cholesky factor-------------------
  if (is.matrix(chol_spike)) {
    if (any(chol_spike[lower.tri(chol_spike, diag = TRUE)] != 0)) {
      stop("If 'chol_spike' is a matrix, it should be an upper triangular form.")
    }
    chol_spike <- chol_spike[upper.tri(chol_spike, diag = FALSE)]
  }
  if (is.matrix(chol_slab)) {
    if (any(chol_slab[lower.tri(chol_slab, diag = TRUE)] != 0)) {
      stop("If 'chol_slab' is a matrix, it should be an upper triangular form.")
    }
    chol_slab <- chol_slab[upper.tri(chol_slab, diag = FALSE)]
  }
  if (is.matrix(chol_mixture)) {
    if (any(chol_mixture[lower.tri(chol_mixture, diag = TRUE)] != 0)) {
      stop("If 'chol_mixture' is a matrix, it should be an upper triangular form.")
    }
    chol_mixture <- chol_mixture[upper.tri(chol_mixture, diag = FALSE)]
  }
  chol_param <- list(
    shape = shape,
    rate = rate,
    chol_spike = chol_spike, 
    chol_slab = chol_slab,
    chol_mixture = chol_mixture,
    process = "VAR",
    prior = "SSVS"
  )
  len_param <- sapply(chol_param, length)
  len_gamma <- len_param[1:2]
  len_eta <- len_param[3:5]
  if (length(unique(len_gamma[len_gamma != 1])) > 1) {
    stop("The length of 'shape' and 'rate' should be the same.")
  }
  if (length(unique(len_eta[len_eta != 1])) > 1) {
    stop("The size of 'chol_spike', 'chol_slab', and 'chol_mixture' should be the same.")
  }
  res <- append(res, chol_param)
  class(res) <- "ssvsinput"
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
    if ((length(dim(init_coef)) == 3 || is.list(init_coef)) &&
        (length(dim(init_coef_dummy)) == 3 || is.list(init_coef_dummy)) &&
        (length(dim(init_chol)) == 3 || is.list(init_chol)) &&
        (length(dim(init_chol_dummy)) == 3 || is.list(init_chol_dummy))) {
      # 3d array to list--------------------------------
      init_coef <- change_to_list(init_coef)
      init_coef_dummy <- change_to_list(init_coef_dummy)
      init_chol <- change_to_list(init_chol)
      init_chol_dummy <- change_to_list(init_chol_dummy)
      # Errors in multiple chain------------------------
      if (length(
        unique(c(length(init_coef), length(init_coef_dummy), length(init_chol), length(init_chol_dummy)))
      ) > 1) {
        stop("Different chain(>1) number has been defined.")
      }
      isnot_identical(init_coef, case = "dim")
      isnot_identical(init_coef_dummy, case = "dim")
      isnot_identical(init_chol, case = "dim")
      isnot_identical(init_chol_dummy, case = "dim")
      isnot_identical(init_coef, case = "values")
      isnot_identical(init_chol, case = "values")
      num_chain <- length(init_coef)
      coef_mat <- init_coef[[1]]
      coef_dummy <- init_coef_dummy[[1]]
      chol_mat <- init_chol[[1]]
      chol_dummy <- init_chol_dummy[[1]]
    } else {
      num_chain <- 1
      coef_mat <- init_coef
      coef_dummy <- init_coef_dummy
      chol_mat <- init_chol
      chol_dummy <- init_chol_dummy
    }
    # Check dimension validity-----------------------------
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
    global_sparsity = global_sparsity#,init_cov = init_cov
  )
  class(res) <- "horseshoespec"
  res
}

#' Stochastic Volatility Specification
#' 
#' `r lifecycle::badge("experimental")` Set SV hyperparameters.
#' 
#' @param ig_shape Inverse-Gamma shape of state variance.
#' @param ig_scl Inverse-Gamma scale of state variance.
#' @param initial_mean Prior mean of initial state.
#' @param initial_prec Prior precision of initial state.
#' @references
#' Carriero, A., Chan, J., Clark, T. E., & Marcellino, M. (2022). *Corrigendum to “Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors” \[J. Econometrics 212 (1)(2019) 137–154\]*. Journal of Econometrics, 227(2), 506-512.
#'
#' Chan, J., Koop, G., Poirier, D., & Tobias, J. (2019). *Bayesian Econometric Methods (2nd ed., Econometric Exercises)*. Cambridge: Cambridge University Press.
#' @order 1
#' @export
set_sv <- function(ig_shape = 3, ig_scl = .01, initial_mean = 1, initial_prec = .1) {
  if (!is.vector(ig_shape) ||
    !is.vector(ig_scl) ||
    !is.vector(initial_mean)) {
    stop("'ig_shape', 'ig_scl', and 'initial_mean' should be a vector.")
  }
  if ((length(ig_shape) != length(ig_scl)) ||
    (length(ig_scl) != length(initial_mean))) {
    stop("'ig_shape', 'ig_scl', and 'initial_mean' should have same length.")
  }
  if (is.vector(initial_prec) && length(initial_prec) > 1) {
    initial_prec <- diag(initial_prec)
  }
  if (is.matrix(initial_prec)) {
    if ((length(ig_shape) != nrow(initial_prec))
        || (length(ig_shape) != ncol(initial_prec))) {
      stop("'initial_prec' should be symmetric matrix of same size with the other vectors.")
    }
  }
  res <- list(
    process = "SV",
    prior = "Cholesky",
    shape = ig_shape,
    scale = ig_scl,
    initial_mean = initial_mean,
    initial_prec = initial_prec
  )
  class(res) <- "svspec"
  res
}
