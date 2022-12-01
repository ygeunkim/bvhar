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
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). doi:[10.1002/jae.1137](https://doi:10.1002/jae.1137)
#' 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. doi:[10.2307/1391384](https://doi.org/10.2307/1391384)
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
  hiearchical <- is.bvharpriorspec(sigma)
  if (missing(delta)) {
    delta <- NULL
  }
  if (hiearchical) {
    if (!all(is.bvharpriorspec(sigma) & is.bvharpriorspec(lambda))) {
      stop("When using hiearchical model, each 'sigma' and 'lambda' should be 'bvharpriorspec'.")
    }
    prior_type <- "Hierarchical"
  } else {
    if (lambda <= 0) {
      stop("'lambda' should be larger than 0.")
    }
    if (missing(sigma)) {
      sigma <- NULL
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
#' @references Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). doi:[10.1080/01621459.2018.1437043](https://doi.org/10.1080/01621459.2018.1437043)
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
#' @references Kim, Y. G., and Baek, C. (n.d.). *Bayesian vector heterogeneous autoregressive modeling*. submitted.
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
#' @references Kim, Y. G., and Baek, C. (n.d.). *Bayesian vector heterogeneous autoregressive modeling*. submitted.
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
