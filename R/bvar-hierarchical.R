#' Hyperpriors for Bayesian Models
#' 
#' Set hyperpriors of Bayesian VAR and VHAR models.
#' 
#' @param mode Mode of Gamma distribution. By default, `.2`.
#' @param sd Standard deviation of Gamma distribution. By default, `.4`.
#' @param lower `r lifecycle::badge("experimental")` Lower bound for [stats::optim()]. By default, `1e-5`.
#' @param upper `r lifecycle::badge("experimental")` Upper bound for [stats::optim()]. By default, `3`.
#' @details 
#' In addition to Normal-IW priors [set_bvar()], [set_bvhar()], and [set_weight_bvhar()],
#' these functions give hierarchical structure to the model.
#' * `set_lambda()` specifies hyperprior for \eqn{\lambda} (`lambda`), which is Gamma distribution.
#' * `set_psi()` specifies hyperprior for \eqn{\psi / (\nu_0 - k - 1) = \sigma^2} (`sigma`), which is Inverse gamma distribution.
#' @examples 
#' # Hirearchical BVAR specification------------------------
#' set_bvar(
#'   sigma = set_psi(shape = 4e-4, scale = 4e-4),
#'   lambda = set_lambda(mode = .2, sd = .4),
#'   delta = rep(1, 3),
#'   eps = 1e-04 # eps = 1e-04
#' )
#' @return `bvharpriorspec` object
#' @references Giannone, D., Lenza, M., & Primiceri, G. E. (2015). *Prior Selection for Vector Autoregressions*. Review of Economics and Statistics, 97(2).
#' @order 1
#' @export
set_lambda <- function(mode = .2, sd = .4, lower = 1e-5, upper = 3) {
  params <- get_gammaparam(mode, sd)
  lam_prior <- list(
    hyperparam = "lambda",
    param = c(params$shape, params$rate),
    mode = mode,
    lower = lower,
    upper = upper
  )
  class(lam_prior) <- "bvharpriorspec"
  lam_prior
}

#' @rdname set_lambda
#' @param shape Shape of Inverse Gamma distribution. By default, `(.02)^2`.
#' @param scale Scale of Inverse Gamma distribution. By default, `(.02)^2`.
#' @param lower `r lifecycle::badge("experimental")` Lower bound for [stats::optim()]. By default, `1e-5`.
#' @param upper `r lifecycle::badge("experimental")` Upper bound for [stats::optim()]. By default, `3`.
#' @details 
#' The following set of `(mode, sd)` are recommended by Sims and Zha (1998) for `set_lambda()`.
#' * `(mode = .2, sd = .4)`: default
#' * `(mode = 1, sd = 1)`
#' 
#' Giannone et al. (2015) suggested data-based selection for `set_psi()`.
#' It chooses (0.02)^2 based on its empirical data set.
#' @order 1
#' @export
set_psi <- function(shape = 4e-4, scale = 4e-4, lower = 1e-5, upper = 3) {
  psi_prior <- list(
    hyperparam = "psi",
    param = c(shape, scale),
    mode = scale / (shape + 1),
    lower = lower,
    upper = upper
  )
  class(psi_prior) <- "bvharpriorspec"
  psi_prior
}

#' Log ML Function of Hierarchical BVAR to be in `optim`
#' 
#' This function is for `fn` of [stats::optim()].
#' 
#' @param param Vector of hyperparameter settings for [set_lambda()] and [set_psi()] in order.
#' @param delta delta in BVAR specification
#' @param y Time series data
#' @param p BVAR lag
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param ... not used
#' @details 
#' `par` is the collection of hyperparameters.
#' * `lambda`
#' * `psi`
#' in order.
#' @noRd
logml_bvarhm <- function(param, delta, eps = 1e-04, y, p, include_mean = TRUE, ...) {
  dim_data <- ncol(y)
  if (length(param) != dim_data + 1) {
    stop("The number of 'param' is wrong.")
  }
  bvar_spec <- set_bvar(
    sigma = param[2:(dim_data + 1)],
    lambda = param[1],
    delta = delta,
    eps = eps
  )
  fit <- bvar_minnesota(y = y, p = p, bayes_spec = bvar_spec, include_mean = include_mean)
  -logml_stable(fit)
}
