#' Log ML Function of Hierarchical BVHAR-S to be in `optim`
#'
#' `r lifecycle::badge("experimental")` This function is for `fn` of [stats::optim()].
#'
#' @param param Vector of hyperparameter settings for [set_lambda()] and [set_psi()] in order.
#' @param delta delta in BVHAR-S specification
#' @param y Time series data
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param ... not used
#' @details
#' `par` is the collection of hyperparameters.
#' * `lambda`
#' * `psi`
#' in order.
#' @noRd
logml_bvharhm <- function(param, delta, eps = 1e-04, y, har = c(5, 22), include_mean = TRUE, ...) {
  dim_data <- ncol(y)
  if (length(param) != dim_data + 1) {
    stop("The number of param is wrong.")
  }
  bvhar_spec <- set_bvhar(
    sigma = param[2:(dim_data + 1)],
    lambda = param[1],
    delta = delta,
    eps = eps
  )
  fit <- bvhar_minnesota(y = y, har = har, bayes_spec = bvhar_spec, include_mean = include_mean)
  -logml_stable(fit)
}

#' Log ML Function of Hierarchical BVHAR-L to be in `optim`
#'
#' `r lifecycle::badge("experimental")` This function is for `fn` of [stats::optim()].
#'
#' @param param Vector of hyperparameter settings for [set_lambda()] and [set_psi()] in order.
#' @param daily daily in BVHAR-L specification
#' @param weekly weekly in BVHAR-L specification
#' @param monthly monthly in BVHAR-L specification
#' @param y Time series data
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param ... not used
#' @details
#' `par` is the collection of hyperparameters.
#' * `lambda`
#' * `psi`
#' in order.
#' @noRd
logml_bvharlhm <- function(param, daily, weekly, monthly, eps = 1e-04, y, har = c(5, 22), include_mean = TRUE, ...) {
  dim_data <- ncol(y)
  if (length(param) != dim_data + 1) {
    stop("The number of param is wrong.")
  }
  bvhar_spec <- set_weight_bvhar(
    sigma = param[2:(dim_data + 1)],
    lambda = param[1],
    eps = eps,
    daily = daily,
    weekly = weekly,
    monthly = monthly
  )
  fit <- bvhar_minnesota(y = y, har = har, bayes_spec = bvhar_spec, include_mean = include_mean)
  -logml_stable(fit)
}
