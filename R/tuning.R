#' Choose the Best VAR based on Information Criteria
#' 
#' This function computes AIC, FPE, BIC, and HQ up to p = `lag_max` of VAR model.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param lag_max Maximum Var lag to explore (default = 5)
#' @param include_mean `r lifecycle::badge("experimental")` Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param parallel Parallel computation using [foreach::foreach()]? By default, `FALSE`.
#' 
#' @return Minimum order and information criteria values
#' 
#' @importFrom foreach foreach %do% %dopar%
#' @export
choose_var <- function(y, lag_max = 5, include_mean = TRUE, parallel = FALSE) {
  if (!all(apply(y, 2, is.numeric))) stop("Every column must be numeric class.")
  if (!is.matrix(y)) y <- as.matrix(y)
  var_list <- NULL
  if (!is.logical(include_mean)) stop("'include_mean' is logical.")
  # compute IC-----------------------
  if (parallel) {
    res <- foreach(p = 1:lag_max, .combine = rbind) %dopar% {
      var_list <- var_lm(y, p, include_mean = include_mean)
      c(
        "AIC" = AIC(var_list, type = "rss"),
        "BIC" = BIC(var_list, type = "rss"),
        "HQ" = HQ(var_list, type = "rss"),
        "FPE" = FPE(var_list)
      )
    }
  } else {
    res <- foreach(p = 1:lag_max, .combine = rbind) %do% {
      var_list <- var_lm(y, p, include_mean = include_mean)
      c(
        "AIC" = AIC(var_list, type = "rss"),
        "BIC" = BIC(var_list, type = "rss"),
        "HQ" = HQ(var_list, type = "rss"),
        "FPE" = FPE(var_list)
      )
    }
  }
  rownames(res) <- 1:lag_max
  # find minimum-----------------------
  list(
    ic = res,
    min_lag = apply(res, 2, which.min)
  )
}

#' Log ML Function to be in `optim`
#' 
#' This function is for `fn` of [stats::optim()].
#' 
#' @param param Vector of hyperparameters
#' @param y Time series data
#' @param p BVAR lag
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param ... not used
#' @details 
#' `par` is the collection of hyperparameters except `eps`.
#' * `sigma`
#' * `lambda`
#' * `delta`
#' in order.
#' 
#' @noRd
logml_bvar <- function(param, y, p, include_mean = TRUE, ...) {
  dim_data <- ncol(y)
  if (length(param) != 2 * dim_data + 1) {
    stop("The number of param is wrong.")
  }
  bvar_spec <- set_bvar(
    sigma = param[1:dim_data],
    lambda = param[dim_data + 1],
    delta = param[(dim_data + 2):length(param)]
  )
  fit <- bvar_minnesota(y = y, p = p, bayes_spec = bvar_spec, include_mean = include_mean)
  -compute_logml(fit) # for maximization
}

#' Finding the Set of Hyperparameters of Bayesian Model
#' 
#' This function chooses the set of hyperparameters of Bayesian model using [stats::optim()] function.
#' 
#' @param sigma Initial vector for `sigma` (Default: `apply(y, 2, sd)`)
#' @param lambda Initial value for `lambda` (Default: `.1`)
#' @param delta Initial vector for `delta` (Default: `rep(0, ncol(y))`)
#' @param lower Lower bound. By default, `-Inf`.
#' @param upper Upper bound. By default, `Inf`.
#' @param ... Additional arguments for [stats::optim()].
#' @param y Time series data
#' @param p BVAR lag
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' 
#' @importFrom stats optim
#' @export
choose_bvar <- function(sigma, 
                        lambda = .1, 
                        delta, 
                        lower, 
                        upper, 
                        ..., 
                        y, 
                        p, 
                        include_mean = TRUE) {
  if (missing(sigma)) {
    sigma <- apply(y, 2, sd)
  }
  if (missing(delta)) {
    delta <- rep(0, ncol(y))
  }
  optim(
    par = c(sigma, lambda, delta), 
    fn = logml_bvar,
    method = "L-BFGS-B",
    lower = lower,
    upper = upper,
    ...,
    y = y,
    p = p,
    include_mean = include_mean
  )
}
