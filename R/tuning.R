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
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  var_list <- NULL
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
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

#' Log ML Function of BVAR to be in `optim`
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
logml_bvar <- function(param, eps = 1e-04, y, p, include_mean = TRUE, ...) {
  dim_data <- ncol(y)
  if (length(param) != 2 * dim_data + 1) {
    stop("The number of param is wrong.")
  }
  bvar_spec <- set_bvar(
    sigma = param[1:dim_data],
    lambda = param[dim_data + 1],
    delta = param[(dim_data + 2):length(param)],
    eps = eps
  )
  fit <- bvar_minnesota(y = y, p = p, bayes_spec = bvar_spec, include_mean = include_mean)
  -logml_stable(fit) # for maximization
}

#' Finding the Set of Hyperparameters of Bayesian Model
#' 
#' This function chooses the set of hyperparameters of Bayesian model using [stats::optim()] function.
#' 
#' @param bayes_spec `r lifecycle::badge("experimental")` A Bayes model specification.
#' @param lower Lower bound. By default, `-Inf`.
#' @param upper Upper bound. By default, `Inf`.
#' @param eps Hyperparameter `eps` is fixed. By default, `1e-04`.
#' @param ... Additional arguments for [stats::optim()].
#' @param y Time series data
#' @param p BVAR lag
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @details 
#' Empirical Bayes method maximizes marginal likelihood and selects the set of hyperparameters.
#' 
#' @importFrom stats optim
#' @order 1
#' @export
choose_bvar <- function(bayes_spec = set_bvar(), 
                        lower = .01, 
                        upper = Inf, 
                        ..., 
                        eps = 1e-04,
                        y, 
                        p, 
                        include_mean = TRUE) {
  dim_data <- ncol(y)
  if (!is.bvharspec(bayes_spec)) {
    stop("Provide 'bvharspec' for 'bayes_spec'.")
  }
  if (bayes_spec$process != "BVAR") {
    stop("'bayes_spec' must be the result of 'set_bvar()'.")
  }
  # sigma------------------------
  if (is.null(bayes_spec$sigma)) {
    sigma <- apply(y, 2, sd)
  } else {
    sigma <- bayes_spec$sigma
  }
  # lambda-----------------------
  lambda <- bayes_spec$lambda
  # delta------------------------
  if (is.null(bayes_spec$delta)) {
    delta <- rep(.1, dim_data)
  } else {
    delta <- bayes_spec$delta
  }
  res <- 
    optim(
      par = c(sigma, lambda, delta), 
      fn = logml_bvar,
      method = "L-BFGS-B",
      lower = lower,
      upper = upper,
      ...,
      eps = eps,
      y = y,
      p = p,
      include_mean = include_mean
    )
  bayes_spec$sigma <- res$par[1:dim_data]
  bayes_spec$lambda <- res$par[dim_data + 1]
  bayes_spec$delta <- res$par[(dim_data + 2):(dim_data * 2 + 1)]
  res$spec <- bayes_spec
  res$fit <- bvar_minnesota(
    y = y,
    p = p,
    bayes_spec = bayes_spec,
    include_mean = include_mean
  )
  res$ml <- compute_logml(res$fit)
  class(res) <- "bvharemp"
  res
}

#' Log ML Function of VAR-type BVHAR to be in `optim`
#' 
#' This function is for `fn` of [stats::optim()].
#' 
#' @param param Vector of hyperparameters
#' @param y Time series data
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
logml_bvhar_var <- function(param, eps = 1e-04, y, include_mean = TRUE, ...) {
  dim_data <- ncol(y)
  if (length(param) != 2 * dim_data + 1) {
    stop("The number of param is wrong.")
  }
  bvhar_spec <- set_bvhar(
    sigma = param[1:dim_data],
    lambda = param[dim_data + 1],
    delta = param[(dim_data + 2):length(param)],
    eps = 1e-04
  )
  fit <- bvhar_minnesota(y = y, bayes_spec = bvhar_spec, include_mean = include_mean)
  -logml_stable(fit) # for maximization
}

#' Log ML Function of VHAR-type BVHAR to be in `optim`
#' 
#' This function is for `fn` of [stats::optim()].
#' 
#' @param param Vector of hyperparameters
#' @param y Time series data
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param ... not used
#' @details 
#' `par` is the collection of hyperparameters except `eps`.
#' * `sigma`
#' * `lambda`
#' * `daily`
#' * `weekly`
#' * `monthly`
#' in order.
#' 
#' @noRd
logml_bvhar_vhar <- function(param, eps = 1e-04, y, include_mean = TRUE, ...) {
  dim_data <- ncol(y)
  if (length(param) != 4 * dim_data + 1) {
    stop("The number of param is wrong.")
  }
  bvhar_spec <- set_weight_bvhar(
    sigma = param[1:dim_data],
    lambda = param[dim_data + 1],
    eps = eps,
    daily = param[(dim_data + 2):((dim_data * 2 + 1))],
    weekly = param[(dim_data * 2 + 2):((dim_data * 3 + 1))],
    monthly = param[(dim_data * 3 + 2):((dim_data * 4 + 1))]
  )
  fit <- bvhar_minnesota(y = y, bayes_spec = bvhar_spec, include_mean = include_mean)
  -logml_stable(fit) # for maximization
}

#' @rdname choose_bvar
#' 
#' @param bayes_spec `r lifecycle::badge("experimental")` A Bayes model specification.
#' @param lower Lower bound. By default, `-Inf`.
#' @param upper Upper bound. By default, `Inf`.
#' @param eps Hyperparameter `eps` is fixed. By default, `1e-04`.
#' @param ... Additional arguments for [stats::optim()].
#' @param y Time series data
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' 
#' @importFrom stats optim
#' @order 1
#' @export
choose_bvhar <- function(bayes_spec = set_bvhar(),
                         lower = .01, 
                         upper = Inf, 
                         ..., 
                         eps = 1e-04,
                         y, 
                         include_mean = TRUE) {
  dim_data <- ncol(y)
  if (!is.bvharspec(bayes_spec)) {
    stop("Provide 'bvharspec' for 'bayes_spec'.")
  }
  if (bayes_spec$process != "BVHAR") {
    stop("'bayes_spec' must be the result of 'set_bvhar()' or 'set_weight_bvhar()'.")
  }
  # sigma-------------------
  if (is.null(bayes_spec$sigma)) {
    sigma <- apply(y, 2, sd)
  } else {
    sigma <- bayes_spec$sigma
  }
  # lambda------------------
  lambda <- bayes_spec$lambda
  minnesota_type <- bayes_spec$prior
  # for each type-----------
  res <- 
    switch(
      minnesota_type,
      "MN_VAR" = {
        if (is.null(bayes_spec$delta)) {
          delta <- rep(.1, dim_data)
        } else {
          delta <- bayes_spec$delta
        }
        optim(
          par = c(sigma, lambda, delta), 
          fn = logml_bvhar_var,
          method = "L-BFGS-B",
          lower = lower,
          upper = upper,
          ...,
          eps = eps,
          y = y,
          include_mean = include_mean
        )
      },
      "MN_VHAR" = {
        if (is.null(bayes_spec$daily)) {
          daily <- rep(.1, dim_data)
        } else {
          daily <- bayes_spec$daily
        }
        if (is.null(bayes_spec$weekly)) {
          weekly <- rep(.1, dim_data)
        } else {
          weekly <- bayes_spec$weekly
        }
        if (is.null(bayes_spec$monthly)) {
          monthly <- rep(.1, dim_data)
        } else {
          monthly <- bayes_spec$monthly
        }
        optim(
          par = c(sigma, lambda, daily, weekly, monthly), 
          fn = logml_bvhar_vhar,
          method = "L-BFGS-B",
          lower = lower,
          upper = upper,
          ...,
          eps = eps,
          y = y,
          include_mean = include_mean
        )
      }
    )
  if (minnesota_type == "MN_VAR") {
    bayes_spec$sigma <- res$par[1:dim_data]
    bayes_spec$lambda <- res$par[dim_data + 1]
    bayes_spec$delta <- res$par[(dim_data + 2):(dim_data * 2 + 1)]
  } else {
    bayes_spec$sigma <- res$par[1:dim_data]
    bayes_spec$lambda <- res$par[dim_data + 1]
    bayes_spec$daily <- res$par[(dim_data + 2):(dim_data * 2 + 1)]
    bayes_spec$weekly <- res$par[(dim_data * 2 + 2):((dim_data * 3 + 1))]
    bayes_spec$monthly <- res$par[(dim_data * 3 + 2):((dim_data * 4 + 1))]
  }
  res$spec <- bayes_spec
  res$fit <- bvhar_minnesota(
    y = y,
    bayes_spec = bayes_spec,
    include_mean = include_mean
  )
  res$ml <- compute_logml(res$fit)
  class(res) <- "bvharemp"
  res
}
