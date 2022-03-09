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
        "AIC" = compute_aic(var_list),
        "BIC" = compute_bic(var_list),
        "HQ" = compute_hq(var_list),
        "FPE" = compute_fpe(var_list)
      )
    }
  } else {
    res <- foreach(p = 1:lag_max, .combine = rbind) %do% {
      var_list <- var_lm(y, p, include_mean = include_mean)
      c(
        "AIC" = compute_aic(var_list),
        "BIC" = compute_bic(var_list),
        "HQ" = compute_hq(var_list),
        "FPE" = compute_hq(var_list)
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
#' @param bayes_spec `r lifecycle::badge("experimental")` Initial Bayes model specification.
#' @param lower `r lifecycle::badge("experimental")` Lower bound. By default, `.01`.
#' @param upper `r lifecycle::badge("experimental")` Upper bound. By default, `10`.
#' @param eps Hyperparameter `eps` is fixed. By default, `1e-04`.
#' @param ... Additional arguments for [stats::optim()].
#' @param y Time series data
#' @param p BVAR lag
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param parallel `r lifecycle::badge("experimental")` List the same argument of [optimParallel::optimParallel()]. By default, this is empty, and the function does not execute parallel computation.
#' @details 
#' Empirical Bayes method maximizes marginal likelihood and selects the set of hyperparameters.
#' These functions implement `"L-BFGS-B"` method of [stats::optim()] to find the maximum of marginal likelihood.
#' 
#' @references 
#' Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). *Bayesian data analysis*. Chapman and Hall/CRC. [http://www.stat.columbia.edu/~gelman/book/](http://www.stat.columbia.edu/~gelman/book/)
#' 
#' Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). *A limited memory algorithm for bound constrained optimization*. SIAM Journal on scientific computing, 16(5), 1190-1208. doi: [10.1137/0916069](https://doi.org/10.1137/0916069).
#' 
#' @importFrom stats optim
#' @importFrom optimParallel optimParallel
#' @order 1
#' @export
choose_bvar <- function(bayes_spec = set_bvar(), 
                        lower = .01, 
                        upper = 10, 
                        ..., 
                        eps = 1e-04,
                        y, 
                        p, 
                        include_mean = TRUE,
                        parallel = list()) {
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
  # find argmax of log(ML)-------
  if (length(parallel) > 0) {
    res <- 
      optimParallel(
        par = c(sigma, lambda, delta), 
        fn = logml_bvar,
        lower = lower,
        upper = upper,
        ...,
        eps = eps,
        y = y,
        p = p,
        include_mean = include_mean,
        parallel = parallel
      )
  } else {
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
  }
  # optimized model spec---------
  bayes_spec$sigma <- res$par[1:dim_data]
  bayes_spec$lambda <- res$par[dim_data + 1]
  bayes_spec$delta <- res$par[(dim_data + 2):(dim_data * 2 + 1)]
  bayes_spec$eps <- eps
  res$spec <- bayes_spec
  # fit using final spec--------
  final_fit <- bvar_minnesota(
    y = y,
    p = p,
    bayes_spec = bayes_spec,
    include_mean = include_mean
  )
  res$fit <- final_fit
  # compute log(ML) of the fit--
  prior_shape <- final_fit$prior_shape # alpha0
  num_obs <- final_fit$obs # s
  const_term <- - dim_data * num_obs / 2 * log(pi) + 
    log_mgammafn((prior_shape + num_obs) / 2, dim_data) - 
    log_mgammafn(prior_shape / 2, dim_data)
  res$ml <- const_term - res$value
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
logml_bvhar_var <- function(param, eps = 1e-04, y, har = c(5, 22), include_mean = TRUE, ...) {
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
  fit <- bvhar_minnesota(y = y, har = har, bayes_spec = bvhar_spec, include_mean = include_mean)
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
logml_bvhar_vhar <- function(param, eps = 1e-04, y, har = c(5, 22), include_mean = TRUE, ...) {
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
  fit <- bvhar_minnesota(y = y, har = har, bayes_spec = bvhar_spec, include_mean = include_mean)
  -logml_stable(fit) # for maximization
}

#' @rdname choose_bvar
#' 
#' @param bayes_spec `r lifecycle::badge("experimental")` Initial Bayes model specification.
#' @param lower `r lifecycle::badge("experimental")` Lower bound. By default, `.01`.
#' @param upper `r lifecycle::badge("experimental")` Upper bound. By default, `10`.
#' @param eps Hyperparameter `eps` is fixed. By default, `1e-04`.
#' @param ... Additional arguments for [stats::optim()].
#' @param y Time series data
#' @param har `r lifecycle::badge("experimental")` Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param parallel `r lifecycle::badge("experimental")` List the same argument of [optimParallel::optimParallel()]. By default, this is empty, and the function does not execute parallel computation.
#' 
#' @importFrom stats optim
#' @importFrom optimParallel optimParallel
#' @order 1
#' @export
choose_bvhar <- function(bayes_spec = set_bvhar(),
                         lower = .01, 
                         upper = 10, 
                         ..., 
                         eps = 1e-04,
                         y, 
                         har = c(5, 22),
                         include_mean = TRUE,
                         parallel = list()) {
  dim_data <- ncol(y)
  if (!is.bvharspec(bayes_spec)) {
    stop("Provide 'bvharspec' for 'bayes_spec'.")
  }
  if (bayes_spec$process != "BVHAR") {
    stop("'bayes_spec' must be the result of 'set_bvhar()' or 'set_weight_bvhar()'.")
  }
  # sigma------------------------
  if (is.null(bayes_spec$sigma)) {
    sigma <- apply(y, 2, sd)
  } else {
    sigma <- bayes_spec$sigma
  }
  # lambda-----------------------
  lambda <- bayes_spec$lambda
  # find argmax of log(ML)-------
  # for each type
  if (bayes_spec$prior == "MN_VAR") {
    if (is.null(bayes_spec$delta)) {
      delta <- rep(.1, dim_data)
    } else {
      delta <- bayes_spec$delta
    }
    # maximize marginal likelihood
    if (length(parallel) > 0) {
      res <- 
        optimParallel(
          par = c(sigma, lambda, delta), 
          fn = logml_bvhar_var,
          lower = lower,
          upper = upper,
          ...,
          eps = eps,
          y = y,
          har = har,
          include_mean = include_mean,
          parallel = parallel
        )
    } else {
      res <- 
        optim(
          par = c(sigma, lambda, delta), 
          fn = logml_bvhar_var,
          method = "L-BFGS-B",
          lower = lower,
          upper = upper,
          ...,
          eps = eps,
          y = y,
          har = har,
          include_mean = include_mean
        )
    }
    # collect the argmax-----------
    bayes_spec$sigma <- res$par[1:dim_data]
    bayes_spec$lambda <- res$par[dim_data + 1]
    bayes_spec$delta <- res$par[(dim_data + 2):(dim_data * 2 + 1)]
  } else {
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
    # maximize marginal likelihood
    if (length(parallel) > 0) {
      res <- 
        optimParallel(
          par = c(sigma, lambda, daily, weekly, monthly), 
          fn = logml_bvhar_vhar,
          lower = lower,
          upper = upper,
          ...,
          eps = eps,
          y = y,
          har = har,
          include_mean = include_mean,
          parallel = parallel
        )
    } else {
      res <- 
        optim(
          par = c(sigma, lambda, daily, weekly, monthly), 
          fn = logml_bvhar_vhar,
          method = "L-BFGS-B",
          lower = lower,
          upper = upper,
          ...,
          eps = eps,
          y = y,
          har = har,
          include_mean = include_mean
        )
    }
    # collect the argmax-----------
    bayes_spec$sigma <- res$par[1:dim_data]
    bayes_spec$lambda <- res$par[dim_data + 1]
    bayes_spec$daily <- res$par[(dim_data + 2):(dim_data * 2 + 1)]
    bayes_spec$weekly <- res$par[(dim_data * 2 + 2):((dim_data * 3 + 1))]
    bayes_spec$monthly <- res$par[(dim_data * 3 + 2):((dim_data * 4 + 1))]
  }
  bayes_spec$eps <- eps
  res$spec <- bayes_spec
  # fit using final spec--------
  final_fit <- bvhar_minnesota(
    y = y,
    har = har,
    bayes_spec = bayes_spec,
    include_mean = include_mean
  )
  res$fit <- final_fit
  # compute log(ML) of the fit--
  prior_shape <- final_fit$prior_shape # alpha0
  num_obs <- final_fit$obs # s
  const_term <- - dim_data * num_obs / 2 * log(pi) + 
    log_mgammafn((prior_shape + num_obs) / 2, dim_data) - 
    log_mgammafn(prior_shape / 2, dim_data)
  res$ml <- const_term - res$value
  class(res) <- "bvharemp"
  res
}
