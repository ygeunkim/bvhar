#' Choose the Best VAR based on Information Criteria
#' 
#' This function computes AIC, FPE, BIC, and HQ up to p = `lag_max` of VAR model.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param lag_max Maximum Var lag to explore (default = 5)
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param parallel Parallel computation using [foreach::foreach()]? By default, `FALSE`.
#' @return Minimum order and information criteria values
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

#' Finding the Set of Hyperparameters of Individual Bayesian Model
#' 
#' Instead of these functions, you can use [choose_bayes()].
#' 
#' @param bayes_spec Initial Bayes model specification.
#' @param lower `r lifecycle::badge("experimental")` Lower bound. By default, `.01`.
#' @param upper `r lifecycle::badge("experimental")` Upper bound. By default, `10`.
#' @param eps Hyperparameter `eps` is fixed. By default, `1e-04`.
#' @param ... Additional arguments for [stats::optim()].
#' @param y Time series data
#' @param p BVAR lag
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param parallel List the same argument of [optimParallel::optimParallel()]. By default, this is empty, and the function does not execute parallel computation.
#' @details 
#' Empirical Bayes method maximizes marginal likelihood and selects the set of hyperparameters.
#' These functions implement `L-BFGS-B` method of [stats::optim()] to find the maximum of marginal likelihood.
#' 
#' If you want to set `lower` and `upper` option more carefully,
#' deal with them like as in [stats::optim()] in order of [set_bvar()], [set_bvhar()], or [set_weight_bvhar()]'s argument (except `eps`).
#' In other words, just arrange them in a vector.
#' @return `bvharemp` [class] is a list that has
#' * [stats::optim()] or [optimParallel::optimParallel()]
#' * chosen `bvharspec` set
#' * Bayesian model fit result with chosen specification
#' \describe{
#'   \item{...}{Many components of [stats::optim()] or [optimParallel::optimParallel()]}
#'   \item{spec}{Corresponding `bvharspec`}
#'   \item{fit}{Chosen Bayesian model}
#'   \item{ml}{Marginal likelihood of the final model}
#' }
#' @references 
#' Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). *A limited memory algorithm for bound constrained optimization*. SIAM Journal on scientific computing, 16(5), 1190-1208.
#' 
#' Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). *Bayesian data analysis*. Chapman and Hall/CRC.
#' 
#' Giannone, D., Lenza, M., & Primiceri, G. E. (2015). *Prior Selection for Vector Autoregressions*. Review of Economics and Statistics, 97(2).
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
#' @param bayes_spec Initial Bayes model specification.
#' @param lower `r lifecycle::badge("experimental")` Lower bound. By default, `.01`.
#' @param upper `r lifecycle::badge("experimental")` Upper bound. By default, `10`.
#' @param eps Hyperparameter `eps` is fixed. By default, `1e-04`.
#' @param ... Additional arguments for [stats::optim()].
#' @param y Time series data
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param parallel List the same argument of [optimParallel::optimParallel()]. By default, this is empty, and the function does not execute parallel computation.
#' @references Kim, Y. G., and Baek, C. (2024). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation, 94(6), 1139-1157.
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

#' Setting Empirical Bayes Optimization Bounds
#' 
#' `r lifecycle::badge("experimental")` This function sets lower and upper bounds for [set_bvar()], [set_bvhar()], or [set_weight_bvhar()].
#' 
#' @param init_spec Initial Bayes model specification
#' @param lower_spec Lower bound Bayes model specification
#' @param upper_spec Upper bound Bayes model specification
#' @return `boundbvharemp` [class]
#' @order 1
#' @export
bound_bvhar <- function(init_spec = set_bvhar(),
                        lower_spec = set_bvhar(),
                        upper_spec = set_bvhar()) {
  if (!is.bvharspec(init_spec)) {
    stop("Provide 'bvharspec' for 'init_spec'.")
  }
  if (!is.bvharspec(lower_spec)) {
    stop("Provide 'bvharspec' for 'lower_spec'.")
  }
  if (!is.bvharspec(upper_spec)) {
    stop("Provide 'bvharspec' for 'upper_spec'.")
  }
  if (init_spec$prior != lower_spec$prior) {
    stop("'init_spec' and 'lower_spec' should be the same prior.")
  }
  if (lower_spec$prior != upper_spec$prior) {
    stop("'lower_spec' and 'upper_spec' should be the same prior.")
  }
  if (init_spec$prior == "Flat") {
    stop("ML not yet for Flat prior.")
  }
  if (length(init_spec$sigma) != length(lower_spec$sigma)) {
    stop("'lower_spec' has wrong dimension.")
  }
  if (length(init_spec$sigma) != length(upper_spec$sigma)) {
    stop("'upper_spec' has wrong dimension.")
  }
  res <- list(spec = init_spec)
  res$lower <- c(
    lower_spec$sigma,
    lower_spec$lambda
  )
  res$upper <- c(
    upper_spec$sigma,
    upper_spec$lambda
  )
  # delta or daily-weekly-monthly---------------------------
  if (init_spec$prior == "MN_VHAR") {
    res$lower <- c(
      res$lower,
      lower_spec$daily,
      lower_spec$weekly,
      lower_spec$monthly
    )
    res$upper <- c(
      res$upper,
      upper_spec$daily,
      upper_spec$weekly,
      upper_spec$monthly
    )
  } else {
    res$lower <- c(
      res$lower,
      lower_spec$delta
    )
    res$upper <- c(
      res$upper,
      upper_spec$delta
    )
  }
  class(res) <- "boundbvharemp"
  res
}

#' Finding the Set of Hyperparameters of Bayesian Model
#' 
#' `r lifecycle::badge("experimental")` This function chooses the set of hyperparameters of Bayesian model using [stats::optim()] function.
#' 
#' @param bayes_bound Empirical Bayes optimization bound specification defined by [bound_bvhar()].
#' @param ... Additional arguments for [stats::optim()].
#' @param eps Hyperparameter `eps` is fixed. By default, `1e-04`.
#' @param ... Additional arguments for [stats::optim()].
#' @param y Time series data
#' @param order Order for BVAR or BVHAR. `p` of [bvar_minnesota()] or `har` of [bvhar_minnesota()]. By default, `c(5, 22)` for `har`.
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param parallel List the same argument of [optimParallel::optimParallel()]. By default, this is empty, and the function does not execute parallel computation.
#' @return `bvharemp` [class] is a list that has
#' \describe{
#'   \item{...}{Many components of [stats::optim()] or [optimParallel::optimParallel()]}
#'   \item{spec}{Corresponding `bvharspec`}
#'   \item{fit}{Chosen Bayesian model}
#'   \item{ml}{Marginal likelihood of the final model}
#' }
#' @seealso 
#' * [bound_bvhar()] to define L-BFGS-B optimization bounds.
#' * Individual functions: [choose_bvar()]
#' @references 
#' Giannone, D., Lenza, M., & Primiceri, G. E. (2015). *Prior Selection for Vector Autoregressions*. Review of Economics and Statistics, 97(2).
#' 
#' Kim, Y. G., and Baek, C. (2024). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation, 94(6), 1139-1157.
#' @order 1
#' @export
choose_bayes <- function(bayes_bound = bound_bvhar(),
                         ...,
                         eps = 1e-04,
                         y, 
                         order = c(5, 22),
                         include_mean = TRUE,
                         parallel = list()) {
  dim_data <- ncol(y)
  if (!is.boundbvharemp(bayes_bound)) {
    stop("Provide 'bayes_bound' for 'boundbvharemp'. See ?bound_bvhar.")
  }
  bayes_spec <- bayes_bound$spec
  switch(
    bayes_spec$process,
    "BVAR" = {
      if (length(order) != 1) {
        stop("Set the 'order' to be VAR order: the length should be 1.")
      }
      choose_bvar(
        bayes_spec = bayes_spec,
        lower = bayes_bound$lower, 
        upper = bayes_bound$upper, 
        ..., 
        eps = eps,
        y = y, 
        p = order, 
        include_mean = include_mean,
        parallel = parallel
      )
    },
    "BVHAR" = {
      if (length(order) != 2) {
        stop("Set the 'order' to be VHAR order (e.g. c(5, 22)): the length should be 2.")
      }
      choose_bvhar(
        bayes_spec = bayes_spec,
        lower = bayes_bound$lower, 
        upper = bayes_bound$upper, 
        ..., 
        eps = eps,
        y = y, 
        har = order,
        include_mean = include_mean,
        parallel = parallel
      )
    }
  )
}
