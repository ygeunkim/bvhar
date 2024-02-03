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

#' Fitting Hierarchical Bayesian VAR(p)
#' 
#' This function fits hierarchical BVAR(p) with general Minnesota prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec A BVAR model specification by [set_ssvs()].
#' @param scale_variance Proposal distribution scaling constant to adjust an acceptance rate
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param parallel List the same argument of [optimParallel::optimParallel()]. By default, this is empty, and the function does not execute parallel computation.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @details 
#' SSVS prior gives prior to parameters \eqn{\alpha = vec(A)} (VAR coefficient) and \eqn{\Sigma_e^{-1} = \Psi \Psi^T} (residual covariance).
#' 
#' \deqn{\alpha_j \mid \gamma_j \sim (1 - \gamma_j) N(0, \kappa_{0j}^2) + \gamma_j N(0, \kappa_{1j}^2)}
#' \deqn{\gamma_j \sim Bernoulli(q_j)}
#' 
#' and for upper triangular matrix \eqn{\Psi},
#' 
#' \deqn{\psi_{jj}^2 \sim Gamma(shape = a_j, rate = b_j)}
#' \deqn{\psi_{ij} \mid w_{ij} \sim (1 - w_{ij}) N(0, \kappa_{0,ij}^2) + w_{ij} N(0, \kappa_{1,ij}^2)}
#' \deqn{w_{ij} \sim Bernoulli(q_{ij})}
#' 
#' Gibbs sampler is used for the estimation.
#' See [ssvs_bvar_algo] how it works.
#' @return `bvar_niwhm` returns an object named `bvarhm` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Coefficient Matrix}
#'   \item{p}{Lag of VAR}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' @references 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1).
#' 
#' Giannone, D., Lenza, M., & Primiceri, G. E. (2015). *Prior Selection for Vector Autoregressions*. Review of Economics and Statistics, 97(2).
#' 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25.
#' @importFrom stats optim
#' @importFrom optimParallel optimParallel
#' @importFrom posterior as_draws_df bind_draws
#' @order 1
#' @export
bvar_niwhm <- function(y,
                       p,
                       num_iter = 1000, 
                       num_burn = floor(num_iter / 2),
                       thinning = 1,
                       bayes_spec = set_bvar(sigma = set_psi(), lambda = set_lambda()),
                       scale_variance = .05,
                       include_mean = TRUE,
                       parallel = list(),
                       verbose = FALSE) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  # model specification---------------
  if (!is.bvharspec(bayes_spec)) {
    if (!(is.list(bayes_spec) && all(sapply(bayes_spec, is.bvharspec)))) {
      stop("Provide 'bvharspec' for 'bayes_spec'. Or, it should be the list of 'bvharspec'.")
    }
  }
  
  if (bayes_spec$process != "BVAR") {
    stop("'bayes_spec' must be the result of 'set_bvar()'.")
  }
  if (bayes_spec$prior != "MN_Hierarchical") {
    stop("'bayes_spec' must be the result of 'set_lambda()' and 'set_psi()'.")
  }
  psi <- bayes_spec$sigma$mode
  dim_data <- ncol(y)
  psi <- rep(psi, dim_data)
  if (is.null(bayes_spec$delta)) {
    bayes_spec$delta <- rep(1, dim_data)
  }
  delta <- bayes_spec$delta
  lambda <- bayes_spec$lambda$mode
  eps <- bayes_spec$eps
  # Y0 = X0 A + Z---------------------
  Y0 <- build_response(y, p, p + 1)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
  }
  colnames(Y0) <- name_var
  X0 <- build_design(y, p, include_mean)
  name_lag <- concatenate_colnames(name_var, 1:p, include_mean)
  colnames(X0) <- name_lag
  # prior selection-------------------
  lower_vec <- unlist(bayes_spec)
  lower_vec <- as.numeric(lower_vec[grepl(pattern = "lower$", x = names(unlist(bayes_spec)))])
  upper_vec <- unlist(bayes_spec)
  upper_vec <- as.numeric(upper_vec[grepl(pattern = "upper$", x = names(unlist(bayes_spec)))])
  if (length(parallel) > 0) {
    init_par <-
      optimParallel(
        par = c(lambda, psi),
        fn = logml_bvarhm,
        lower = lower_vec,
        upper = upper_vec,
        hessian = TRUE,
        delta = delta,
        eps = bayes_spec$eps, 
        y = y, 
        p = p, 
        include_mean = include_mean,
        parallel = parallel
      )
  } else {
    init_par <- 
      optim(
        par = c(lambda, psi),
        fn = logml_bvarhm,
        method = "L-BFGS-B",
        lower = lower_vec,
        upper = upper_vec,
        hessian = TRUE,
        delta = delta,
        eps = bayes_spec$eps, 
        y = y, 
        p = p, 
        include_mean = include_mean
      )
  }
  lambda <- init_par$par[1]
  psi <- init_par$par[2:(1 + dim_data)]
  hess <- init_par$hessian
  # dummy-----------------------------
  Yp <- build_ydummy_export(p, psi, lambda, delta, numeric(dim_data), numeric(dim_data), include_mean)
  colnames(Yp) <- name_var
  Xp <- build_xdummy_export(1:p, lambda, psi, eps, include_mean)
  colnames(Xp) <- name_lag
  # NIW-------------------------------
  posterior <- estimate_bvar_mn(X0, Y0, Xp, Yp)
  # Prior-----------------------------
  prior_mean <- posterior$prior_mean
  prior_prec <- posterior$prior_prec
  prior_scale <- posterior$prior_scale
  prior_shape <- posterior$prior_shape
  # Matrix normal---------------------
  mn_mean <- posterior$mnmean # matrix normal mean
  colnames(mn_mean) <- name_var
  rownames(mn_mean) <- name_lag
  mn_prec <- posterior$mnprec # matrix normal precision
  colnames(mn_prec) <- name_lag
  rownames(mn_prec) <- name_lag
  yhat <- posterior$fitted
  colnames(yhat) <- name_var
  # Inverse-wishart-------------------
  iw_scale <- posterior$iwscale # IW scale
  colnames(iw_scale) <- name_var
  rownames(iw_scale) <- name_var
  iw_shape <- prior_shape + nrow(Y0)
  # Metropolis algorithm--------------
  metropolis_res <- estimate_hierachical_niw(
    num_iter = num_iter,
    num_burn = num_burn,
    x = X0,
    y = Y0,
    prior_prec = prior_prec,
    prior_scale = prior_scale,
    prior_shape = prior_shape,
    mn_mean = mn_mean,
    mn_prec = mn_prec,
    iw_scale = iw_scale,
    posterior_shape = iw_shape,
    gamma_shp = bayes_spec$lambda$param[1],
    gamma_rate = bayes_spec$lambda$param[2],
    invgam_shp = bayes_spec$sigma$param[1],
    invgam_scl = bayes_spec$sigma$param[2],
    acc_scale = scale_variance,
    obs_information = hess,
    init_lambda = lambda,
    init_psi = psi,
    display_progress = verbose
  )
  
  
  # preprocess the results------------
  thin_id <- seq(from = 1, to = num_iter - num_burn, by = thinning)
  # thinparam_id <- seq(from = 1, to = num_iter - 1 - num_burn, by = thinning)
  
  thinparam_id <- seq(from = 1, to = num_iter - num_burn, by = thinning)
  
  metropolis_res$psi_record <- metropolis_res$psi_record[thin_id,]
  metropolis_res$alpha_record <- metropolis_res$alpha_record[thinparam_id,]
  
  # hyperparameters------------------
  colnames(metropolis_res$psi_record) <- paste0("psi[", seq_len(ncol(metropolis_res$psi_record)), "]")
  metropolis_res$lambda_record <- as.matrix(metropolis_res$lambda_record[thin_id])
  colnames(metropolis_res$lambda_record) <- "lambda"
  metropolis_res$psi_record <- as_draws_df(metropolis_res$psi_record)
  metropolis_res$lambda_record <- as_draws_df(metropolis_res$lambda_record)
  # parameters-----------------------
  colnames(metropolis_res$alpha_record) <- paste0("alpha[", seq_len(ncol(metropolis_res$alpha_record)), "]")
  metropolis_res$coefficients <- 
    colMeans(metropolis_res$alpha_record) %>% 
    matrix(ncol = dim_data)
  colnames(metropolis_res$coefficients) <- name_var
  rownames(metropolis_res$coefficients) <- name_lag
  metropolis_res$alpha_record <- as_draws_df(metropolis_res$alpha_record)
  # posterior mean-------------------
  metropolis_res$covmat <- Reduce("+", split_psirecord(metropolis_res$sigma_record, 1, "sigma")) / length(thinparam_id)
  colnames(metropolis_res$covmat) <- name_var
  rownames(metropolis_res$covmat) <- name_var
  # metropolis_res$sigma_record <- split_psirecord(metropolis_res$sigma_record, 1, "sigma")
  metropolis_res$sigma_record <- 
    t(metropolis_res$sigma_record) %>% 
    matrix(ncol = 1) %>% 
    split.data.frame(gl(dim_data^2, 1, nrow(metropolis_res$sigma_record) * dim_data)) %>% 
    lapply(function(x) x[thinparam_id,])
  # metropolis_res$sigma_record <- metropolis_res$sigma_record[thinparam_id]
  names(metropolis_res$sigma_record) <- paste0("sigma[", 1:(dim_data^2), "]")
  metropolis_res$sigma_record <- as_draws_df(metropolis_res$sigma_record)
  # acceptance rate------------------
  metropolis_res$acc_rate <- mean(metropolis_res$acceptance) # change to matrix and define in chain > 1 later
  metropolis_res$hyperparam <- bind_draws(
    metropolis_res$lambda_record,
    metropolis_res$psi_record
  )
  metropolis_res$param <- bind_draws(
    metropolis_res$alpha_record,
    metropolis_res$sigma_record
  )
  # variables-------------------------
  metropolis_res$df <- nrow(mn_mean)
  metropolis_res$p <- p
  metropolis_res$m <- dim_data
  metropolis_res$obs <- nrow(Y0)
  metropolis_res$totobs <- nrow(y)
  # model-----------------------------
  metropolis_res$call <- match.call()
  metropolis_res$process <- paste(bayes_spec$process, bayes_spec$prior, sep = "_")
  metropolis_res$type <- ifelse(include_mean, "const", "none")
  metropolis_res$spec <- bayes_spec
  metropolis_res$iter <- num_iter
  metropolis_res$burn <- num_burn
  metropolis_res$thin <- thinning
  # data------------------------------
  metropolis_res$y0 <- Y0
  metropolis_res$design <- X0
  metropolis_res$y <- y
  # return S3 object------
  class(metropolis_res) <- c("bvarhm", "bvharsp")
  metropolis_res
}

#' @rdname bvar_niwhm
#' @param x `bvarhm` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvarhm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  cat(sprintf("BVAR(%i) with Hierarchical Prior\n", x$p))
  cat("Fitted by Metropolis algorithm\n")
  cat(paste0("Total number of iteration: ", x$iter, "\n"))
  cat(paste0("Number of burn-in: ", x$burn, "\n"))
  if (x$thin > 1) {
    cat(paste0("Thinning: ", x$thin, "\n"))
  }
  cat("====================================================\n\n")
  cat("Hyperparameter Selection:\n")
  print(
    x$hyperparam,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\n--------------------------------------------------\n")
  cat("Coefficients ~ Matrix Normal Record:\n")
  print(
    x$alpha_record,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\nSigma ~ Inverse-Wishart Record:\n")
  print(
    x$sigma_record,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname bvar_niwhm
#' @exportS3Method knitr::knit_print
knit_print.bvarhm <- function(x, ...) {
  print(x)
}

#' @rdname set_lambda
#' @param x `bvharpriorspec` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharpriorspec <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(paste0("Hyperprior specification for ", x$hyperparam, "\n\n"))
  hyper_prior <- ifelse(x$hyperparam == "lambda", "Gamma", "Inv-Gamma")
  switch (hyper_prior,
    "Gamma" = {
      print.default(
        paste0(
          x$hyperparam,
          " ~ ",
          hyper_prior,
          "(shape = ",
          x$param[1],
          ", rate =",
          x$param[2],
          ")"
        ),
        digits = digits,
        print.gap = 2L,
        quote = FALSE
      )
    },
    "Inv-Gamma" = {
      print.default(
        paste0(
          x$hyperparam,
          " ~ ",
          hyper_prior,
          "(shape = ",
          x$param[1],
          ", scale =",
          x$param[2],
          ")"
        ),
        digits = digits,
        print.gap = 2L,
        quote = FALSE
      )
    }
  )
  cat(sprintf("with mode: %.3f", x$mode))
  invisible(x)
}

#' @rdname set_lambda
#' @exportS3Method knitr::knit_print
knit_print.bvharpriorspec <- function(x, ...) {
  print(x)
}
