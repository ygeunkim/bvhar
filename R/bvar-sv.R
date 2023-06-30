#' Fitting Bayesian VAR-SV of Minnesota Belief
#' 
#' `r lifecycle::badge("experimental")` This function fits VAR-SV with Minnesota belief.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec A BVAR model specification by [set_bvar()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' 
#' @references 
#' Cogley, T., & Sargent, T. J. (2005). *Drifts and volatilities: monetary policies and outcomes in the post WWII US*. Review of Economic Dynamics, 8(2), 262–302. doi:[10.1016/j.red.2004.10.009](https://doi.org/10.1016/j.red.2004.10.009)
#' 
#' Chan, J., Koop, G., Poirier, D., & Tobias, J. (2019). *Bayesian Econometric Methods (2nd ed., Econometric Exercises)*. Cambridge: Cambridge University Press. doi:[10.1017/9781108525947](https://doi.org/10.1017/9781108525947)
#' @importFrom posterior as_draws_df bind_draws
#' @order 1
#' @export
bvar_sv <- function(y,
                    p,
                    num_iter = 1000,
                    num_burn = floor(num_iter / 2),
                    thinning = 1,
                    bayes_spec = set_bvar(),
                    include_mean = TRUE,
                    verbose = FALSE) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  # model specification---------------
  if (!is.bvharspec(bayes_spec)) {
    stop("Provide 'bvharspec' for 'bayes_spec'.")
  }
  if (bayes_spec$process != "BVAR") {
    stop("'bayes_spec' must be the result of 'set_bvar()'.")
  }
  if (bayes_spec$prior != "Minnesota") {
    stop("In 'set_bvar()', just input numeric values.")
  }
  if (is.null(bayes_spec$sigma)) {
    bayes_spec$sigma <- apply(y, 2, sd)
  }
  sigma <- bayes_spec$sigma
  dim_data <- ncol(y)
  if (is.null(bayes_spec$delta)) {
    bayes_spec$delta <- rep(1, dim_data)
  }
  delta <- bayes_spec$delta
  lambda <- bayes_spec$lambda
  eps <- bayes_spec$eps
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, p, p + 1)
  num_design <- nrow(Y0)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
  }
  colnames(Y0) <- name_var
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  X0 <- build_design(y, p, include_mean)
  name_lag <- concatenate_colnames(name_var, 1:p, include_mean) # in misc-r.R file
  colnames(X0) <- name_lag
  dim_design <- ncol(X0)
  # Minnesota-moment--------------------------------------
  Yp <- build_ydummy(p, sigma, lambda, delta, numeric(dim_data), numeric(dim_data), include_mean)
  colnames(Yp) <- name_var
  Xp <- build_xdummy(1:p, lambda, sigma, eps, include_mean)
  colnames(Xp) <- name_lag
  mn_prior <- minnesota_prior(Xp, Yp)
  prior_mean <- mn_prior$prior_mean
  prior_prec <- mn_prior$prior_prec
  # MCMC---------------------------------------------------
  res <- estimate_var_sv(
    num_iter = num_iter,
    num_burn = num_burn,
    x = X0,
    y = Y0,
    prior_coef_mean = prior_mean,
    prior_coef_prec = prior_prec,
    prec_diag = diag(1 / sigma),
    display_progress = verbose
  )
  # Preprocess the results--------------------------------
  thin_id <- seq(from = 1, to = num_iter - num_burn, by = thinning)
  res$alpha_record <- res$alpha_record[thin_id,]
  res$a_record <- res$a_record[thin_id,]
  res$h0_record <- res$h0_record[thin_id,]
  res$sigh_record <- res$sigh_record[thin_id,]
  res$h_record <- split.data.frame(res$h_record, gl(num_iter - num_burn, num_design))
  res$h_record <- res$h_record[thin_id]
  res$coefficients <- matrix(colMeans(res$alpha_record), ncol = dim_data)
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_lag
  colnames(res$alpha_record) <- paste0("alpha[", seq_len(ncol(res$alpha_record)), "]")
  colnames(res$a_record) <- paste0("a[", seq_len(ncol(res$a_record)), "]")
  colnames(res$h0_record) <- paste0("h0[", seq_len(ncol(res$h0_record)), "]")
  res$alpha_record <- as_draws_df(res$alpha_record)
  res$a_record <- as_draws_df(res$a_record)
  res$h0_record <- as_draws_df(res$h0_record)
  res$param <- bind_draws(
    res$alpha_record,
    res$a_record,
    res$h0_record
  )
  # variables------------
  res$df <- dim_design
  res$p <- p
  res$m <- dim_data
  res$obs <- nrow(Y0)
  res$totobs <- nrow(y)
  # model-----------------
  res$call <- match.call()
  res$process <- paste("VAR", bayes_spec$prior, "SV", sep = "_")
  res$type <- ifelse(include_mean, "const", "none")
  res$spec <- bayes_spec
  res$iter <- num_iter
  res$burn <- num_burn
  res$thin <- thinning
  # data------------------
  res$y0 <- Y0
  res$design <- X0
  res$y <- y
  class(res) <- c("bvarsv", "svmod")
  res
}

#' @rdname bvar_sv
#' @param x `bvarsv` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvarsv <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  cat(sprintf("BVAR(%i) with Stochastic Volatility\n", x$p))
  cat("Fitted by Gibbs sampling\n")
  cat(paste0("Total number of iteration: ", x$iter, "\n"))
  cat(paste0("Number of burn-in: ", x$burn, "\n"))
  if (x$thin > 1) {
    cat(paste0("Thinning: ", x$thin, "\n"))
  }
  cat("====================================================\n\n")
  cat("Parameter Record:\n")
  print(
    x$param,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname bvar_sv
#' @param x `bvarsv` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvarsv <- function(x, ...) {
  print(x)
}
