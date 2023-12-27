#' Fitting Bayesian TVP-VAR of Minnesota Belief
#' 
#' `r lifecycle::badge("experimental")` This function fits TVP-VAR with Minnesota belief.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec A BVAR model specification by [set_bvar()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @references 
#' Chan, J., Koop, G., Poirier, D., & Tobias, J. (2019). *Bayesian Econometric Methods (2nd ed., Econometric Exercises)*. Cambridge: Cambridge University Press.
#' @order 1
#' @export
bvar_tvp <- function(y,
                     p,
                     num_iter = 1000,
                     num_burn = floor(num_iter / 2),
                     thinning = 1,
                     bayes_spec = set_bvar(),
                     include_mean = TRUE,
                     verbose = FALSE,
                     num_thread = 1) {
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
  prior_scale <- mn_prior$prior_scale
  prior_shape <- mn_prior$prior_shape
  # MCMC---------------------------------------------------
  res <- estimate_var_tvp(
    num_iter = num_iter,
    num_burn = num_burn,
    x = X0,
    y = Y0,
    prior_coef_mean = prior_mean,
    prior_coef_prec = prior_prec,
    prec_diag = diag(1 / sigma),
    prior_sig_df = prior_shape,
    prior_sig_scale = prior_scale,
    display_progress = verbose,
    nthreads = num_thread
  )
  # Preprocess the results--------------------------------
  thin_id <- seq(from = 1, to = num_iter - num_burn, by = thinning)
  res$alpha_record <- res$alpha_record[thin_id,]
  res$alpha0_record <- res$alpha0_record[thin_id,]
  res$q_record <- res$q_record[thin_id,]
  res$sig_record <- split_psirecord(res$sig_record, 1, varname = "sig")
  res$sig_record <- res$sig_record[thin_id]
  res$alpha_posterior <- colMeans(res$alpha_record)
  res$alpha_posterior <-
    matrix(res$alpha_posterior, ncol = dim_data, byrow = TRUE) %>%
    split.data.frame(gl(num_design, dim_design)) %>%
    lapply(
      function(x) {
        mat_coef <- matrix(t(x), ncol = dim_data)
        colnames(mat_coef) <- name_var
        rownames(mat_coef) <- name_lag
        mat_coef
      }
    )
  res$coefficients <- Reduce("+", res$alpha_posterior) / length(res$alpha_posterior) # over 1, ..., n
  res$covmat <- Reduce("+", res$sig_record) / length(res$sig_record)
  
  # variables------------
  res$df <- dim_design
  res$p <- p
  res$m <- dim_data
  res$obs <- nrow(Y0)
  res$totobs <- nrow(y)
  # model-----------------
  res$call <- match.call()
  res$process <- paste("VAR", bayes_spec$prior, "TVP", sep = "_")
  res$type <- ifelse(include_mean, "const", "none")
  res$spec <- bayes_spec
  res$iter <- num_iter
  res$burn <- num_burn
  res$thin <- thinning
  # prior-----------------
  res$prior_mean <- prior_mean
  res$prior_prec <- prior_prec
  res$prior_scale <- prior_scale
  res$prior_shape <- prior_shape
  # data------------------
  res$y0 <- Y0
  res$design <- X0
  res$y <- y
  class(res) <- c("bvartvp", "tvpmod")
  res
}