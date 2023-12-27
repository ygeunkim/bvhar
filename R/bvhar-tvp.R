#' Fitting Bayesian TVP-VHAR of Minnesota Belief
#'
#' `r lifecycle::badge("experimental")` This function fits TVP-VHAR with Minnesota belief.
#'
#' @param y Time series data of which columns indicate the variables
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec A BVHAR model specification by [set_bvhar()] (default) or [set_weight_bvhar()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @references
#' Chan, J., Koop, G., Poirier, D., & Tobias, J. (2019). *Bayesian Econometric Methods (2nd ed., Econometric Exercises)*. Cambridge: Cambridge University Press.
#' @order 1
#' @export
bvhar_tvp <- function(y,
                      har = c(5, 22),
                      num_iter = 1000,
                      num_burn = floor(num_iter / 2),
                      thinning = 1,
                      bayes_spec = set_bvhar(),
                      include_mean = TRUE,
                      verbose = FALSE,
                      num_thread = 1) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  if (!is.bvharspec(bayes_spec)) {
    stop("Provide 'bvharspec' for 'bayes_spec'.")
  }
  if (bayes_spec$process != "BVHAR") {
    stop("'bayes_spec' must be the result of 'set_bvhar()' or 'set_weight_bvhar()'.")
  }
  if (length(har) != 2 || !is.numeric(har)) {
    stop("'har' should be numeric vector of length 2.")
  }
  if (har[1] > har[2]) {
    stop("'har[1]' should be smaller than 'har[2]'.")
  }
  week <- har[1] # 5
  month <- har[2] # 22
  minnesota_type <- bayes_spec$prior
  dim_data <- ncol(y)
  # N <- nrow(y)
  # num_coef <- 3 * dim_data + 1
  # model specification---------------
  if (is.null(bayes_spec$sigma)) {
    bayes_spec$sigma <- apply(y, 2, sd)
  }
  sigma <- bayes_spec$sigma
  lambda <- bayes_spec$lambda
  eps <- bayes_spec$eps
  # Y0 = X0 A + Z---------------------
  Y0 <- build_y0(y, month, month + 1)
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
  X0 <- build_design(y, month, include_mean)
  HARtrans <- scale_har(dim_data, week, month, include_mean)
  name_har <- concatenate_colnames(name_var, c("day", "week", "month"), include_mean) # in misc-r.R file
  X1 <- X0 %*% t(HARtrans)
  colnames(X1) <- name_har
  dim_har <- ncol(X1) # 3 * dim_data + 1
  # Minnesota-moment--------------------------------------
  Yh <- switch(minnesota_type,
    "MN_VAR" = {
      if (is.null(bayes_spec$delta)) {
        bayes_spec$delta <- rep(1, dim_data)
      }
      Yh <- build_ydummy(3, sigma, lambda, bayes_spec$delta, numeric(dim_data), numeric(dim_data), include_mean)
      colnames(Yh) <- name_var
      Yh
    },
    "MN_VHAR" = {
      if (is.null(bayes_spec$daily)) {
        bayes_spec$daily <- rep(1, dim_data)
      }
      if (is.null(bayes_spec$weekly)) {
        bayes_spec$weekly <- rep(1, dim_data)
      }
      if (is.null(bayes_spec$monthly)) {
        bayes_spec$monthly <- rep(1, dim_data)
      }
      Yh <- build_ydummy(
        3,
        sigma,
        lambda,
        bayes_spec$daily,
        bayes_spec$weekly,
        bayes_spec$monthly,
        include_mean
      )
      colnames(Yh) <- name_var
      Yh
    }
  )
  Xh <- build_xdummy(1:3, lambda, sigma, eps, include_mean)
  colnames(Xh) <- name_har
  mn_prior <- minnesota_prior(Xh, Yh)
  prior_mean <- mn_prior$prior_mean
  prior_prec <- mn_prior$prior_prec
  prior_scale <- mn_prior$prior_scale
  prior_shape <- mn_prior$prior_shape
  # MCMC---------------------------------------------------
  res <- estimate_var_tvp(
    num_iter = num_iter,
    num_burn = num_burn,
    x = X1,
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
  names(res) <- gsub(pattern = "^alpha", replacement = "phi", x = names(res)) # alpha to phi
  thin_id <- seq(from = 1, to = num_iter - num_burn, by = thinning)
  res$phi_record <- res$phi_record[thin_id, ]
  res$phi0_record <- res$phi0_record[thin_id, ]
  res$q_record <- res$q_record[thin_id, ]
  res$sig_record <- split_psirecord(res$sig_record, 1, varname = "sig")
  res$sig_record <- res$sig_record[thin_id]

  res$phi_posterior <- colMeans(res$phi_record)
  res$phi_posterior <-
    matrix(res$phi_posterior, ncol = dim_data, byrow = TRUE) %>%
    split.data.frame(gl(num_design, dim_har)) %>%
    lapply(
      function(x) {
        mat_coef <- matrix(t(x), ncol = dim_data)
        colnames(mat_coef) <- name_var
        rownames(mat_coef) <- name_har
        mat_coef
      }
    )
  res$coefficients <- Reduce("+", res$phi_posterior) / length(res$phi_posterior) # over 1, ..., n
  res$covmat <- Reduce("+", res$sig_record) / length(res$sig_record)

  # variables------------
  res$df <- dim_har
  res$p <- 3
  res$week <- week
  res$month <- month
  res$m <- dim_data
  res$obs <- num_design
  res$totobs <- nrow(y)
  # model-----------------
  res$call <- match.call()
  res$process <- paste("VHAR", bayes_spec$prior, "TVP", sep = "_")
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
  res$HARtrans <- HARtrans
  res$y0 <- Y0
  res$design <- X0
  res$y <- y
  class(res) <- c("bvhartvp", "tvpmod")
  res
}