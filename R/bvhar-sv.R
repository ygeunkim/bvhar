#' Fitting Bayesian VHAR-SV of Minnesota Belief
#' 
#' `r lifecycle::badge("experimental")` This function fits VHAR-SV with Minnesota belief.
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
#' @details
#' Cholesky stochastic volatility modeling for VHAR based on
#' \deqn{\Sigma_t = L^T D_t^{-1} L}
#' @references 
#' Chan, J., Koop, G., Poirier, D., & Tobias, J. (2019). *Bayesian Econometric Methods (2nd ed., Econometric Exercises)*. Cambridge: Cambridge University Press. doi:[10.1017/9781108525947](https://doi.org/10.1017/9781108525947)
#' 
#' Cogley, T., & Sargent, T. J. (2005). *Drifts and volatilities: monetary policies and outcomes in the post WWII US*. Review of Economic Dynamics, 8(2), 262–302. doi:[10.1016/j.red.2004.10.009](https://doi.org/10.1016/j.red.2004.10.009)
#' @importFrom posterior as_draws_df bind_draws
#' @importFrom dplyr mutate
#' @importFrom tidyr pivot_longer pivot_wider unite
#' @order 1
#' @export
bvhar_sv <- function(y,
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
  dim_data <- ncol(y)
  week <- har[1] # 5
  month <- har[2] # 22
  # Y0 = X0 A + Z---------------------
  Y0 <- build_y0(y, month, month + 1)
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
  num_design <- nrow(Y0)
  dim_har <- ncol(X1) # 3 * dim_data + 1
  # model specification---------------
  if (!(is.bvharspec(bayes_spec) || is.horseshoespec(bayes_spec))) {
    stop("Provide 'bvharspec' or 'horseshoespec' for 'bayes_spec'.")
  }
  prior_nm <- ifelse(
    bayes_spec$prior == "MN_VAR" || bayes_spec$prior == "MN_VHAR",
    "Minnesota",
    bayes_spec$prior
  )
  res <- switch(
    prior_nm,
    "Minnesota" = {
      if (bayes_spec$process != "BVHAR") {
        stop("'bayes_spec' must be the result of 'set_bvhar()' or 'set_weight_bvhar()'.")
      }
      if (length(har) != 2 || !is.numeric(har)) {
        stop("'har' should be numeric vector of length 2.")
      }
      if (har[1] > har[2]) {
        stop("'har[1]' should be smaller than 'har[2]'.")
      }
      minnesota_type <- bayes_spec$prior
      if (is.null(bayes_spec$sigma)) {
        bayes_spec$sigma <- apply(y, 2, sd)
      }
      sigma <- bayes_spec$sigma
      lambda <- bayes_spec$lambda
      eps <- bayes_spec$eps
      # Minnesota-moment--------------------------------------
      Yh <- switch(
        minnesota_type,
        "MN_VAR" = {
          if (is.null(bayes_spec$delta)) {
            bayes_spec$delta <- rep(1, dim_data)
          }
          Yh <- build_ydummy(
            3,
            sigma,
            lambda,
            bayes_spec$delta,
            numeric(dim_data),
            numeric(dim_data),
            include_mean
          )
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
      # MCMC---------------------------------------------------
      estimate_var_sv(
        num_iter = num_iter,
        num_burn = num_burn,
        x = X1,
        y = Y0,
        prior_coef_mean = prior_mean,
        prior_coef_prec = prior_prec,
        prec_diag = diag(1 / sigma),
        prior_type = 1,
        init_local = rep(.1, ifelse(include_mean, dim_data^2 * 3 + 1, dim_data^2 * 3)),
        init_global = .1,
        include_mean = include_mean,
        display_progress = verbose,
        nthreads = num_thread
      )
    },
    "Horseshoe" = {
      num_restrict <- ifelse(include_mean, dim_data^2 * 3 + 1, dim_data^2 * 3)
      if (length(bayes_spec$local_sparsity) != dim_har) {
        if (length(bayes_spec$local_sparsity) == 1) {
          bayes_spec$local_sparsity <- rep(bayes_spec$local_sparsity, num_restrict)
        } else {
          stop("Length of the vector 'local_sparsity' should be dim * 3 or dim * 3 + 1.")
        }
      }
      init_local <- bayes_spec$local_sparsity
      init_global <- bayes_spec$global_sparsity
      # MCMC---------------------------------------------------
      estimate_var_sv(
        num_iter = num_iter,
        num_burn = num_burn,
        x = X1,
        y = Y0,
        prior_coef_mean = matrix(0L, nrow = dim_har, ncol = dim_data),
        prior_coef_prec = diag(dim_har),
        prec_diag = diag(dim_data),
        prior_type = 3,
        init_local = init_local,
        init_global = init_global,
        include_mean = include_mean,
        display_progress = verbose,
        nthreads = num_thread
      )
    }
  )
  # Preprocess the results--------------------------------
  names(res) <- gsub(pattern = "^alpha", replacement = "phi", x = names(res)) # alpha to phi
  thin_id <- seq(from = 1, to = num_iter - num_burn, by = thinning)
  res$phi_record <- res$phi_record[thin_id,]
  res$a_record <- res$a_record[thin_id,]
  res$h0_record <- res$h0_record[thin_id,]
  res$sigh_record <- res$sigh_record[thin_id,]
  # res$h_record <- split.data.frame(res$h_record, gl(num_iter - num_burn, num_design))
  # res$h_record <- res$h_record[thin_id]
  colnames(res$h_record) <- paste("h", seq_len(ncol(res$h_record)), sep = "_")
  res$h_record <- 
    res$h_record %>% 
    as.data.frame() %>% 
    mutate(
      iter_id = gl(num_iter - num_burn, num_design),
      id = rep(1:num_design, num_iter - num_burn)
    ) %>% 
    pivot_longer(-c(iter_id, id), names_to = "h_name", values_to = "h_value") %>% 
    unite("varying_name", h_name, id, sep = "") %>% 
    pivot_wider(names_from = "varying_name", values_from = "h_value")
  res$h_record <- as_draws_df(res$h_record[,-1])
  res$coefficients <- matrix(colMeans(res$phi_record), ncol = dim_data)
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_har
  colnames(res$phi_record) <- paste0("phi[", seq_len(ncol(res$phi_record)), "]")
  colnames(res$a_record) <- paste0("a[", seq_len(ncol(res$a_record)), "]")
  colnames(res$h0_record) <- paste0("h0[", seq_len(ncol(res$h0_record)), "]")
  colnames(res$sigh_record) <- paste0("sigh[", seq_len(ncol(res$sigh_record)), "]")
  res$phi_record <- as_draws_df(res$phi_record)
  res$a_record <- as_draws_df(res$a_record)
  res$h0_record <- as_draws_df(res$h0_record)
  res$sigh_record <- as_draws_df(res$sigh_record)
  
  if (bayes_spec$prior == "Horseshoe") {
    res$tau_record <- as.matrix(res$tau_record[thin_id])
    colnames(res$tau_record) <- "tau"
    res$tau_record <- as_draws_df(res$tau_record)
    res$lambda_record <- as.matrix(res$lambda_record[thin_id])
    colnames(res$lambda_record) <- "lambda"
    res$lambda_record <- as_draws_df(res$lambda_record)
    # res$covmat <- mean(res$sigma) * diag(dim_data)
    # res$psi_posterior <- diag(dim_data) / mean(res$sigma)
    # colnames(res$covmat) <- name_var
    # rownames(res$covmat) <- name_var
    # colnames(res$psi_posterior) <- name_var
    # rownames(res$psi_posterior) <- name_var
    # res$sigma_record <- as.matrix(res$sigma_record[thin_id])
    # colnames(res$sigma_record) <- "sigma"
    # res$sigma_record <- as_draws_df(res$sigma_record)
    res$kappa_record <- res$kappa_record[thin_id,]
    colnames(res$kappa_record) <- paste0("kappa[", seq_len(ncol(res$kappa_record)), "]")
    res$pip <- matrix(1 - colMeans(res$kappa_record), ncol = dim_data)
    colnames(res$pip) <- name_var
    rownames(res$pip) <- name_har
    res$kappa_record <- as_draws_df(res$kappa_record)
  }
  
  
  res$param <- bind_draws(
    res$phi_record,
    res$a_record,
    res$h_record,
    res$h0_record,
    res$sigh_record
  )
  if (bayes_spec$prior == "MN_VAR" || bayes_spec$prior == "MN_VHAR") {
    res$prior_mean <- prior_mean
    res$prior_prec <- prior_prec
  } else if (bayes_spec$prior == "Horseshoe") {
    res$param <- bind_draws(
      res$param,
      res$lambda_record,
      res$tau_record
    )
  }
  
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
  res$process <- paste("VHAR", bayes_spec$prior, "SV", sep = "_")
  res$type <- ifelse(include_mean, "const", "none")
  res$spec <- bayes_spec
  res$iter <- num_iter
  res$burn <- num_burn
  res$thin <- thinning
  # data------------------
  res$HARtrans <- HARtrans
  res$y0 <- Y0
  res$design <- X0
  res$y <- y
  class(res) <- c("bvharsv", "svmod")
  res
}

#' @rdname bvhar_sv
#' @param x `bvarsv` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharsv <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  cat("BVHAR with Stochastic Volatility\n")
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

#' @rdname bvhar_sv
#' @param x `bvarsv` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharsv <- function(x, ...) {
  print(x)
}
