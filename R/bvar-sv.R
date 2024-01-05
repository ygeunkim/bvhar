#' Fitting Bayesian VAR-SV
#' 
#' `r lifecycle::badge("experimental")` This function fits VAR-SV.
#'  It can have Minnesota, SSVS, and Horseshoe prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec `r lifecycle::badge("experimental")` SV specification by [set_sv()].
#' @param init_spec `r lifecycle::badge("experimental")` SV initialization by [init_sv()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param minnesota Apply cross-variable shrinkage structure (Minnesota-way). By default, `FALSE`.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @details
#' Cholesky stochastic volatility modeling for VAR based on
#' \deqn{\Sigma_t = L^T D_t^{-1} L}
#' @return `bvar_sv()` returns an object named `bvarsv` [class].
#' @references 
#' Carriero, A., Chan, J., Clark, T. E., & Marcellino, M. (2022). *Corrigendum to “Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors” \[J. Econometrics 212 (1)(2019) 137–154\]*. Journal of Econometrics, 227(2), 506-512.
#' 
#' Chan, J., Koop, G., Poirier, D., & Tobias, J. (2019). *Bayesian Econometric Methods (2nd ed., Econometric Exercises)*. Cambridge: Cambridge University Press.
#' 
#' Cogley, T., & Sargent, T. J. (2005). *Drifts and volatilities: monetary policies and outcomes in the post WWII US*. Review of Economic Dynamics, 8(2), 262–302.
#' 
#' Gruber, L., & Kastner, G. (2022). *Forecasting macroeconomic data with Bayesian VARs: Sparse or dense? It depends!* arXiv.
#' @importFrom posterior as_draws_df bind_draws
#' @order 1
#' @export
bvar_sv <- function(y,
                    p,
                    num_iter = 1000,
                    num_burn = floor(num_iter / 2),
                    thinning = 1,
                    bayes_spec = set_sv(),
                    init_spec = init_sv(),
                    include_mean = TRUE,
                    minnesota = FALSE,
                    verbose = FALSE,
                    num_thread = 1) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  dim_data <- ncol(y)
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, p, p + 1)
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
  num_design <- nrow(Y0)
  dim_design <- ncol(X0)
  num_alpha <- dim_data^2 * p
  num_eta <- dim_data * (dim_data - 1) / 2
  # model specification---------------
  if (!is.svspec(bayes_spec)) {
    stop("Provide 'svspec' for 'bayes_spec'.")
  }
  prior_spec <- bayes_spec$prior
  if (!is.svinit(init_spec)) {
    stop("Provide 'svinit' for 'init_spec'.")
  }
  coef_init <- init_spec$coef
  if (length(bayes_spec$shape) == 1) {
    bayes_spec$shape <- rep(bayes_spec$shape, dim_data)
    bayes_spec$scale <- rep(bayes_spec$scale, dim_data)
    bayes_spec$initial_mean <- rep(bayes_spec$initial_mean, dim_data)
  }
  if (length(bayes_spec$initial_prec) == 1) {
    bayes_spec$initial_prec <- bayes_spec$initial_prec * diag(dim_data)
  }
  # MCMC iterations-------------------
  if (num_iter < 1) {
    stop("Iterate more than 1 times for MCMC.")
  }
  if (num_iter < num_burn) {
    stop("'num_iter' should be larger than 'num_burn'.")
  }
  if (thinning < 1) {
    stop("'thinning' should be non-negative.")
  }
  res <- switch(
    prior_spec$prior,
    "Minnesota" = {
      if (prior_spec$process != "BVAR") {
        stop("'prior_spec' must be the result of 'set_bvar()'.")
      }
      if (prior_spec$prior != "Minnesota") {
        stop("In 'set_bvar()', just input numeric values.")
      }
      if (is.null(prior_spec$sigma)) {
        prior_spec$sigma <- apply(y, 2, sd)
      }
      sigma <- prior_spec$sigma
      if (is.null(prior_spec$delta)) {
        prior_spec$delta <- rep(1, dim_data)
      }
      delta <- prior_spec$delta
      lambda <- prior_spec$lambda
      eps <- prior_spec$eps
      # Minnesota-moment--------------------------------------
      Yp <- build_ydummy(p, sigma, lambda, delta, numeric(dim_data), numeric(dim_data), include_mean)
      colnames(Yp) <- name_var
      Xp <- build_xdummy(1:p, lambda, sigma, eps, include_mean)
      colnames(Xp) <- name_lag
      mn_prior <- minnesota_prior(Xp, Yp)
      prior_mean <- mn_prior$prior_mean
      prior_prec <- mn_prior$prior_prec
      # MCMC---------------------------------------------------
      estimate_var_sv(
        num_iter = num_iter,
        num_burn = num_burn,
        x = X0,
        y = Y0,
        param_sv = bayes_spec[3:6],
        param_prior = list(
          prior_mean = prior_mean,
          prior_prec = prior_prec,
          sigma = diag(1 / sigma)
        ),
        init_prior = list(),
        # prior_coef_mean = prior_mean,
        # prior_coef_prec = prior_prec,
        # prec_diag = diag(1 / sigma),
        prior_type = 1,
        # init_local = rep(.1, ifelse(include_mean, num_alpha + dim_data, num_alpha)),
        # init_global = .1,
        # init_contem_local = rep(.1, num_eta),
        # init_contem_global = .1,
        grp_id = 1,
        grp_mat = matrix(0L, nrow = dim_design, ncol = dim_data),
        # prior_sig_shp = bayes_spec$shape,
        # prior_sig_scl = bayes_spec$scale,
				# prior_init_mean = bayes_spec$initial_mean,
				# prior_init_prec = bayes_spec$initial_prec,
        # coef_spike = rep(0.1, num_alpha),
        # coef_slab = rep(5, num_alpha),
        # coef_slab_weight = rep(.5, 1),
        # chol_spike = rep(.1, num_eta),
        # chol_slab = rep(5, num_eta),
        # chol_slab_weight = rep(.5, num_eta),
        # coef_s1 = 1,
        # coef_s2 = 1,
        # chol_s1 = 1,
        # chol_s2 = 1,
        # mean_non = rep(0, dim_data),
        # sd_non = .1,
        include_mean = include_mean,
        display_progress = verbose,
        nthreads = num_thread
      )
    },
    "SSVS" = {
      prior_spec$process <- "VAR"
      init_coef <- 1L
      init_coef_dummy <- 1L
      glob_idmat <- build_grpmat(
        p = p,
        dim_data = dim_data,
        dim_design = num_alpha / dim_data,
        num_coef = num_alpha,
        minnesota = ifelse(minnesota, "short", "no"),
        include_mean = FALSE
      )
      grp_id <- unique(c(glob_idmat))
      num_grp <- length(grp_id)
      if (length(prior_spec$coef_spike) == 1) {
        prior_spec$coef_spike <- rep(prior_spec$coef_spike, num_alpha)
      }
      if (length(prior_spec$coef_slab) == 1) {
        prior_spec$coef_slab <- rep(prior_spec$coef_slab, num_alpha)
      }
      if (length(coef_init$init_coef_weight) == 1) {
        coef_init$init_coef_weight <- rep(coef_init$init_coef_weight, num_grp)
      }
      if (length(prior_spec$mean_non) == 1) {
        prior_spec$mean_non <- rep(prior_spec$mean_non, dim_data)
      }
      if (length(prior_spec$chol_spike) == 1) {
        prior_spec$chol_spike <- rep(prior_spec$chol_spike, num_eta)
      }
      if (length(prior_spec$chol_slab) == 1) {
        prior_spec$chol_slab <- rep(prior_spec$chol_slab, num_eta)
      }
      if (length(coef_init$init_chol_weight) == 1) {
        coef_init$init_chol_weight <- rep(coef_init$init_chol_weight, num_eta)
      }
      if (all(is.na(prior_spec$coef_spike)) || all(is.na(prior_spec$coef_slab))) {
        # Conduct semiautomatic function using var_lm()
        stop("Specify spike-and-slab of coefficients.")
      }
      if (!(
        length(prior_spec$coef_spike) == num_alpha &&
        length(prior_spec$coef_slab) == num_alpha
      )) {
        stop("Invalid 'coef_spike' and 'coef_slab' size.")
      }
      # MCMC---------------------------------------------------
      estimate_var_sv(
        num_iter = num_iter,
        num_burn = num_burn,
        x = X0,
        y = Y0,
        param_sv = bayes_spec[3:6],
        param_prior = prior_spec,
        init_prior = coef_init,
        # prior_coef_mean = matrix(0L, nrow = dim_design, ncol = dim_data),
        # prior_coef_prec = diag(dim_design),
        # prec_diag = diag(dim_data),
        prior_type = 2,
        # init_local = rep(.1, ifelse(include_mean, num_alpha + dim_data, num_alpha)),
        # init_global = rep(.1, num_grp),
        # init_contem_local = rep(.1, num_eta),
        # init_contem_global = .1,
        grp_id = grp_id,
        grp_mat = glob_idmat,
        # prior_sig_shp = bayes_spec$shape,
        # prior_sig_scl = bayes_spec$scale,
				# prior_init_mean = bayes_spec$initial_mean,
				# prior_init_prec = bayes_spec$initial_prec,
        # coef_spike = prior_spec$coef_spike,
        # coef_slab = prior_spec$coef_slab,
        # coef_slab_weight = coef_init$init_coef_weight,
        # chol_spike = prior_spec$chol_spike,
        # chol_slab = prior_spec$chol_slab,
        # chol_slab_weight = coef_init$init_chol_weight,
        # coef_s1 = prior_spec$coef_s1,
        # coef_s2 = prior_spec$coef_s2,
        # chol_s1 = prior_spec$chol_s1,
        # chol_s2 = prior_spec$chol_s2,
        # mean_non = prior_spec$mean_non,
        # sd_non = prior_spec$sd_non,
        include_mean = include_mean,
        display_progress = verbose,
        nthreads = num_thread
      )
    },
    "Horseshoe" = {
      prior_spec$process <- "VAR"
      num_restrict <- ifelse(
        include_mean,
        num_alpha + dim_data,
        num_alpha
      )
      if (length(coef_init$local_sparsity) != dim_design) {
        if (length(coef_init$local_sparsity) == 1) {
          coef_init$local_sparsity <- rep(coef_init$local_sparsity, num_restrict)
        } else {
          stop("Length of the vector 'local_sparsity' should be dim * p or dim * p + 1.")
        }
      }
      glob_idmat <- build_grpmat(
        p = p,
        dim_data = dim_data,
        dim_design = dim_design,
        num_coef = num_restrict,
        minnesota = ifelse(minnesota, "short", "no"),
        include_mean = include_mean
      )
      # init_local <- coef_init$local_sparsity
      grp_id <- unique(c(glob_idmat))
      # init_global <- rep(coef_init$global_sparsity, length(grp_id))
      coef_init$global_sparsity <- rep(coef_init$global_sparsity, length(grp_id))
      # MCMC---------------------------------------------------
      estimate_var_sv(
        num_iter = num_iter,
        num_burn = num_burn,
        x = X0,
        y = Y0,
        param_sv = bayes_spec[3:6],
        param_prior = list(),
        init_prior = append(
          coef_init,
          list(
            contem_local_sparsity = rep(.1, num_eta),
            contem_global_sparsity = .1
          )
        ),
        # prior_coef_mean = matrix(0L, nrow = dim_design, ncol = dim_data),
        # prior_coef_prec = diag(dim_design),
        # prec_diag = diag(dim_data),
        prior_type = 3,
        # init_local = init_local,
        # init_global = init_global,
        # init_contem_local = rep(.1, num_eta),
        # init_contem_global = .1,
        grp_id = grp_id,
        grp_mat = glob_idmat,
        # prior_sig_shp = bayes_spec$shape,
        # prior_sig_scl = bayes_spec$scale,
				# prior_init_mean = bayes_spec$initial_mean,
				# prior_init_prec = bayes_spec$initial_prec,
        # coef_spike = rep(0.1, num_alpha),
        # coef_slab = rep(5, num_alpha),
        # coef_slab_weight = rep(.5, length(grp_id)),
        # chol_spike = rep(.1, num_eta),
        # chol_slab = rep(5, num_eta),
        # chol_slab_weight = rep(.5, num_eta),
        # coef_s1 = 1,
        # coef_s2 = 1,
        # chol_s1 = 1,
        # chol_s2 = 1,
        # mean_non = rep(0, dim_data),
        # sd_non = .1,
        include_mean = include_mean,
        display_progress = verbose,
        nthreads = num_thread
      )
    }
  )
  # Preprocess the results--------------------------------
  thin_id <- seq(from = 1, to = num_iter - num_burn, by = thinning)
  res$alpha_record <- res$alpha_record[thin_id,]
  res$h_record <- res$h_record[thin_id,]
  res$a_record <- res$a_record[thin_id,]
  res$h0_record <- res$h0_record[thin_id,]
  res$sigh_record <- res$sigh_record[thin_id,]
  colnames(res$h_record) <- paste0(
    paste0("h[", seq_len(dim_data), "]"),
    gl(num_design, dim_data)
  )
  res$h_record <- as_draws_df(res$h_record)
  res$coefficients <- matrix(colMeans(res$alpha_record), ncol = dim_data)
  mat_lower <- matrix(0L, nrow = dim_data, ncol = dim_data)
  diag(mat_lower) <- rep(1L, dim_data)
  mat_lower[lower.tri(mat_lower, diag = FALSE)] <- colMeans(res$a_record)
  res$chol_posterior <- mat_lower
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_lag
  colnames(res$chol_posterior) <- name_var
  rownames(res$chol_posterior) <- name_var
  colnames(res$alpha_record) <- paste0("alpha[", seq_len(ncol(res$alpha_record)), "]")
  colnames(res$a_record) <- paste0("a[", seq_len(ncol(res$a_record)), "]")
  colnames(res$h0_record) <- paste0("h0[", seq_len(ncol(res$h0_record)), "]")
  colnames(res$sigh_record) <- paste0("sigh[", seq_len(ncol(res$sigh_record)), "]")
  res$alpha_record <- as_draws_df(res$alpha_record)
  res$a_record <- as_draws_df(res$a_record)
  res$h0_record <- as_draws_df(res$h0_record)
  res$sigh_record <- as_draws_df(res$sigh_record)
  if (prior_spec$prior == "SSVS") {
    res$gamma_record <- res$gamma_record[thin_id,]
    res$pip <- colMeans(res$gamma_record)
    res$pip <- matrix(res$pip, ncol = dim_data)
    if (include_mean) {
      res$pip <- rbind(res$pip, rep(1L, dim_data))
    }
    colnames(res$gamma_record) <- paste0("gamma[", 1:num_alpha, "]")
    res$gamma_record <- as_draws_df(res$gamma_record)
    colnames(res$pip) <- name_var
    rownames(res$pip) <- name_lag
  } else if (prior_spec$prior == "Horseshoe") {
    if (minnesota) {
      res$tau_record <- res$tau_record[thin_id,]
      colnames(res$tau_record) <- paste0("tau[", seq_len(ncol(res$tau_record)), "]")
    } else {
      res$tau_record <- as.matrix(res$tau_record[thin_id])
      colnames(res$tau_record) <- "tau"
    }
    res$tau_record <- as_draws_df(res$tau_record)
    res$lambda_record <- res$lambda_record[thin_id,]
    colnames(res$lambda_record) <- paste0(
      "lambda[",
      seq_len(ncol(res$lambda_record)),
      "]"
    )
    res$lambda_record <- as_draws_df(res$lambda_record)
    res$kappa_record <- res$kappa_record[thin_id,]
    colnames(res$kappa_record) <- paste0("kappa[", seq_len(ncol(res$kappa_record)), "]")
    res$pip <- matrix(colMeans(res$kappa_record), ncol = dim_data)
    colnames(res$pip) <- name_var
    rownames(res$pip) <- name_lag
    res$kappa_record <- as_draws_df(res$kappa_record)
  }
  res$param <- bind_draws(
    res$alpha_record,
    res$a_record,
    res$h_record,
    res$h0_record,
    res$sigh_record
  )
  if (prior_spec$prior == "SSVS" || prior_spec$prior == "Horseshoe") {
    res$group <- glob_idmat
    res$num_group <- length(grp_id)
  }
  if (prior_spec$prior == "Minnesota") {
    res$prior_mean <- prior_mean
    res$prior_prec <- prior_prec
  } else if (prior_spec$prior == "SSVS") {
    res$param <- bind_draws(
      res$param,
      res$gamma_record
    )
  } else {
    res$param <- bind_draws(
      res$param,
      res$lambda_record,
      res$tau_record
    )
  }
  # variables------------
  res$df <- dim_design
  res$p <- p
  res$m <- dim_data
  res$obs <- nrow(Y0)
  res$totobs <- nrow(y)
  # model-----------------
  res$call <- match.call()
  res$process <- paste("VAR", prior_spec$prior, bayes_spec$process, sep = "_")
  res$type <- ifelse(include_mean, "const", "none")
  bayes_spec$prior <- prior_spec
  res$spec <- bayes_spec
  # init member
  res$iter <- num_iter
  res$burn <- num_burn
  res$thin <- thinning
  # data------------------
  res$y0 <- Y0
  res$design <- X0
  res$y <- y
  class(res) <- c("bvharsp", "bvarsv", "svmod")
  if (prior_spec$prior == "Horseshoe") {
    class(res) <- c("hsmod", class(res))
  } else if (prior_spec$prior == "SSVS") {
    class(res) <- c("ssvsmod", class(res))
  }
  res
}
