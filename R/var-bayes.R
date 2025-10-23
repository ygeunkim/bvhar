#' Fitting Bayesian VAR with Coefficient and Covariance Prior
#' 
#' `r lifecycle::badge("maturing")`
#' This function fits BVAR.
#' Covariance term can be homoskedastic or heteroskedastic (stochastic volatility).
#' It can have Minnesota, SSVS, and Horseshoe prior.
#'
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param exogen Unmodeled variables
#' @param s Lag of exogeneous variables in VARX(p, s). By default, `s = 0`.
#' @param factor_spec `r lifecycle::badge("experimental")` Factor augmentation specification by [set_factor()].
#' @param num_chains Number of MCMC chains
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param coef_spec Coefficient prior specification by [set_bvar()], [set_ssvs()], or [set_horseshoe()].
#' @param contem_spec Contemporaneous coefficient prior specification by [set_bvar()], [set_ssvs()], or [set_horseshoe()].
#' @param exogen_spec Exogenous coefficient prior specification.
#' @param loading_spec `r lifecycle::badge("experimental")` Factor loading prior specification.
#' @param cov_spec `r lifecycle::badge("experimental")` SV specification by [set_sv()].
#' @param intercept `r lifecycle::badge("experimental")` Prior for the constant term by [set_intercept()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param minnesota Apply cross-variable shrinkage structure (Minnesota-way). By default, `TRUE`.
#' @param ggl If `TRUE` (default), use additional group shrinkage parameter for group structure.
#' Otherwise, use group shrinkage parameter instead of global shirnkage parameter.
#' Applies to HS, NG, and DL priors.
#' @param save_init Save every record starting from the initial values (`TRUE`).
#' By default, exclude the initial values in the record (`FALSE`), even when `num_burn = 0` and `thinning = 1`.
#' If `num_burn > 0` or `thinning != 1`, this option is ignored.
#' @param convergence Convergence threshold for rhat < convergence. By default, `NULL` which means no warning.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @param num_thread Number of threads
#' @details
#' Cholesky stochastic volatility modeling for VAR based on
#' \deqn{\Sigma_t^{-1} = L^T D_t^{-1} L},
#' and implements corrected triangular algorithm for Gibbs sampler.
#' @return `var_bayes()` returns an object named `bvarsv` [class].
#' \describe{
#'   \item{coefficients}{Posterior mean of coefficients.}
#'   \item{chol_posterior}{Posterior mean of contemporaneous effects.}
#'   \item{param}{Every set of MCMC trace.}
#'   \item{param_names}{Name of every parameter.}
#'   \item{group}{Indicators for group.}
#'   \item{num_group}{Number of groups.}
#'   \item{df}{Numer of Coefficients: `3m + 1` or `3m`}
#'   \item{p}{VAR lag}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Description of the model, e.g. `VHAR_SSVS_SV`, `VHAR_Horseshoe_SV`, or `VHAR_minnesota-part_SV`}
#'   \item{type}{include constant term (`const`) or not (`none`)}
#'   \item{spec}{Coefficients prior specification}
#'   \item{sv}{log volatility prior specification}
#'   \item{intercept}{Intercept prior specification}
#'   \item{init}{Initial values}
#'   \item{chain}{The numer of chains}
#'   \item{iter}{Total iterations}
#'   \item{burn}{Burn-in}
#'   \item{thin}{Thinning}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' If it is SSVS or Horseshoe:
#' \describe{
#'   \item{pip}{Posterior inclusion probabilities.}
#' }
#' @references
#' Carriero, A., Chan, J., Clark, T. E., & Marcellino, M. (2022). *Corrigendum to “Large Bayesian vector autoregressions with stochastic volatility and non-conjugate priors” \[J. Econometrics 212 (1)(2019) 137-154\]*. Journal of Econometrics, 227(2), 506-512.
#'
#' Chan, J., Koop, G., Poirier, D., & Tobias, J. (2019). *Bayesian Econometric Methods (2nd ed., Econometric Exercises)*. Cambridge: Cambridge University Press.
#'
#' Cogley, T., & Sargent, T. J. (2005). *Drifts and volatilities: monetary policies and outcomes in the post WWII US*. Review of Economic Dynamics, 8(2), 262-302.
#'
#' Gruber, L., & Kastner, G. (2022). *Forecasting macroeconomic data with Bayesian VARs: Sparse or dense? It depends!* arXiv.
#' 
#' Huber, F., Koop, G., & Onorante, L. (2021). *Inducing Sparsity and Shrinkage in Time-Varying Parameter Models*. Journal of Business & Economic Statistics, 39(3), 669-683.
#' 
#' Korobilis, D. (2022). A new algorithm for structural restrictions in Bayesian vector autoregressions. European Economic Review, 148, 104241.
#' 
#' Korobilis, D., & Shimizu, K. (2022). *Bayesian Approaches to Shrinkage and Sparse Estimation*. Foundations and Trends® in Econometrics, 11(4), 230-354.
#' 
#' Ray, P., & Bhattacharya, A. (2018). *Signal Adaptive Variable Selector for the Horseshoe Prior*. arXiv.
#' @importFrom posterior as_draws_df bind_draws summarise_draws
#' @order 1
#' @export
var_bayes <- function(y,
                      p,
                      exogen = NULL,
                      s = 0,
                      factor_spec = set_factor(),
                      num_chains = 1,
                      num_iter = 1000,
                      num_burn = floor(num_iter / 2),
                      thinning = 1,
                      coef_spec = set_bvar(),
                      contem_spec = coef_spec,
                      cov_spec = set_ldlt(),
                      intercept = set_intercept(),
                      exogen_spec = coef_spec,
                      loading_spec = coef_spec,
                      include_mean = TRUE,
                      minnesota = TRUE,
                      ggl = TRUE,
                      save_init = FALSE,
                      convergence = NULL,
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
  Y0 <- build_response(y, p, p + 1)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
    colnames(y) <- name_var
  }
  colnames(Y0) <- name_var
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  X0 <- build_design(y, p, include_mean)
  name_lag <- concatenate_colnames(name_var, 1:p, include_mean) # in misc-r.R file
  exogen_prior <- list()
  exogen_init <- list()
  exogen_prior_type <- 0
  dim_exogen_design <- 0
  if (!is.null(exogen)) {
    validate_prior(exogen_spec)
    if (!is.matrix(exogen)) {
      exogen <- as.matrix(exogen)
    }
    if (!is.null(colnames(exogen))) {
      name_exogen <- colnames(exogen)
    } else {
      name_exogen <- paste0("x", seq_len(ncol(exogen)))
    }
    dim_exogen <- ncol(exogen)
    # exogen_spec <- validate_spec(
    #   bayes_spec = exogen_spec,
    #   y = exogen,
    #   dim_data = dim_exogen,
    #   process = "BVAR"
    # )
    # exogen_prior <- get_spec(
    #   bayes_spec = exogen_spec,
    #   p = 0,
    #   dim_data = dim_exogen
    # )
    # dim_exogen_design <- dim_exogen * (s + 1)
    # num_exogen <- dim_data * dim_exogen_design
    # Might be better use group also in exogen!
    exogen_prior_type <- enumerate_prior(exogen_spec$prior)
    exogen_id <- length(name_lag) + 1:((s + 1) * dim_exogen)
    name_lag <- c(
      name_lag,
      concatenate_colnames(name_exogen, 0:s, FALSE)
    )
    X0 <- build_exogen_design(y, exogen, p, s, include_mean)
    # exogen_id <- length(name_lag) + 1:((s + 1) * dim_exogen)
    dim_exogen_design <- length(exogen_id)
    num_exogen <- dim_data * dim_exogen_design
    exogen_spec <- validate_spec(
      bayes_spec = exogen_spec,
      y = exogen,
      dim_data = num_exogen,
      process = "BVAR"
    )
    exogen_prior <- get_spec(
      bayes_spec = exogen_spec,
      p = 0,
      dim_data = num_exogen
    )
  }
  colnames(X0) <- name_lag
  num_design <- nrow(Y0)
  dim_design <- ncol(X0)
  num_alpha <- dim_data^2 * p
  num_eta <- dim_data * (dim_data - 1) / 2
  size_factor <- factor_spec$size_factor
  lag_factor <- factor_spec$lag
  num_factor <- size_factor * dim_data
  factor_prior_type <- 0
  factor_prior <- list()
  factor_init <- list()
  if (size_factor > 0) {
    validate_prior(loading_spec)
    factor_prior_type <- enumerate_prior(loading_spec$prior)
    loading_spec <- validate_spec(
      bayes_spec = loading_spec,
      y = NULL,
      dim_data = num_factor,
      process = "BVAR"
    )
    factor_prior <- get_spec(
      bayes_spec = loading_spec,
      p = 0,
      dim_data = num_factor
    )
    X0 <- cbind(X0, matrix(0L, nrow = num_design, ncol = size_factor))
    dim_design <- ncol(X0)
    factor_id <- length(name_lag) + seq_len(size_factor)
    name_lag <- c(
      name_lag,
      concatenate_colnames("factor", seq_len(size_factor), FALSE)
    )
    colnames(X0) <- name_lag
  }
  # model specification---------------
  validate_prior(coef_spec)
  validate_prior(contem_spec)
  if (!is.covspec(cov_spec)) {
    stop("Provide 'covspec' for 'cov_spec'.")
  }
  if (!is.interceptspec(intercept)) {
    stop("Provide 'interceptspec' for 'intercept'.")
  }
  if (length(cov_spec$shape) == 1) {
    cov_spec$shape <- rep(cov_spec$shape, dim_data)
    cov_spec$scale <- rep(cov_spec$scale, dim_data)
  }
  if (length(intercept$mean_non) == 1) {
    intercept$mean_non <- rep(intercept$mean_non, dim_data)
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
  # prior_nm <- coef_spec$prior
  # prior_nm <- ifelse(
  #   coef_spec$prior == "MN_Hierarchical",
  #   "Minnesota",
  #   coef_spec$prior
  # )
  # Initialization--------------------
  # param_init <- lapply(
  #   seq_len(num_chains),
  #   function(x) {
  #     list(
  #       init_coef = matrix(runif(dim_data * dim_design, -1, 1), ncol = dim_data),
  #       init_contem = exp(runif(num_eta, -1, 0)) # Cholesky factor
  #     )
  #   }
  # )
  param_init <- get_coef_init(
    num_chains = num_chains,
    dim_data = dim_data,
    dim_design = dim_design,
    num_eta = num_eta
  )
  glob_idmat <- build_grpmat(
    p = p,
    dim_data = dim_data,
    dim_design = num_alpha / dim_data,
    num_coef = num_alpha,
    minnesota = ifelse(minnesota, "longrun", "no"),
    include_mean = FALSE
  )
  grp_id <- unique(c(glob_idmat))
  if (minnesota) {
    # own_id <- 2
    # cross_id <- seq_len(p + 1)[-2]
    own_id <- seq(2, 2 * p, by = 2)
    cross_id <- seq(1, 2 * p, by = 2)
  } else {
    own_id <- 1
    cross_id <- 2
  }
  num_grp <- length(grp_id)
  coef_spec <- validate_spec(
    bayes_spec = coef_spec,
    y = y,
    dim_data = dim_data,
    num_grp = num_grp,
    grp_id = grp_id,
    own_id = own_id,
    cross_id = cross_id,
    process = "BVAR"
  )
  contem_spec <- validate_spec(
    bayes_spec = contem_spec,
    y = y,
    dim_data = num_eta,
    # num_grp = 1,
    # grp_id = grp_id,
    # own_id = own_id,
    # cross_id = cross_id,
    process = "BVAR"
  )
  coef_prior <- get_spec(
    bayes_spec = coef_spec,
    p = p,
    dim_data = dim_data
  )
  contem_prior <- get_spec(
    bayes_spec = contem_spec,
    p = 0,
    dim_data = num_eta
  )
  contem_init <- get_init(
    param_init = param_init,
    prior_nm = contem_spec$prior,
    num_alpha = num_eta,
    num_grp = ifelse(contem_spec$prior == "SSVS" || contem_spec$prior == "GDP", num_eta, 1)
  )
  if (!is.null(exogen)) {
    exogen_init <- get_init(
      param_init = param_init,
      prior_nm = exogen_spec$prior,
      num_alpha = num_exogen,
      num_grp = ifelse(exogen_spec$prior == "SSVS" || exogen_spec$prior == "GDP", num_exogen, 1)
    )
  }
  if (size_factor > 0) {
    factor_init <- get_init(
      param_init = param_init,
      prior_nm = loading_spec$prior,
      num_alpha = num_factor,
      num_grp = ifelse(loading_spec$prior == "SSVS" || loading_spec$prior == "GDP", num_factor, 1)
    )
  }
  param_init <- get_init(
    param_init = param_init,
    prior_nm = coef_spec$prior,
    num_alpha = num_alpha,
    num_grp = num_grp
  )
  prior_type <- enumerate_prior(coef_spec$prior)
  contem_prior_type <- enumerate_prior(contem_spec$prior)
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_chains && num_chains != 1) {
    warning("'num_thread' > 'num_chains' will not use every thread. Specify as 'num_thread' <= 'num_chains'.")
  }
  if (num_burn == 0 && thinning == 1 && save_init) {
    num_burn <- -1
  }
  if (is.svspec(cov_spec)) {
    if (length(cov_spec$initial_mean) == 1) {
      cov_spec$initial_mean <- rep(cov_spec$initial_mean, dim_data)
    }
    if (length(cov_spec$initial_prec) == 1) {
      # cov_spec$initial_prec <- cov_spec$initial_prec * diag(dim_data)
      cov_spec$initial_prec <- rep(cov_spec$initial_prec, dim_data)
    }
    param_init <- lapply(
      param_init,
      function(init) {
        append(
          init,
          list(
            lvol_init = runif(dim_data, -1, 1),
            lvol = matrix(exp(runif(dim_data * num_design, -1, 1)), ncol = dim_data), # log-volatilities
            lvol_sig = exp(runif(dim_data, -1, 1)) # always positive
          )
        )
      }
    )
    param_cov <- cov_spec[c("shape", "scale", "initial_mean", "initial_prec")]
  } else {
    param_init <- lapply(
      param_init,
      function(init) {
        append(
          init,
          list(init_diag = exp(runif(dim_data, -1, 1))) # always positive
        )
      }
    )
    param_cov <- cov_spec[c("shape", "scale")]
  }
  res <- estimate_sur(
    num_chains = num_chains,
    num_iter = num_iter,
    num_burn = num_burn,
    thin = thinning,
    x = X0,
    y = Y0,
    param_reg = param_cov,
    # param_prior = param_prior,
    param_prior = coef_prior,
    param_intercept = intercept[c("mean_non", "sd_non")],
    param_init = param_init,
    prior_type = prior_type,
    ggl = ggl,
    contem_prior = contem_prior,
    contem_init = contem_init,
    contem_prior_type = contem_prior_type,
    exogen_prior = exogen_prior,
    exogen_init = exogen_init,
    exogen_prior_type = exogen_prior_type,
    exogen_cols = dim_exogen_design,
    factor_prior = factor_prior,
    factor_init = factor_init,
    factor_prior_type = factor_prior_type,
    size_factor = size_factor,
    grp_id = grp_id,
    own_id = own_id,
    cross_id = cross_id,
    grp_mat = glob_idmat,
    include_mean = include_mean,
    seed_chain = sample.int(.Machine$integer.max, size = num_chains),
    display_progress = verbose,
    nthreads = num_thread
  )
  res <- do.call(rbind, res)
  rec_names <- colnames(res)
  param_names <- gsub(pattern = "_record$", replacement = "", rec_names)
  # res <- apply(res, 2, function(x) do.call(rbind, x))
  res <- apply(
    res,
    2,
    function(x) {
      if (is.vector(x[[1]])) {
        return(as.matrix(unlist(x)))
      }
      do.call(rbind, x)
    }
  )
  names(res) <- rec_names
  # summary across chains--------------------------------
  res$coefficients <- matrix(colMeans(res$alpha_record), ncol = dim_data)
  res$sparse_coef <- matrix(colMeans(res$alpha_sparse_record), ncol = dim_data)
  if (include_mean) {
    res$coefficients <- rbind(res$coefficients, colMeans(res$c_record))
    res$sparse_coef <- rbind(res$sparse_coef, colMeans(res$c_sparse_record))
  }
  if (!is.null(exogen)) {
    res$coefficients <- rbind(
      res$coefficients,
      matrix(colMeans(res$b_record), ncol = dim_data)
    )
    res$sparse_coef <- rbind(
      res$sparse_coef,
      matrix(colMeans(res$b_sparse_record), ncol = dim_data)
    )
  }
  if (size_factor > 0) {
    res$coefficients <- rbind(
      res$coefficients,
      matrix(colMeans(res$Lambda_record), ncol = dim_data)
    )
    res$sparse_coef <- rbind(
      res$sparse_coef,
      matrix(colMeans(res$Lambda_sparse_record), ncol = dim_data)
    )
  }
  mat_lower <- matrix(0L, nrow = dim_data, ncol = dim_data)
  diag(mat_lower) <- rep(1L, dim_data)
  mat_lower[lower.tri(mat_lower, diag = FALSE)] <- colMeans(res$a_record)
  res$chol_posterior <- mat_lower
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_lag
  colnames(res$sparse_coef) <- name_var
  rownames(res$sparse_coef) <- name_lag
  colnames(res$chol_posterior) <- name_var
  rownames(res$chol_posterior) <- name_var
  res$pip <- colMeans(res$alpha_sparse_record != 0)
  res$pip <- matrix(res$pip, ncol = dim_data)
  if (include_mean) {
    res$pip <- rbind(res$pip, rep(1L, dim_data))
  }
  if (!is.null(exogen)) {
    res$pip <- rbind(
      res$pip,
      matrix(colMeans(res$b_sparse_record != 0), ncol = dim_data)
    )
  }
  if (size_factor > 0) {
    res$pip <- rbind(
      res$pip,
      matrix(colMeans(res$Lambda_sparse_record != 0), ncol = dim_data)
    )
  }
  colnames(res$pip) <- name_var
  rownames(res$pip) <- name_lag
  # if (coef_spec$prior == "SSVS") {
  #   res$pip <- colMeans(res$gamma_record)
  #   res$pip <- matrix(res$pip, ncol = dim_data)
  #   if (include_mean) {
  #     res$pip <- rbind(res$pip, rep(1L, dim_data))
  #   }
  #   colnames(res$pip) <- name_var
  #   rownames(res$pip) <- name_lag
  # } else if (coef_spec$prior == "Horseshoe") {
  #   res$pip <- 1 - matrix(colMeans(res$kappa_record), ncol = dim_data)
  #   if (include_mean) {
  #     res$pip <- rbind(res$pip, rep(1L, dim_data))
  #   }
  #   colnames(res$pip) <- name_var
  #   rownames(res$pip) <- name_lag
  # }
  # Preprocess the results--------------------------------
  if (num_chains > 1) {
    res[rec_names] <- lapply(
      seq_along(res[rec_names]),
      function(id) {
        split_chain(res[rec_names][[id]], chain = num_chains, varname = param_names[id])
      }
    )
  } else {
    res[rec_names] <- lapply(
      seq_along(res[rec_names]),
      function(id) {
        colnames(res[rec_names][[id]]) <- paste0(param_names[id], "[", seq_len(ncol(res[rec_names][[id]])), "]")
        res[rec_names][[id]]
      }
    )
  }
  res[rec_names] <- lapply(res[rec_names], as_draws_df)
  # rec$param <- bind_draws(res[rec_names])
  res$param <- Reduce(bind_draws, res[rec_names])
  # res$param <- bind_draws(
  #   res$alpha_record,
  #   res$a_record,
  #   res$alpha_sparse_record,
  #   res$a_sparse_record
  # )
  # if (is.svspec(cov_spec)) {
  #   res$param <- bind_draws(
  #     res$param,
  #     res$h_record,
  #     res$h0_record,
  #     res$sigh_record
  #   )
  # } else {
  #   res$param <- bind_draws(
  #     res$param,
  #     res$d_record
  #   )
  # }
  # if (include_mean) {
  #   res$param <- bind_draws(
  #     res$param,
  #     res$c_record,
  #     res$c_sparse_record
  #   )
  # }
  # if (!is.null(exogen)) {
  #   res$param <- bind_draws(
  #     res$param,
  #     res$b_record,
  #     res$b_sparse_record
  #   )
  # }
  # if (coef_spec$prior == "SSVS") {
  #   res$param <- bind_draws(
  #     res$param,
  #     res$gamma_record
  #   )
  # } else if (coef_spec$prior == "Horseshoe") {
  #   res$param <- bind_draws(
  #     res$param,
  #     res$lambda_record,
  #     res$eta_record,
  #     res$tau_record,
  #     res$kappa_record
  #   )
  # } else if (coef_spec$prior == "NG") {
  #   res$param <- bind_draws(
  #     res$param,
  #     res$lambda_record,
  #     res$eta_record,
  #     res$tau_record
  #   )
  # } else if (coef_spec$prior == "DL") {
  #   res$param <- bind_draws(
  #     res$param,
  #     res$lambda_record,
  #     res$tau_record
  #   )
  # } else if (coef_spec$prior == "GDP") {
  #   # 
  # }
  res[rec_names] <- NULL
  res$param_names <- param_names
  if (!is.null(convergence)) {
    conv_diagnostics <- summarise_draws(res$param, "rhat")
    if (any(conv_diagnostics$rhat >= convergence)) {
      warning(
        sprintf(
          "Convergence warning with Rhat >= %f:\n%s",
          convergence,
          paste0(conv_diagnostics$variable[conv_diagnostics$rhat >= convergence], collapse = ", ")
        )
      )
    }
  }
  res$group <- glob_idmat
  res$num_group <- length(grp_id)
  # if (coef_spec$prior == "Minnesota") {
  #   res$prior_mean <- prior_mean
  #   res$prior_prec <- prior_prec
  # }
  res$ggl <- ggl
  # variables------------
  res$df <- dim_design
  res$p <- p
  res$m <- dim_data
  res$obs <- nrow(Y0)
  res$totobs <- nrow(y)
  # model-----------------
  res$call <- match.call()
  res$process <- paste("VAR", coef_spec$prior, cov_spec$process, sep = "_")
  res$type <- ifelse(include_mean, "const", "none")
  # res$spec <- coef_spec
  res$spec_coef <- coef_spec
  res$spec_contem <- contem_spec
  res$sv <- cov_spec
  res$intercept <- intercept
  # res$init <- param_init
  res$init_coef <- param_init
  res$init_contem <- contem_init
  res$chain <- num_chains
  res$iter <- num_iter
  res$burn <- num_burn
  res$thin <- thinning
  # data------------------
  if (!is.null(exogen)) {
    res$spec_exogen <- exogen_spec
    res$init_exogen <- exogen_init
    res$exogen_data <- exogen
    res$s <- s
    res$exogen_m <- dim_exogen
    res$exogen_id <- exogen_id
  }
  if (size_factor > 0) {
    res$spec_factor <- loading_spec
    res$init_factor <- factor_init
    res$factor_size <- size_factor
    res$factor_lag <- lag_factor
    res$factor_id <- factor_id
  }
  res$y0 <- Y0
  res$design <- X0
  res$y <- y
  class(res) <- "bvharsp"
  if (is.svspec(cov_spec)) {
    class(res) <- c("bvarsv", "svmod", class(res)) # remove bvarsv later
  } else {
    class(res) <- c("bvarldlt", "ldltmod", class(res))
  }
  if (coef_spec$prior == "Horseshoe") {
    class(res) <- c(class(res), "hsmod")
  } else if (coef_spec$prior == "SSVS") {
    class(res) <- c(class(res), "ssvsmod")
  } else if (coef_spec$prior == "NG") {
    class(res) <- c(class(res), "ngmod")
  } else if (coef_spec$prior == "DL") {
    class(res) <- c(class(res), "dlmod")
  } else if (coef_spec$prior == "GDP") {
    class(res) <- c(class(res), "gdpmod")
  }
  res
}
