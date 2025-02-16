#' Fitting Bayesian VHAR with Coefficient and Covariance Prior
#' 
#' `r lifecycle::badge("maturing")`
#' This function fits BVHAR.
#' Covariance term can be homoskedastic or heteroskedastic (stochastic volatility).
#' It can have Minnesota, SSVS, and Horseshoe prior.
#'
#' @param y Time series data of which columns indicate the variables
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param num_chains Number of MCMC chains
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec A BVHAR model specification by [set_bvhar()] (default) [set_weight_bvhar()], [set_ssvs()], or [set_horseshoe()].
#' @param cov_spec `r lifecycle::badge("experimental")` SV specification by [set_sv()].
#' @param intercept `r lifecycle::badge("experimental")` Prior for the constant term by [set_intercept()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param minnesota Apply cross-variable shrinkage structure (Minnesota-way). Two type: `short` type and `longrun` (default) type.
#' You can also set `no`.
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
#' Cholesky stochastic volatility modeling for VHAR based on
#' \deqn{\Sigma_t^{-1} = L^T D_t^{-1} L}
#' @return `vhar_bayes()` returns an object named `bvharsv` [class]. It is a list with the following components:
#' \describe{
#'   \item{coefficients}{Posterior mean of coefficients.}
#'   \item{chol_posterior}{Posterior mean of contemporaneous effects.}
#'   \item{param}{Every set of MCMC trace.}
#'   \item{param_names}{Name of every parameter.}
#'   \item{group}{Indicators for group.}
#'   \item{num_group}{Number of groups.}
#'   \item{df}{Numer of Coefficients: `3m + 1` or `3m`}
#'   \item{p}{3 (The number of terms. It contains this element for usage in other functions.)}
#'   \item{week}{Order for weekly term}
#'   \item{month}{Order for monthly term}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Description of the model, e.g. `VHAR_SSVS_SV`, `VHAR_Horseshoe_SV`, or `VHAR_minnesota-part_SV`}
#'   \item{type}{include constant term (`const`) or not (`none`)}
#'   \item{spec}{Coefficients prior specification}
#'   \item{sv}{log volatility prior specification}
#'   \item{init}{Initial values}
#'   \item{intercept}{Intercept prior specification}
#'   \item{chain}{The numer of chains}
#'   \item{iter}{Total iterations}
#'   \item{burn}{Burn-in}
#'   \item{thin}{Thinning}
#'   \item{HARtrans}{VHAR linear transformation matrix}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' If it is SSVS or Horseshoe:
#' \describe{
#'   \item{pip}{Posterior inclusion probabilities.}
#' }
#' @references
#' Kim, Y. G., and Baek, C. (2024). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation, 94(6), 1139-1157.
#'
#' Kim, Y. G., and Baek, C. (n.d.). Working paper.
#' @importFrom posterior as_draws_df bind_draws summarise_draws
#' @importFrom stats runif rbinom
#' @order 1
#' @export
vhar_bayes <- function(y,
                       har = c(5, 22),
                       num_chains = 1,
                       num_iter = 1000,
                       num_burn = floor(num_iter / 2),
                       thinning = 1,
                       bayes_spec = set_bvhar(),
                       cov_spec = set_ldlt(),
                       intercept = set_intercept(),
                       include_mean = TRUE,
                       minnesota = c("longrun", "short", "no"),
                       ggl = TRUE,
                       save_init = FALSE,
                       convergence = NULL,
                       verbose = FALSE,
                       num_thread = 1) {
  y <- validate_input(y)
  minnesota <- match.arg(minnesota)
  dim_data <- ncol(y)
  week <- har[1] # 5
  month <- har[2] # 22
  num_phi <- 3 * dim_data^2
  num_eta <- dim_data * (dim_data - 1) / 2
  # Y0 = X0 A + Z---------------------
  Y0 <- build_response(y, month, month + 1)
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
  X0 <- build_design(y, month, include_mean)
  HARtrans <- scale_har(dim_data, week, month, include_mean)
  name_har <- concatenate_colnames(name_var, c("day", "week", "month"), include_mean) # in misc-r.R file
  X1 <- X0 %*% t(HARtrans)
  colnames(X1) <- name_har
  num_design <- nrow(Y0)
  dim_har <- ncol(X1) # 3 * dim_data + 1
  # model specification---------------
  param_cov <- validate_covspec(cov_spec = cov_spec, dim_data = dim_data)
  if (length(intercept$mean_non) == 1){
    intercept$mean_non <- rep(intercept$mean_non, dim_data)
  }
  prior_nm <- ifelse(
    bayes_spec$prior == "MN_VAR" || bayes_spec$prior == "MN_VHAR",
    "Minnesota",
    bayes_spec$prior
  )
  # Group structure-------------------
  glob_idmat <- build_grpmat(
    p = 3,
    dim_data = dim_data,
    dim_design = num_phi / dim_data,
    num_coef = num_phi,
    minnesota = minnesota,
    include_mean = FALSE
  )
  grp_id <- unique(c(glob_idmat))
  if (minnesota == "longrun") {
    own_id <- c(2, 4, 6)
    cross_id <- c(1, 3, 5)
  } else if (minnesota == "short") {
    own_id <- 2
    cross_id <- c(1, 3, 4)
  } else {
    own_id <- 1
    cross_id <- 2
  }
  num_grp <- length(grp_id)
  # Initialization--------------------
  param_init <- init_coef(
    num_chains = num_chains,
    dim_data = dim_data,
    dim_design = dim_har,
    num_eta = num_eta
  )
  param_init <- init_shrinkage_prior(
    param_init = param_init,
    prior_nm = prior_nm,
    num_eta = num_eta,
    num_alpha = num_phi,
    num_grp = num_grp
  )
  param_init <- init_cov(
    param_init = param_init,
    cov_spec = cov_spec,
    dim_data = dim_data,
    num_design = num_design
  )
  param_prior <- validate_spec(
    y = y,
    dim_data = dim_data,
    p = 3,
    num_grp = num_grp,
    grp_id = grp_id,
    own_id = own_id,
    cross_id = cross_id,
    bayes_spec = bayes_spec,
    cov_spec = cov_spec,
    intercept = intercept,
    prior_nm = prior_nm,
    process = "BVHAR"
  )
  prior_type <- switch(prior_nm,
    "Minnesota" = 1,
    "SSVS" = 2,
    "Horseshoe" = 3,
    "MN_Hierarchical" = 4,
    "NG" = 5,
    "DL" = 6,
    "GDP" = 7
  )
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_chains && num_chains != 1) {
    warning("'num_thread' > 'num_chains' will not use every thread. Specify as 'num_thread' <= 'num_chains'.")
  }
  if (num_burn == 0 && thinning == 1 && save_init) {
    num_burn <- -1 # should fix this
  }
  res <- estimate_sur(
    num_chains = num_chains,
    num_iter = num_iter,
    num_burn = num_burn,
    thin = thinning,
    x = X1,
    y = Y0,
    param_reg = param_cov$hyperparam,
    param_prior = param_prior$hyperparam,
    param_intercept = intercept[c("mean_non", "sd_non")],
    param_init = param_init,
    prior_type = prior_type,
    ggl = ggl,
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
  colnames(res) <- gsub(pattern = "^alpha", replacement = "phi", x = colnames(res)) # alpha to phi
  rec_names <- colnames(res) # *_record
  param_names <- gsub(pattern = "_record$", replacement = "", rec_names) # phi, h, ...
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
  names(res) <- rec_names # *_record
  # summary across chains--------------------------------
  res$coefficients <- matrix(colMeans(res$phi_record), ncol = dim_data)
  res$sparse_coef <- matrix(colMeans(res$phi_sparse_record), ncol = dim_data)
  if (include_mean) {
    res$coefficients <- rbind(res$coefficients, colMeans(res$c_record))
    res$sparse_coef <- rbind(res$sparse_coef, colMeans(res$c_sparse_record))
  }
  mat_lower <- matrix(0L, nrow = dim_data, ncol = dim_data)
  diag(mat_lower) <- rep(1L, dim_data)
  mat_lower[lower.tri(mat_lower, diag = FALSE)] <- colMeans(res$a_record)
  res$chol_posterior <- mat_lower
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_har
  colnames(res$sparse_coef) <- name_var
  rownames(res$sparse_coef) <- name_har
  colnames(res$chol_posterior) <- name_var
  rownames(res$chol_posterior) <- name_var
  res$pip <- colMeans(res$phi_sparse_record != 0)
  res$pip <- matrix(res$pip, ncol = dim_data)
  if (include_mean) {
    res$pip <- rbind(res$pip, rep(1L, dim_data))
  }
  colnames(res$pip) <- name_var
  rownames(res$pip) <- name_har
  # if (bayes_spec$prior == "SSVS") {
  #   res$pip <- colMeans(res$gamma_record)
  #   res$pip <- matrix(res$pip, ncol = dim_data)
  #   if (include_mean) {
  #     res$pip <- rbind(res$pip, rep(1L, dim_data))
  #   }
  #   colnames(res$pip) <- name_var
  #   rownames(res$pip) <- name_har
  # } else if (bayes_spec$prior == "Horseshoe") {
  #   res$pip <- 1 - matrix(colMeans(res$kappa_record), ncol = dim_data)
  #   if (include_mean) {
  #     res$pip <- rbind(res$pip, rep(1L, dim_data))
  #   }
  #   colnames(res$pip) <- name_var
  #   rownames(res$pip) <- name_har
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
  # res$param <- bind_draws(res[rec_names])
  res$param <- bind_draws(
    res$phi_record,
    res$a_record,
    res$phi_sparse_record,
    res$a_sparse_record
  )
  if (is.svspec(cov_spec)) {
    res$param <- bind_draws(
      res$param,
      res$h_record,
      res$h0_record,
      res$sigh_record
    )
  } else if (is.ldltspec(cov_spec)) {
    res$param <- bind_draws(
      res$param,
      res$d_record
    )
  }
  if (include_mean) {
    res$param <- bind_draws(
      res$param,
      res$c_record,
      res$c_sparse_record
    )
  }
  if (prior_nm == "SSVS") {
    res$param <- bind_draws(
      res$param,
      res$gamma_record
    )
  } else if (prior_nm == "Horseshoe") {
    res$param <- bind_draws(
      res$param,
      res$lambda_record,
      res$eta_record,
      res$tau_record,
      res$kappa_record
    )
  } else if (prior_nm == "NG") {
    res$param <- bind_draws(
      res$param,
      res$lambda_record,
      res$eta_record,
      res$tau_record
    )
  } else if (prior_nm == "DL") {
    res$param <- bind_draws(
      res$param,
      res$lambda_record,
      res$tau_record
    )
  } else if (prior_nm == "GDP") {
    # 
  }
  res[rec_names] <- NULL
  res$param_names <- param_names
  # if (bayes_spec$prior == "SSVS" || bayes_spec$prior == "Horseshoe") {
  #   res$group <- glob_idmat
  #   res$num_group <- length(grp_id)
  # }
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
  # if (bayes_spec$prior == "MN_VAR" || bayes_spec$prior == "MN_VHAR") {
  #   res$prior_mean <- prior_mean
  #   res$prior_prec <- prior_prec
  # }
  res$ggl <- ggl
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
  res$process <- paste("VHAR", bayes_spec$prior, cov_spec$process, sep = "_")
  res$type <- ifelse(include_mean, "const", "none")
  res$spec <- param_prior$spec
  res$sv <- param_cov$spec
  res$init <- param_init
  # if (include_mean) {
  #   res$intercept <- intercept
  # }
  res$intercept <- intercept
  res$chain <- num_chains
  res$iter <- num_iter
  res$burn <- num_burn
  res$thin <- thinning
  # data------------------
  res$HARtrans <- HARtrans
  res$y0 <- Y0
  res$design <- X0
  res$y <- y
  class(res) <- "bvharsp"
  if (is.svspec(cov_spec)) {
    class(res) <- c("bvharsv", "svmod", class(res)) # remove bvharsv later
  } else if (is.ldltspec(cov_spec)) {
    class(res) <- c("bvharldlt", "ldltmod", class(res))
  }
  if (prior_nm == "Horseshoe") {
    class(res) <- c(class(res), "hsmod")
  } else if (prior_nm == "SSVS") {
    class(res) <- c(class(res), "ssvsmod")
  } else if (prior_nm == "NG") {
    class(res) <- c(class(res), "ngmod")
  } else if (prior_nm == "DL") {
    class(res) <- c(class(res), "dlmod")
  } else if (prior_nm == "GDP") {
    class(res) <- c(class(res), "gdpmod")
  }
  res
}
