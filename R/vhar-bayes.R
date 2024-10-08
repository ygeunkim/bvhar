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
  if (!(
    is.bvharspec(bayes_spec) ||
    is.ssvsinput(bayes_spec) ||
    is.horseshoespec(bayes_spec) ||
    is.ngspec(bayes_spec) ||
    is.dlspec(bayes_spec)
  )) {
    stop("Provide 'bvharspec', 'ssvsinput', 'horseshoespec', 'ngspec', or 'dlspec' for 'bayes_spec'.")
  }
  if (!is.covspec(cov_spec)) {
    stop("Provide 'covspec' for 'cov_spec'.")
  }
  # cov_sv <- FALSE
  # if (is.svspec(cov_spec)) {
  #   cov_sv <- TRUE
  # }
  if (!is.interceptspec(intercept)) {
    stop("Provide 'interceptspec' for 'intercept'.")
  }
  if (length(cov_spec$shape) == 1) {
    cov_spec$shape <- rep(cov_spec$shape, dim_data)
    cov_spec$scale <- rep(cov_spec$scale, dim_data)
  }
  if (length(intercept$mean_non) == 1){
    intercept$mean_non <- rep(intercept$mean_non, dim_data)
  }
  prior_nm <- ifelse(
    bayes_spec$prior == "MN_VAR" || bayes_spec$prior == "MN_VHAR" || bayes_spec$prior == "MN_Hierarchical",
    "Minnesota",
    bayes_spec$prior
  )
  # Initialization--------------------
  param_init <- lapply(
    seq_len(num_chains),
    function(x) {
      list(
        init_coef = matrix(runif(dim_data * dim_har, -1, 1), ncol = dim_data),
        init_contem = exp(runif(num_eta, -1, 0)) # Cholesky factor
      )
    }
  )
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
  if (prior_nm == "Minnesota") {
    if (bayes_spec$process != "BVHAR") {
      stop("'bayes_spec' must be the result of 'set_bvhar()' or 'set_weight_bvhar()'.")
    }
    if (length(har) != 2 || !is.numeric(har)) {
      stop("'har' should be numeric vector of length 2.")
    }
    if (har[1] > har[2]) {
      stop("'har[1]' should be smaller than 'har[2]'.")
    }
    # minnesota_type <- bayes_spec$prior
    if (is.null(bayes_spec$sigma)) {
      bayes_spec$sigma <- apply(y, 2, sd)
    }
    if ("delta" %in% names(bayes_spec)) {
      if (is.null(bayes_spec$delta)) {
        bayes_spec$delta <- rep(0, dim_data)
      }
    } else {
      if (is.null(bayes_spec$daily)) {
        bayes_spec$daily <- rep(0, dim_data)
      }
      if (is.null(bayes_spec$weekly)) {
        bayes_spec$weekly <- rep(0, dim_data)
      }
      if (is.null(bayes_spec$monthly)) {
        bayes_spec$monthly <- rep(0, dim_data)
      }
    }
    param_prior <- append(bayes_spec, list(p = 3))
    if (bayes_spec$hierarchical) {
      param_prior$shape <- bayes_spec$lambda$param[1]
      param_prior$rate <- bayes_spec$lambda$param[2]
      prior_nm <- "MN_Hierarchical"
      param_init <- lapply(
        param_init,
        function(init) {
          append(
            init,
            list(
              own_lambda = runif(1, 0, 1),
              cross_lambda = runif(1, 0, 1),
              contem_lambda = runif(1, 0, 1)
            )
          )
        }
      )
    }
  } else if (prior_nm == "SSVS") {
    # if (length(bayes_spec$coef_spike) == 1) {
    #   bayes_spec$coef_spike <- rep(bayes_spec$coef_spike, num_phi)
    # }
    # if (length(bayes_spec$coef_slab) == 1) {
    #   bayes_spec$coef_slab <- rep(bayes_spec$coef_slab, num_phi)
    # }
    # if (length(bayes_spec$coef_mixture) == 1) {
    #   bayes_spec$coef_mixture <- rep(bayes_spec$coef_mixture, num_grp)
    # }
    if (length(bayes_spec$coef_s1) == 2) {
      # bayes_spec$coef_s1 <- rep(bayes_spec$coef_s1, num_grp)
      coef_s1 <- numeric(num_grp)
      coef_s1[grp_id %in% own_id] <- bayes_spec$coef_s1[1]
      coef_s1[grp_id %in% cross_id] <- bayes_spec$coef_s1[2]
      bayes_spec$coef_s1 <- coef_s1
    }
    if (length(bayes_spec$coef_s2) == 2) {
      coef_s2 <- numeric(num_grp)
      coef_s2[grp_id %in% own_id] <- bayes_spec$coef_s2[1]
      coef_s2[grp_id %in% cross_id] <- bayes_spec$coef_s2[2]
      bayes_spec$coef_s2 <- coef_s2
    }
    if (length(bayes_spec$shape) == 1) {
      bayes_spec$shape <- rep(bayes_spec$shape, dim_data)
    }
    if (length(bayes_spec$rate) == 1) {
      bayes_spec$rate <- rep(bayes_spec$rate, dim_data)
    }
    # if (length(bayes_spec$chol_spike) == 1) {
    #   bayes_spec$chol_spike <- rep(bayes_spec$chol_spike, num_eta)
    # }
    # if (length(bayes_spec$chol_slab) == 1) {
    #   bayes_spec$chol_slab <- rep(bayes_spec$chol_slab, num_eta)
    # }
    # if (length(bayes_spec$chol_mixture) == 1) {
    #   bayes_spec$chol_mixture <- rep(bayes_spec$chol_mixture, num_eta)
    # }
    # if (all(is.na(bayes_spec$coef_spike)) || all(is.na(bayes_spec$coef_slab))) {
    #   # Conduct semiautomatic function using var_lm()
    #   stop("Specify spike-and-slab of coefficients.")
    # }
    # if (all(is.na(bayes_spec$chol_spike)) || all(is.na(bayes_spec$chol_slab))) {
    #   # Conduct semiautomatic function using var_lm()
    #   stop("Specify spike-and-slab of cholesky factor.")
    # }
    # if (!(
    #   length(bayes_spec$coef_spike) == num_phi &&
    #     length(bayes_spec$coef_slab) == num_phi &&
    #     length(bayes_spec$coef_mixture) == num_grp
    # )) {
    #   stop("Invalid 'coef_spike', 'coef_slab', and 'coef_mixture' size. The vector size should be the same as 3 * dim^2.")
    # }
    if (!(length(bayes_spec$shape) == dim_data && length(bayes_spec$rate) == dim_data)) {
      stop("Size of SSVS 'shape' and 'rate' vector should be the same as the time series dimension.")
    }
    # if (!(
    #   length(bayes_spec$chol_spike) == num_eta &&
    #     length(bayes_spec$chol_slab) == length(bayes_spec$chol_spike) &&
    #     length(bayes_spec$chol_mixture) == length(bayes_spec$chol_spike)
    # )) {
    #   stop("Invalid 'chol_spike', 'chol_slab', and 'chol_mixture' size. The vector size should be the same as dim * (dim - 1) / 2.")
    # }
    param_prior <- bayes_spec
    param_init <- lapply(
      param_init,
      function(init) {
        coef_mixture <- runif(num_grp, -1, 1)
        coef_mixture <- exp(coef_mixture) / (1 + exp(coef_mixture)) # minnesota structure?
        init_coef_dummy <- rbinom(num_phi, 1, .5) # minnesota structure?
        chol_mixture <- runif(num_eta, -1, 1)
        chol_mixture <- exp(chol_mixture) / (1 + exp(chol_mixture))
        # init_chol_dummy <- rbinom(num_eta, 1, .5)
        init_coef_slab <- exp(runif(num_phi, -1, 1))
        init_contem_slab <- exp(runif(num_eta, -1, 1))
        append(
          init,
          list(
            init_coef_dummy = init_coef_dummy,
            coef_mixture = coef_mixture,
            coef_slab = init_coef_slab,
            chol_mixture = chol_mixture,
            contem_slab = init_contem_slab
          )
        )
      }
    )
  } else if (prior_nm == "Horseshoe") {
    if (length(bayes_spec$local_sparsity) != num_phi) { # -> change other files too: dim_har (dim_design) to num_restrict
      if (length(bayes_spec$local_sparsity) == 1) {
        bayes_spec$local_sparsity <- rep(bayes_spec$local_sparsity, num_phi)
      } else {
        stop("Length of the vector 'local_sparsity' should be dim^2 * 3 or dim^2 * 3 + 1.")
      }
    }
    bayes_spec$group_sparsity <- rep(bayes_spec$group_sparsity, num_grp)
    param_prior <- list()
    param_init <- lapply(
      param_init,
      function(init) {
        local_sparsity <- exp(runif(num_phi, -1, 1))
        global_sparsity <- exp(runif(1, -1, 1))
        group_sparsity <- exp(runif(num_grp, -1, 1))
        contem_local_sparsity <- exp(runif(num_eta, -1, 1)) # sd = local * global
        contem_global_sparsity <- exp(runif(1, -1, 1)) # sd = local * global
        append(
          init,
          list(
            local_sparsity = local_sparsity,
            group_sparsity = group_sparsity,
            global_sparsity = global_sparsity,
            contem_local_sparsity = contem_local_sparsity,
            contem_global_sparsity = contem_global_sparsity
          )
        )
      }
    )
  } else if (prior_nm == "NG") {
    # if (length(bayes_spec$local_shape) == 1) {
    #   bayes_spec$local_shape <- rep(bayes_spec$local_shape, num_phi)
    # }
    # if (length(bayes_spec$contem_shape) == 1) {
    #   bayes_spec$contem_shape <- rep(bayes_spec$contem_shape, num_eta)
    # }
    param_prior <- bayes_spec
    param_init <- lapply(
      param_init,
      function(init) {
        local_sparsity <- exp(runif(num_phi, -1, 1))
        global_sparsity <- exp(runif(1, -1, 1))
        group_sparsity <- exp(runif(num_grp, -1, 1))
        contem_local_sparsity <- exp(runif(num_eta, -1, 1)) # sd = local * global
        contem_global_sparsity <- exp(runif(1, -1, 1)) # sd = local * global
        append(
          init,
          list(
            local_shape = runif(num_grp, 0, 1),
            contem_shape = runif(1, 0, 1),
            local_sparsity = local_sparsity,
            group_sparsity = group_sparsity,
            global_sparsity = global_sparsity,
            contem_local_sparsity = contem_local_sparsity,
            contem_global_sparsity = contem_global_sparsity
          )
        )
      }
    )
  } else if (prior_nm == "DL") {
    # if (length(bayes_spec$dirichlet) == 1) {
    #   bayes_spec$dirichlet <- 1 / num_phi^(1 + .01)
    # }
    # if (length(bayes_spec$contem_dirichlet) == 1) {
    #   bayes_spec$contem_dirichlet <- 1 / num_eta^(1 + .01)
    # }
    param_prior <- bayes_spec
    param_init <- lapply(
      param_init,
      function(init) {
        local_sparsity <- exp(runif(num_phi, -1, 1))
        global_sparsity <- exp(runif(1, -1, 1))
        # group_sparsity <- exp(runif(num_grp, -1, 1))
        contem_local_sparsity <- exp(runif(num_eta, -1, 1)) # sd = local * global
        contem_global_sparsity <- exp(runif(1, -1, 1)) # sd = local * global
        append(
          init,
          list(
            local_sparsity = local_sparsity,
            # group_sparsity = group_sparsity,
            global_sparsity = global_sparsity,
            contem_local_sparsity = contem_local_sparsity,
            contem_global_sparsity = contem_global_sparsity
          )
        )
      }
    )
  }
  prior_type <- switch(prior_nm,
    "Minnesota" = 1,
    "SSVS" = 2,
    "Horseshoe" = 3,
    "MN_Hierarchical" = 4,
    "NG" = 5,
    "DL" = 6
  )
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
      cov_spec$initial_prec <- cov_spec$initial_prec * diag(dim_data)
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
    res <- estimate_var_sv(
      num_chains = num_chains,
      num_iter = num_iter,
      num_burn = num_burn,
      thin = thinning,
      x = X1,
      y = Y0,
      param_sv = cov_spec[c("shape", "scale", "initial_mean", "initial_prec")],
      param_prior = param_prior,
      param_intercept = intercept[c("mean_non", "sd_non")],
      param_init = param_init,
      prior_type = prior_type,
      grp_id = grp_id,
      own_id = own_id,
      cross_id = cross_id,
      grp_mat = glob_idmat,
      include_mean = include_mean,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      display_progress = verbose,
      nthreads = num_thread
    )
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
    res <- estimate_sur(
      num_chains = num_chains,
      num_iter = num_iter,
      num_burn = num_burn,
      thin = thinning,
      x = X1,
      y = Y0,
      param_reg = cov_spec[c("shape", "scale")],
      param_prior = param_prior,
      param_intercept = intercept[c("mean_non", "sd_non")],
      param_init = param_init,
      prior_type = prior_type,
      grp_id = grp_id,
      own_id = own_id,
      cross_id = cross_id,
      grp_mat = glob_idmat,
      include_mean = include_mean,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      display_progress = verbose,
      nthreads = num_thread
    )
  }
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
    res$sparse_coef <- rbind(res$sparse_coef, colMeans(res$c_record))
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
  } else {
    res$param <- bind_draws(
      res$param,
      res$d_record
    )
  }
  if (include_mean) {
    res$param <- bind_draws(
      res$param,
      res$c_record
    )
  }
  if (bayes_spec$prior == "SSVS") {
    res$param <- bind_draws(
      res$param,
      res$gamma_record
    )
  } else if (bayes_spec$prior == "Horseshoe") {
    res$param <- bind_draws(
      res$param,
      res$lambda_record,
      res$eta_record,
      res$tau_record,
      res$kappa_record
    )
  } else if (bayes_spec$prior == "NG") {
    res$param <- bind_draws(
      res$param,
      res$lambda_record,
      res$eta_record,
      res$tau_record
    )
  } else if (bayes_spec$prior == "DL") {
    res$param <- bind_draws(
      res$param,
      res$lambda_record,
      res$tau_record
    )
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
  res$spec <- bayes_spec
  res$sv <- cov_spec
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
  } else {
    class(res) <- c("bvharldlt", "ldltmod", class(res))
  }
  if (bayes_spec$prior == "Horseshoe") {
    class(res) <- c(class(res), "hsmod")
  } else if (bayes_spec$prior == "SSVS") {
    class(res) <- c(class(res), "ssvsmod")
  } else if (bayes_spec$prior == "NG") {
    class(res) <- c(class(res), "ngmod")
  } else if (bayes_spec$prior == "DL") {
    class(res) <- c(class(res), "dlmod")
  }
  res
}
