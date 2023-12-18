#' Fitting Bayesian VHAR-SV
#' 
#' `r lifecycle::badge("experimental")` This function fits VHAR-SV.
#' It can have Minnesota, SSVS, and Horseshoe prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec A BVHAR model specification by [set_bvhar()] (default) or [set_weight_bvhar()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param minnesota Apply cross-variable shrinkage structure (Minnesota-way). Two type: `"short"` type and `"longrun"` type. By default, `"no"`.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @details
#' Cholesky stochastic volatility modeling for VHAR based on
#' \deqn{\Sigma_t = L^T D_t^{-1} L}
#' @return `bvhar_sv()` returns an object named `bvharsv` [class]. It is a list with the following components:
#' \describe{
#'   \item{phi_record}{MCMC trace for vectorized coefficients (phi \eqn{\phi}) with [posterior::draws_df] format.}
#'   \item{h_record}{MCMC trace for log-volatilities.}
#'   \item{a_record}{MCMC trace for contemporaneous coefficients.}
#'   \item{h0_record}{MCMC trace for initial log-volatilities.}
#'   \item{sigh_record}{MCMC trace for log-volatilities variance.}
#'   \item{coefficients}{Posterior mean of coefficients.}
#'   \item{chol_posterior}{Posterior mean of contemporaneous effects.}
#'   \item{pip}{Posterior inclusion probabilities.}
#'   \item{param}{Every set of MCMC trace.}
#'   \item{group}{Indicators for group.}
#'   \item{df}{Numer of Coefficients: `3m + 1` or `3m`}
#'   \item{p}{3 (The number of terms. It contains this element for usage in other functions.)}
#'   \item{week}{Order for weekly term}
#'   \item{month}{Order for monthly term}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Description of the model, e.g. `"VHAR_SSVS_SV", `"VHAR_Horseshoe_SV", or `"VHAR_minnesota-part_SV"}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{spec}{SSVS specification defined by [set_ssvs()]}
#'   \item{init}{Initial specification defined by [init_ssvs()]}
#'   \item{iter}{Total iterations}
#'   \item{burn}{Burn-in}
#'   \item{thin}{Thinning}
#'   \item{chain}{The numer of chains}
#'   \item{HARtrans}{VHAR linear transformation matrix}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' Different members are added according to priors. If it is SSVS:
#' \describe{
#'   \item{gamma_record}{MCMC trace for dummy variable.}
#' }
#' Horseshoe:
#' \describe{
#'   \item{lambda_record}{MCMC trace for local shrinkage level.}
#'   \item{tau_record}{MCMC trace for global shrinkage level.}
#'   \item{kappa_record}{MCMC trace for shrinkage factor.}
#' }
#' @references 
#' Kim, Y. G., and Baek, C. (2023+). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation.
#' 
#' Kim, Y. G., and Baek, C. (n.d.). Working paper.
#' @importFrom posterior as_draws_df bind_draws
#' @order 1
#' @export
bvhar_sv <- function(y,
                     har = c(5, 22),
                     num_iter = 1000,
                     num_burn = floor(num_iter / 2),
                     thinning = 1,
                     bayes_spec = set_bvhar(),
                     include_mean = TRUE,
                     minnesota = c("no", "short", "longrun"),
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
  if (!(
    is.bvharspec(bayes_spec) ||
    is.ssvsinput(bayes_spec) ||
    is.horseshoespec(bayes_spec)
  )) {
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
        init_local = rep(.1, ifelse(include_mean, num_phi + dim_data, num_phi)),
        init_global = .1,
        init_contem_local = rep(.1, dim_data * (dim_data - 1) / 2),
        init_contem_global = .1,
        grp_id = 1,
        grp_mat = matrix(0L, nrow = dim_har, ncol = dim_data),
        coef_spike = rep(0.1, num_phi),
        coef_slab = rep(5, num_phi),
        coef_slab_weight = rep(.5, 1),
        chol_spike = rep(.1, num_eta),
        chol_slab = rep(5, num_eta),
        chol_slab_weight = rep(.5, num_eta),
        coef_s1 = 1,
        coef_s2 = 1,
        chol_s1 = 1,
        chol_s2 = 1,
        mean_non = rep(0, dim_data),
        sd_non = .1,
        include_mean = include_mean,
        display_progress = verbose,
        nthreads = num_thread
      )
    },
    "SSVS" = {
      init_coef <- 1L
      init_coef_dummy <- 1L
      glob_idmat <- build_grpmat(
        p = 3,
        dim_data = dim_data,
        dim_design = num_phi / dim_data,
        num_coef = num_phi,
        minnesota = minnesota,
        include_mean = FALSE
      )
      grp_id <- unique(c(glob_idmat))
      num_grp <- length(grp_id)
      if (length(bayes_spec$coef_spike) == 1) {
        bayes_spec$coef_spike <- rep(bayes_spec$coef_spike, num_phi)
      }
      if (length(bayes_spec$coef_slab) == 1) {
        bayes_spec$coef_slab <- rep(bayes_spec$coef_slab, num_phi)
      }
      if (length(bayes_spec$coef_mixture) == 1) {
        bayes_spec$coef_mixture <- rep(bayes_spec$coef_mixture, num_grp)
      }
      if (length(bayes_spec$mean_non) == 1) {
        bayes_spec$mean_non <- rep(bayes_spec$mean_non, dim_data)
      }
      if (length(bayes_spec$shape) == 1) {
        bayes_spec$shape <- rep(bayes_spec$shape, dim_data)
      }
      if (length(bayes_spec$rate) == 1) {
        bayes_spec$rate <- rep(bayes_spec$rate, dim_data)
      }
      if (length(bayes_spec$chol_spike) == 1) {
        bayes_spec$chol_spike <- rep(bayes_spec$chol_spike, num_eta)
      }
      if (length(bayes_spec$chol_slab) == 1) {
        bayes_spec$chol_slab <- rep(bayes_spec$chol_slab, num_eta)
      }
      if (length(bayes_spec$chol_mixture) == 1) {
        bayes_spec$chol_mixture <- rep(bayes_spec$chol_mixture, num_eta)
      }
      if (all(is.na(bayes_spec$coef_spike)) || all(is.na(bayes_spec$coef_slab))) {
        # Conduct semiautomatic function using var_lm()
        stop("Specify spike-and-slab of coefficients.")
      }
      if (all(is.na(bayes_spec$chol_spike)) || all(is.na(bayes_spec$chol_slab))) {
        # Conduct semiautomatic function using var_lm()
        stop("Specify spike-and-slab of cholesky factor.")
      }
      if (!(
        length(bayes_spec$coef_spike) == num_phi &&
        length(bayes_spec$coef_slab) == num_phi &&
        length(bayes_spec$coef_mixture) == num_grp
        # && length(bayes_spec$mean_coef) == num_restrict
      )) {
        stop("Invalid 'coef_spike', 'coef_slab', and 'coef_mixture' size. The vector size should be the same as 3 * dim^2.")
      }
      if (!(length(bayes_spec$shape) == dim_data && length(bayes_spec$rate) == dim_data)) {
        stop("Size of SSVS 'shape' and 'rate' vector should be the same as the time series dimension.")
      }
      if (!(
        length(bayes_spec$chol_spike) == num_eta &&
        length(bayes_spec$chol_slab) == length(bayes_spec$chol_spike) &&
        length(bayes_spec$chol_mixture) == length(bayes_spec$chol_spike)
      )) {
        stop("Invalid 'chol_spike', 'chol_slab', and 'chol_mixture' size. The vector size should be the same as dim * (dim - 1) / 2.")
      }
      # MCMC---------------------------------------------------
      estimate_var_sv(
        num_iter = num_iter,
        num_burn = num_burn,
        x = X1,
        y = Y0,
        prior_coef_mean = matrix(0L, nrow = dim_har, ncol = dim_data),
        prior_coef_prec = diag(dim_har),
        prec_diag = diag(dim_data),
        prior_type = 2,
        init_local = rep(.1, ifelse(include_mean, num_phi + dim_data, num_phi)),
        init_global = rep(.1, num_grp),
        init_contem_local = rep(.1, dim_data * (dim_data - 1) / 2),
        init_contem_global = .1,
        grp_id = grp_id,
        grp_mat = glob_idmat,
        coef_spike = bayes_spec$coef_spike,
        coef_slab = bayes_spec$coef_slab,
        coef_slab_weight = bayes_spec$coef_mixture,
        chol_spike = bayes_spec$chol_spike,
        chol_slab = bayes_spec$chol_slab,
        chol_slab_weight = bayes_spec$chol_mixture,
        coef_s1 = 1,
        coef_s2 = 1,
        chol_s1 = 1,
        chol_s2 = 1,
        mean_non = rep(0, dim_data),
        sd_non = .1,
        include_mean = include_mean,
        display_progress = verbose,
        nthreads = num_thread
      )
    },
    "Horseshoe" = {
      num_restrict <- ifelse(
        include_mean,
        num_phi + dim_data,
        num_phi
      )
      if (length(bayes_spec$local_sparsity) != dim_har) {
        if (length(bayes_spec$local_sparsity) == 1) {
          bayes_spec$local_sparsity <- rep(bayes_spec$local_sparsity, num_restrict)
        } else {
          stop("Length of the vector 'local_sparsity' should be dim * 3 or dim * 3 + 1.")
        }
      }
      glob_idmat <- build_grpmat(
        p = 3,
        dim_data = dim_data,
        dim_design = dim_har,
        num_coef = num_restrict,
        minnesota = minnesota,
        include_mean = include_mean
      )
      init_local <- bayes_spec$local_sparsity
      grp_id <- unique(c(glob_idmat))
      init_global <- rep(bayes_spec$global_sparsity, length(grp_id))
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
        init_contem_local = rep(.1, dim_data * (dim_data - 1) / 2),
        init_contem_global = .1,
        grp_id = grp_id,
        grp_mat = glob_idmat,
        coef_spike = rep(0.1, num_phi),
        coef_slab = rep(5, num_phi),
        coef_slab_weight = rep(.5, length(grp_id)),
        chol_spike = rep(.1, num_eta),
        chol_slab = rep(5, num_eta),
        chol_slab_weight = rep(.5, num_eta),
        coef_s1 = 1,
        coef_s2 = 1,
        chol_s1 = 1,
        chol_s2 = 1,
        mean_non = rep(0, dim_data),
        sd_non = .1,
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
  res$h_record <- split.data.frame(
    res$h_record,
    gl(num_design, num_iter + 1)
  ) %>%
    lapply(
      function(x) {
        colnames(x) <- paste0("h[", seq_len(ncol(x)), "]")
        x[thin_id + 1,] # since num_iter + 1 rows
      }
    )
  res$h_record <- lapply(
    seq_along(num_design),
    function(x) {
      colnames(res$h_record[[x]]) <- paste0(colnames(res$h_record[[x]]), x)
      res$h_record[[x]]
    }
  )
  res$h_record <- do.call(cbind, res$h_record)
  res$h_record <- as_draws_df(res$h_record)
  res$coefficients <- matrix(colMeans(res$phi_record), ncol = dim_data)
  mat_lower <- matrix(0L, nrow = dim_data, ncol = dim_data)
  diag(mat_lower) <- rep(1L, dim_data)
  mat_lower[lower.tri(mat_lower, diag = FALSE)] <- colMeans(res$a_record)
  res$chol_posterior <- mat_lower
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_har
  colnames(res$chol_posterior) <- name_var
  rownames(res$chol_posterior) <- name_var
  colnames(res$phi_record) <- paste0("phi[", seq_len(ncol(res$phi_record)), "]")
  colnames(res$a_record) <- paste0("a[", seq_len(ncol(res$a_record)), "]")
  colnames(res$h0_record) <- paste0("h0[", seq_len(ncol(res$h0_record)), "]")
  colnames(res$sigh_record) <- paste0("sigh[", seq_len(ncol(res$sigh_record)), "]")
  res$phi_record <- as_draws_df(res$phi_record)
  res$a_record <- as_draws_df(res$a_record)
  res$h0_record <- as_draws_df(res$h0_record)
  res$sigh_record <- as_draws_df(res$sigh_record)
  if (bayes_spec$prior == "SSVS") {
    res$gamma_record <- res$gamma_record[thin_id,]
    res$pip <- colMeans(res$gamma_record)
    res$pip <- matrix(res$pip, ncol = dim_data)
    if (include_mean) {
      res$pip <- rbind(res$pip, rep(1L, dim_data))
    }
    colnames(res$gamma_record) <- paste0("gamma[", 1:num_phi, "]")
    res$gamma_record <- as_draws_df(res$gamma_record)
    colnames(res$pip) <- name_var
    rownames(res$pip) <- name_har
  } else if (bayes_spec$prior == "Horseshoe") {
    if (minnesota == "no") {
      res$tau_record <- as.matrix(res$tau_record[thin_id])
      colnames(res$tau_record) <- "tau"
    } else {
      res$tau_record <- res$tau_record[thin_id,]
      colnames(res$tau_record) <- paste0("tau[", seq_len(ncol(res$tau_record)), "]")
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
  if (bayes_spec$prior == "SSVS" || bayes_spec$prior == "Horseshoe") {
    res$group <- glob_idmat
    res$num_group <- length(grp_id)
  }
  if (bayes_spec$prior == "MN_VAR" || bayes_spec$prior == "MN_VHAR") {
    res$prior_mean <- prior_mean
    res$prior_prec <- prior_prec
  } else if (bayes_spec$prior == "Horseshoe") {
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
  class(res) <- c("bvharsp", "bvharsv", "svmod")
  if (bayes_spec$prior == "Horseshoe") {
    class(res) <- c("hsmod", class(res))
  } else if (bayes_spec$prior == "SSVS") {
    class(res) <- c("ssvsmod", class(res))
  }
  res
}
