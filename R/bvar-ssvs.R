#' Fitting Bayesian VAR(p) of SSVS Prior
#' 
#' `r lifecycle::badge("deprecated")` This function fits BVAR(p) with stochastic search variable selection (SSVS) prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param num_chains Number of MCMC chains
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec A SSVS model specification by [set_ssvs()]. By default, use a default semiautomatic approach [choose_ssvs()].
#' @param init_spec SSVS initialization specification by [init_ssvs()]. By default, use OLS for coefficient and cholesky factor while 1 for dummies.
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param minnesota Apply cross-variable shrinkage structure (Minnesota-way). By default, `FALSE`.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
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
#' @return `bvar_ssvs` returns an object named `bvarssvs` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Posterior mean of VAR coefficients.}
#'   \item{chol_posterior}{Posterior mean of cholesky factor matrix}
#'   \item{covmat}{Posterior mean of covariance matrix}
#'   \item{omega_posterior}{Posterior mean of omega}
#'   \item{pip}{Posterior inclusion probability}
#'   \item{param}{[posterior::draws_df] with every variable: alpha, eta, psi, omega, and gamma}
#'   \item{param_names}{Name of every parameter.}
#'   \item{df}{Numer of Coefficients: `mp + 1` or `mp`}
#'   \item{p}{Lag of VAR}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Description of the model, e.g. `"VAR_SSVS"`}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{spec}{SSVS specification defined by [set_ssvs()]}
#'   \item{init}{Initial specification defined by [init_ssvs()]}
#'   \item{chain}{The numer of chains}
#'   \item{iter}{Total iterations}
#'   \item{burn}{Burn-in}
#'   \item{thin}{Thinning}
#'   \item{group}{Indicators for group.}
#'   \item{num_group}{Number of groups.}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' @references 
#' George, E. I., & McCulloch, R. E. (1993). *Variable Selection via Gibbs Sampling*. Journal of the American Statistical Association, 88(423), 881–889.
#' 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580.
#' 
#' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358.
#' @importFrom posterior as_draws_df bind_draws
#' @order 1
#' @export
bvar_ssvs <- function(y, 
                      p,
                      num_chains = 1,
                      num_iter = 1000, 
                      num_burn = floor(num_iter / 2), 
                      thinning = 1,
                      bayes_spec = choose_ssvs(y = y, ord = p, type = "VAR", param = c(.1, 10), include_mean = include_mean, gamma_param = c(.01, .01), mean_non = 0, sd_non = .1), 
                      init_spec = init_ssvs(type = "auto"),
                      include_mean = TRUE,
                      minnesota = FALSE,
                      verbose = FALSE,
                      num_thread = 1) {
  deprecate_warn("2.0.1", "bvar_ssvs()", "var_bayes()")
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  # model specification---------------
  if (!is.ssvsinput(bayes_spec)) {
    stop("Provide 'ssvsinput' for 'bayes_spec'.")
  }
  if (!is.ssvsinit(init_spec)) {
    stop("Provide 'ssvsinit' for 'init_spec'.")
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
  # Y0 = X0 A + Z---------------------
  dim_data <- ncol(y) # k
  Y0 <- build_response(y, p, p + 1) # n x k
  num_design <- nrow(Y0) # n
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
  }
  colnames(Y0) <- name_var
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  X0 <- build_design(y, p, include_mean) # n x dim_design
  name_lag <- concatenate_colnames(name_var, 1:p, include_mean) # in misc-r.R file
  colnames(X0) <- name_lag
  dim_design <- ncol(X0)
  # no regularization for diagonal term---------------------
  num_restrict <- dim_data^2 * p # restrict only coefficients
  glob_idmat <- build_grpmat(
    p = p,
    dim_data = dim_data,
    dim_design = num_restrict / dim_data,
    num_coef = num_restrict,
    minnesota = ifelse(minnesota, "short", "no"),
    include_mean = FALSE
  )
  grp_id <- unique(c(glob_idmat))
  if (minnesota) {
    own_id <- 2
    cross_id <- seq_len(p + 1)[-2]
  } else {
    own_id <- 1
    cross_id <- 2
  }
  num_grp <- length(grp_id)
  # length 1 of bayes_spec--------------
  num_eta <- dim_data * (dim_data - 1) / 2 # number of upper element of Psi
  if (length(bayes_spec$coef_spike) == 1) {
    bayes_spec$coef_spike <- rep(bayes_spec$coef_spike, num_restrict)
  }
  if (length(bayes_spec$coef_slab) == 1) {
    bayes_spec$coef_slab <- rep(bayes_spec$coef_slab, num_restrict)
  }
  if (length(bayes_spec$coef_mixture) == 1) {
    bayes_spec$coef_mixture <- rep(bayes_spec$coef_mixture, num_grp)
  }
  
  # if (length(bayes_spec$mean_coef) == 1) {
  #   bayes_spec$mean_coef <- rep(bayes_spec$mean_coef, num_restrict)
  # }
  if (length(bayes_spec$coef_s1) == 2) {
    # bayes_spec$coef_s1 <- rep(bayes_spec$coef_s1, num_grp)
    coef_s1 <- numeric(num_grp)
    coef_s1[grp_id %in% own_id] <- bayes_spec$coef_s1[1]
    coef_s1[grp_id %in% cross_id] <- bayes_spec$coef_s1[2]
    bayes_spec$coef_s1 <- coef_s1
  }
  if (length(bayes_spec$coef_s2) == 2) {
    # bayes_spec$coef_s1 <- rep(bayes_spec$coef_s1, num_grp)
    coef_s2 <- numeric(num_grp)
    coef_s2[grp_id %in% own_id] <- bayes_spec$coef_s2[1]
    coef_s2[grp_id %in% cross_id] <- bayes_spec$coef_s2[2]
    bayes_spec$coef_s2 <- coef_s2
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
  # Temporary before making semiautomatic function---------
  if (all(is.na(bayes_spec$coef_spike)) || all(is.na(bayes_spec$coef_slab))) {
    # Conduct semiautomatic function using var_lm()
    stop("Specify spike-and-slab of coefficients.")
  }
  if (all(is.na(bayes_spec$chol_spike)) || all(is.na(bayes_spec$chol_slab))) {
    # Conduct semiautomatic function using var_lm()
    stop("Specify spike-and-slab of cholesky factor.")
  }
  # Error----------------------------
  if (!(
    length(bayes_spec$coef_spike) == num_restrict &&
      length(bayes_spec$coef_slab) == num_restrict
    #  &&
    # length(bayes_spec$coef_mixture) == num_grp
  )) {
    stop("Invalid 'coef_spike' and 'coef_slab' size. The vector size should be the same as dim^2 * p.")
  }
  if (length(bayes_spec$coef_mixture) != num_grp) {
    stop("Invalid 'coef_mixture' size. The vector size should be the same as group number.")
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
  # Initial vectors-------------------
  if (init_spec$type == "user") {
    if (!(nrow(init_spec$init_coef) == dim_design || ncol(init_spec$init_coef) == dim_data)) {
      stop("Dimension of 'init_coef' should be (dim * p) x dim or (dim * p + 1) x dim.")
    }
    if (!(nrow(init_spec$init_coef_dummy) == num_restrict || ncol(init_spec$init_coef_dummy) == dim_data)) {
      stop("Dimension of 'init_coef_dummy' should be (dim * p) x dim x dim.")
    }
    init_coef <- c(init_spec$init_coef)
    init_coef_dummy <- c(init_spec$init_coef_dummy)
    init_chol_diag <- diag(init_spec$init_chol)
    init_chol_upper <- init_spec$init_chol[upper.tri(init_spec$init_chol, diag = FALSE)]
    init_chol_dummy <- init_spec$init_chol_dummy[upper.tri(init_spec$init_chol_dummy, diag = FALSE)]
    init_gibbs <- TRUE
  } else {
    init_coef <- 1L
    init_coef_dummy <- 1L
    init_chol_diag <- 1L
    init_chol_upper <- 1L
    init_chol_dummy <- 1L
    init_gibbs <- FALSE
  }
  # MCMC-----------------------------
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_chains && num_chains != 1) {
    warning("'num_thread' > 'num_chains' will not use every thread. Specify as 'num_thread' <= 'num_chains'.")
  }
  res <- estimate_bvar_ssvs(
    num_chains = num_chains,
    num_iter = num_iter,
    num_burn = num_burn,
    thin = thinning,
    x = X0,
    y = Y0,
    init_coef = init_coef, # initial alpha
    init_chol_diag = init_chol_diag, # initial psi_jj
    init_chol_upper = init_chol_upper, # initial psi_ij
    init_coef_dummy = init_coef_dummy, # initial gamma
    init_chol_dummy = init_chol_dummy, # initial omega
    coef_spike = bayes_spec$coef_spike, # alpha spike
    coef_slab = bayes_spec$coef_slab, # alpha slab
    coef_slab_weight = bayes_spec$coef_mixture, # pj
    shape = bayes_spec$shape, # shape of gamma distn
    rate = bayes_spec$rate, # rate of gamma distn
    coef_s1 = bayes_spec$coef_s1,
    coef_s2 = bayes_spec$coef_s2,
    chol_spike = bayes_spec$chol_spike, # eta spike
    chol_slab = bayes_spec$chol_slab, # eta slab
    chol_slab_weight = bayes_spec$chol_mixture, # qij
    chol_s1 = bayes_spec$chol_s1,
    chol_s2 = bayes_spec$chol_s2,
    grp_id = grp_id,
    grp_mat = glob_idmat,
    mean_non = bayes_spec$mean_non,
    sd_non = bayes_spec$sd_non, # c for constant c I,
    include_mean = include_mean,
    seed_chain = sample.int(.Machine$integer.max, size = num_chains),
    init_gibbs = init_gibbs,
    display_progress = verbose,
    nthreads = num_thread
  )
  res <- do.call(rbind, res)
  rec_names <- colnames(res)
  param_names <- gsub(pattern = "_record$", replacement = "", rec_names)
  res <- apply(res, 2, function(x) do.call(rbind, x))
  names(res) <- rec_names
  res$coefficients <- matrix(colMeans(res$alpha_record), ncol = dim_data)
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_lag
  res$chol_posterior <- matrix(colMeans(res$chol_record), ncol = dim_data)
  colnames(res$chol_posterior) <- name_var
  rownames(res$chol_posterior) <- name_var
  res$covmat <- solve(res$chol_posterior %*% t(res$chol_posterior))
  mat_upper <- matrix(0L, nrow = dim_data, ncol = dim_data)
  diag(mat_upper) <- rep(1L, dim_data)
  mat_upper[upper.tri(mat_upper, diag = FALSE)] <- colMeans(res$omega_record)
  res$omega_posterior <- mat_upper
  colnames(res$omega_posterior) <- name_var
  rownames(res$omega_posterior) <- name_var
  res$pip <- colMeans(res$gamma_record)
  res$pip <- matrix(res$pip, ncol = dim_data)
  if (include_mean) {
    res$pip <- rbind(res$pip, rep(1L, dim_data))
  }
  colnames(res$pip) <- name_var
  rownames(res$pip) <- name_lag
  # preprocess the results------------
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
  res$param <- bind_draws(
    res$alpha_record,
    res$gamma_record,
    res$psi_record,
    res$eta_record,
    res$omega_record
  )
  res[rec_names] <- NULL
  res$param_names <- param_names
  # variables------------
  res$df <- dim_design
  res$p <- p
  res$m <- dim_data
  res$obs <- nrow(Y0)
  res$totobs <- nrow(y)
  # model-----------------
  res$call <- match.call()
  res$process <- paste("VAR", bayes_spec$prior, sep = "_")
  res$type <- ifelse(include_mean, "const", "none")
  res$spec <- bayes_spec
  res$init <- init_spec
  if (!init_gibbs) {
    # res$init$init_coef <- res$ols_coef # fix this line -> Get OLS and assign
    res$init$init_coef_dummy <- matrix(1L, nrow = dim_design, ncol = dim_data)
    # res$init$init_chol <- res$ols_cholesky # fix this line -> Get OLS and assign
    res$init$init_chol_dummy <- matrix(0L, nrow = dim_data, ncol = dim_data)
    res$init$init_chol_dummy[upper.tri(res$init$init_chol_dummy, diag = FALSE)] <- rep(1L, num_eta)
  }
  res$chain <- num_chains
  res$iter <- num_iter
  res$burn <- num_burn
  res$thin <- thinning
  # res$chain <- init_spec$chain
  res$group <- glob_idmat
  res$num_group <- length(grp_id)
  # data------------------
  res$y0 <- Y0
  res$design <- X0
  res$y <- y
  # return S3 object------
  class(res) <- c("bvarssvs", "ssvsmod", "bvharsp")
  res
}
