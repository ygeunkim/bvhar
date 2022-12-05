#' Fitting Bayesian VHAR of SSVS Prior
#' 
#' This function fits BVAR(p) with stochastic search variable selection (SSVS) prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param num_iter MCMC iteration number
#' @param num_warm Number of warm-up (burn-in). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec A SSVS model specification by [set_ssvs()].
#' @param init_spec SSVS initialization specification by [init_ssvs()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
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
#' 
#' Gibbs sampler is used for the estimation.
#' See [ssvs_bvar_algo] how it works.
#' @return `bvar_ssvs` returns an object named `bvarsp` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Coefficient Matrix}
#'   \item{p}{Lag of VAR}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' @references 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
#' 
#' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
#' @seealso 
#' * Vectorization formulation [var_vec_formulation]
#' * Gibbs sampler algorithm [ssvs_bvar_algo]
#' @importFrom posterior as_draws_df bind_draws
#' @order 1
#' @export
bvhar_ssvs <- function(y, 
                       har = c(5, 22), 
                       num_iter = 1000, 
                       num_warm = floor(num_iter / 2), 
                       thinning = 1,
                       bayes_spec = set_ssvs(), 
                       init_spec = init_ssvs(),
                       include_mean = TRUE) {
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
  if (num_iter < num_warm) {
    stop("'num_iter' should be larger than 'num_warm'.")
  }
  if (thinning < 1) {
    stop("'thinning' should be non-negative.")
  }
  dim_data <- ncol(y) # k
  # dim_har <- 3 * dim_data + 1
  Y0 <- build_y0(y, har[2], har[2] + 1) # n x k
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
  X0 <- build_design(y, har[2], include_mean) # n x dim_design
  name_lag <- concatenate_colnames(name_var, 1:har[2], include_mean)
  colnames(X0) <- name_lag
  hartrans_mat <- scale_har(dim_data, har[1], har[2], include_mean)
  name_har <- concatenate_colnames(name_var, c("day", "week", "month"), include_mean)
  X1 <- X0 %*% t(hartrans_mat)
  colnames(X1) <- name_har
  dim_har <- ncol(X1)
  # length 1 of bayes_spec--------------
  num_restrict <- 3 * dim_data^2 # restrict only coefficients
  num_eta <- dim_data * (dim_data - 1) / 2 # number of upper element of Psi
  if (length(bayes_spec$coef_spike) == 1) {
    bayes_spec$coef_spike <- rep(bayes_spec$coef_spike, num_restrict)
  }
  if (length(bayes_spec$coef_slab) == 1) {
    bayes_spec$coef_slab <- rep(bayes_spec$coef_slab, num_restrict)
  }
  if (length(bayes_spec$coef_mixture) == 1) {
    bayes_spec$coef_mixture <- rep(bayes_spec$coef_mixture, num_restrict)
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
  # Error----------------------------
  if (!(
    length(bayes_spec$coef_spike) == num_restrict &&
    length(bayes_spec$coef_slab) == num_restrict &&
    length(bayes_spec$coef_mixture) == num_restrict
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
  # Initial vectors-------------------
  if (init_spec$chain == 1) {
    if (!(nrow(init_spec$init_coef) == dim_har || ncol(init_spec$init_coef) == dim_data)) {
      stop("Dimension of 'init_coef' should be (3 * dim) x dim or (3 * dim + 1) x dim.")
    }
    if (!(nrow(init_spec$init_coef_dummy) == num_restrict || ncol(init_spec$init_coef_dummy) == dim_data)) {
      stop("Dimension of 'init_coef_dummy' should be (dim * p) x dim x dim.")
    }
    init_coef <- c(init_spec$init_coef)
    init_coef_dummy <- c(init_spec$init_coef_dummy)
    init_chol_diag <- diag(init_spec$init_chol)
    init_chol_upper <- init_spec$init_chol[upper.tri(init_spec$init_chol, diag = FALSE)]
    init_chol_dummy <- init_spec$init_chol_dummy[upper.tri(init_spec$init_chol_dummy, diag = FALSE)]
  } else {
    if (!(nrow(init_spec$init_coef[[1]]) == dim_har || ncol(init_spec$init_coef[[1]]) == dim_data)) {
      stop("Dimension of 'init_coef' should be (3 * dim) x dim or (3 * dim + 1) x dim.")
    }
    if (!(nrow(init_spec$init_coef_dummy[[1]]) == num_restrict || ncol(init_spec$init_coef_dummy[[1]]) == dim_data)) {
      stop("Dimension of 'init_coef_dummy' should be (3 * dim) x dim x dim.")
    }
    init_coef <- unlist(init_spec$init_coef)
    init_coef_dummy <- unlist(init_spec$init_coef_dummy)
    init_chol_diag <- unlist(lapply(init_spec$init_chol, diag))
    init_chol_upper <- unlist(lapply(
      init_spec$init_chol,
      function(x) x[upper.tri(x, diag = FALSE)]
    ))
    init_chol_dummy <- unlist(lapply(
      init_spec$init_chol_dummy,
      function(x) x[upper.tri(x, diag = FALSE)]
    ))
  }
  # MCMC-----------------------------
  ssvs_res <- estimate_bvar_ssvs(
    num_iter = num_iter,
    num_warm = num_warm,
    x = X1,
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
    chol_spike = bayes_spec$chol_spike, # eta spike
    chol_slab = bayes_spec$chol_slab, # eta slab
    chol_slab_weight = bayes_spec$chol_mixture, # qij
    intercept_var = bayes_spec$coef_non, # c for constant c I
    chain = init_spec$chain
  )
  # preprocess the results------------
  thin_id <- seq(from = 1, to = num_iter - num_warm, by = thinning)
  ssvs_res$alpha_record <- ssvs_res$alpha_record[thin_id,]
  ssvs_res$eta_record <- ssvs_res$eta_record[thin_id,]
  ssvs_res$psi_record <- ssvs_res$psi_record[thin_id,]
  ssvs_res$omega_record <- ssvs_res$omega_record[thin_id,]
  ssvs_res$gamma_record <- ssvs_res$gamma_record[thin_id,]
  if (ssvs_res$chain > 1) {
    ssvs_res$alpha_record <- 
      split_paramarray(ssvs_res$alpha_record, chain = ssvs_res$chain, param_name = "alpha") %>% 
      as_draws_df()
    ssvs_res$gamma_record <- 
      split_paramarray(ssvs_res$gamma_record, chain = ssvs_res$chain, param_name = "gamma") %>% 
      as_draws_df()
    ssvs_res$psi_record <- 
      split_paramarray(ssvs_res$psi_record, chain = ssvs_res$chain, param_name = "psi") %>% 
      as_draws_df()
    ssvs_res$eta_record <- 
      split_paramarray(ssvs_res$eta_record, chain = ssvs_res$chain, param_name = "eta") %>% 
      as_draws_df()
    ssvs_res$omega_record <- 
      split_paramarray(ssvs_res$omega_record, chain = ssvs_res$chain, param_name = "omega") %>% 
      as_draws_df()
    ssvs_res$param <- bind_draws(
      ssvs_res$alpha_record, 
      ssvs_res$gamma_record,
      ssvs_res$psi_record,
      ssvs_res$eta_record,
      ssvs_res$omega_record
    )
    # Cholesky factor 3d array-------------
    ssvs_res$chol_record <- split_psirecord(ssvs_res$chol_record, ssvs_res$chain, "cholesky")
    ssvs_res$chol_record <- ssvs_res$chol_record[(num_warm + 1):num_iter] # burn in
    # Posterior mean-------------------------
    ssvs_res$alpha_posterior <- array(ssvs_res$alpha_posterior, dim = c(dim_har, dim_data, ssvs_res$chain))
    # mat_upper <- array(0L, dim = c(dim_data, dim_data, ssvs_res$chain))
    
    
  } else {
    colnames(ssvs_res$alpha_record) <- paste0("alpha[", seq_len(ncol(ssvs_res$alpha_record)), "]")
    colnames(ssvs_res$gamma_record) <- paste0("gamma[", 1:num_restrict, "]")
    colnames(ssvs_res$psi_record) <- paste0("psi[", 1:dim_data, "]")
    colnames(ssvs_res$eta_record) <- paste0("eta[", 1:num_eta, "]")
    colnames(ssvs_res$omega_record) <- paste0("omega[", 1:num_eta, "]")
    ssvs_res$alpha_record <- as_draws_df(ssvs_res$alpha_record)
    ssvs_res$gamma_record <- as_draws_df(ssvs_res$gamma_record)
    ssvs_res$psi_record <- as_draws_df(ssvs_res$psi_record)
    ssvs_res$eta_record <- as_draws_df(ssvs_res$eta_record)
    ssvs_res$omega_record <- as_draws_df(ssvs_res$omega_record)
    ssvs_res$param <- bind_draws(
      ssvs_res$alpha_record, 
      ssvs_res$gamma_record,
      ssvs_res$psi_record,
      ssvs_res$eta_record,
      ssvs_res$omega_record
    )
    # Cholesky factor 3d array---------------
    ssvs_res$chol_record <- split_psirecord(ssvs_res$chol_record, 1, "cholesky")
    ssvs_res$chol_record <- ssvs_res$chol_record[seq(from = num_warm + 1, to = num_iter, by = thinning)] # burn in
    # Posterior mean-------------------------
    ssvs_res$alpha_posterior <- matrix(ssvs_res$alpha_posterior, ncol = dim_data)
    mat_upper <- matrix(0L, nrow = dim_data, ncol = dim_data)
    diag(mat_upper) <- rep(1L, dim_data)
    mat_upper[upper.tri(mat_upper, diag = FALSE)] <- ssvs_res$omega_posterior
    ssvs_res$omega_posterior <- mat_upper
    ssvs_res$gamma_posterior <- matrix(ssvs_res$gamma_posterior, ncol = dim_data)
    if (include_mean) {
      ssvs_res$gamma_posterior <- rbind(ssvs_res$gamma_posterior, rep(1L, dim_data))
    }
  }
  # variables------------
  ssvs_res$df <- dim_har
  ssvs_res$p <- 3
  ssvs_res$week <- har[1]
  ssvs_res$month <- har[2]
  ssvs_res$m <- dim_data
  ssvs_res$obs <- nrow(Y0)
  ssvs_res$totobs <- nrow(y)
  # model-----------------
  ssvs_res$call <- match.call()
  ssvs_res$process <- paste(bayes_spec$process, bayes_spec$prior, sep = "_")
  ssvs_res$type <- ifelse(include_mean, "const", "none")
  ssvs_res$spec <- bayes_spec
  ssvs_res$init <- init_spec
  ssvs_res$iter <- num_iter
  ssvs_res$burn <- num_warm
  ssvs_res$thin <- thinning
  # data------------------
  ssvs_res$HARtrans <- hartrans_mat
  ssvs_res$y0 <- Y0
  ssvs_res$design <- X0
  ssvs_res$y <- y
  # return S3 object------
  class(ssvs_res) <- c("bvharsp", "bvharssvs", "bvharmod")
  ssvs_res
}

#' @rdname bvhar_ssvs
#' @param x `bvharsp` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharsp <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  cat("BVHAR with Hierarchical Prior\n")
  cat("Fitted by Gibbs sampling\n")
  cat(paste0("Total number of iteration: ", x$iter, "\n"))
  cat(paste0("Number of warm-up: ", x$burn, "\n"))
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

#' @rdname bvhar_ssvs
#' @param x `bvharsp` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharsp <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvharsp",
  knit_print.bvharsp,
  envir = asNamespace("knitr")
)
