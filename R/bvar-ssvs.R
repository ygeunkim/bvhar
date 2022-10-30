#' Fitting Bayesian VAR(p) of SSVS Prior
#' 
#' This function fits BVAR(p) with stochastic search variable selection (SSVS) prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in
#' @param bayes_spec A BVAR model specification by [set_ssvs()].
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
#' MCMC is used for the estimation.
#' 
#' @references 
#' Jochmann, M., Koop, G., & Strachan, R. W. (2010). *Bayesian forecasting using stochastic search variable selection in a VAR subject to breaks*. International Journal of Forecasting, 26(2), 326–347. doi:[10.1016/j.ijforecast.2009.11.002](https://www.sciencedirect.com/science/article/abs/pii/S0169207009001782?via%3Dihub)
#' 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001753?via%3Dihub)
#' @order 1
#' @export
bvar_ssvs <- function(y, 
                      p, 
                      num_iter, 
                      num_burn, 
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
  # Y0 = X0 B + Z---------------------
  dim_data <- ncol(y) # k
  dim_design <- dim_data * p + 1
  Y0 <- build_y0(y, p, p + 1) # n x k
  num_design <- nrow(Y0) # n
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
  }
  colnames(Y0) <- name_var
  X0 <- build_design(y, p) # s x k
  name_lag <- concatenate_colnames(name_var, 1:p) # in misc-r.R file
  colnames(X0) <- name_lag
  # const or none---------------------
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  if (!include_mean) {
    X0 <- X0[, -dim_design] # exclude 1 column
    name_lag <- name_lag[-dim_design] # colnames(X0)
    dim_design <- dim_design - 1 # df = no intercept
  }
  # length 1 of bayes_spec--------------
  num_restrict <- dim_data^2 * p # restrict only coefficients
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
  if (!(nrow(init_spec$init_coef) == dim_design || ncol(init_spec$init_coef) == dim_data)) {
    stop("Dimension of 'init_coef' should be (dim * p) x dim or (dim * p + 1) x dim.")
  }
  if (!(nrow(init_spec$init_coef_dummy) == num_restrict || ncol(init_spec$init_coef_dummy) == dim_data)) {
    stop("Dimension of 'init_coef_dummy' should be (dim * p) x dim x dim.")
  }
  if (!(
    length(bayes_spec$coef_spike) == num_restrict &&
    length(bayes_spec$coef_slab) == num_restrict &&
    length(bayes_spec$coef_mixture) == num_restrict
  )) {
    stop("Invalid 'coef_spike', 'coef_slab', and 'coef_mixture' size. The vector size should be the same as dim^2 * p.")
  }
  if (!(length(bayes_spec$shape) == dim_data && length(bayes_spec$rate) == dim_data)) {
    stop("Size of SSVS 'shape' and 'rate' vector should be the same as the time series dimension.")
  }
  if (!(
    length(bayes_spec$chol_spike) == dim_data * (dim_data - 1) / 2 &&
    length(bayes_spec$chol_slab) == length(bayes_spec$chol_spike) &&
    length(bayes_spec$chol_mixture) == length(bayes_spec$chol_spike)
  )) {
    stop("Invalid 'chol_spike', 'chol_slab', and 'chol_mixture' size. The vector size should be the same as dim * (dim - 1) / 2.")
  }
  # MCMC-----------------------------
  ssvs_res <- estimate_bvar_ssvs(
    num_iter = num_iter,
    num_burn = num_burn,
    x = X0,
    y = Y0,
    init_coef = init_spec$init_coef, # initial alpha (matrix)
    init_chol_diag = diag(init_spec$init_chol), # initial psi_jj (vector)
    init_chol_upper = init_spec$init_chol[upper.tri(init_spec$init_chol, diag = FALSE)], # initial psi_ij (vector)
    init_coef_dummy = c(init_spec$init_coef_dummy), # initial gamma (vector)
    init_chol_dummy = init_spec$init_chol_dummy[upper.tri(init_spec$init_chol_dummy, diag = FALSE)], # initial omega (vector)
    coef_spike = bayes_spec$coef_spike, # alpha spike (vector)
    coef_slab = bayes_spec$coef_slab, # alpha slab (vector)
    coef_slab_weight = bayes_spec$coef_mixture, # pj (vector)
    shape = bayes_spec$shape, # shape of gamma distn (vector)
    rate = bayes_spec$rate, # rate of gamma distn (vector)
    chol_spike = bayes_spec$chol_spike, # eta spike (vector)
    chol_slab = bayes_spec$chol_slab, # eta slab (vector)
    chol_slab_weight = bayes_spec$chol_mixture, # qij (vector)
    intercept_var = bayes_spec$coef_non # c for constant c I
  )
  class(ssvs_res) <- c("bvarssvs", "bvharmod")
  ssvs_res
}
