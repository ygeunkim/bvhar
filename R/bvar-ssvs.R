#' Fitting Bayesian VAR(p) of SSVS Prior
#' 
#' This function fits BVAR(p) with stochastic search variable selection (SSVS) prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param bayes_spec A BVAR model specification by [set_ssvs()].
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' 
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
#' 
bvar_ssvs <- function(y, 
                      p, 
                      num_iter, 
                      num_burn, 
                      num_thin = 1L, 
                      bayes_spec = set_ssvs(), 
                      include_mean = TRUE) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  # model specification---------------
  if (!is.bvharss_spec(bayes_spec)) {
    stop("Provide 'bvharss_spec' for 'bayes_spec'.")
  }
  # Y0 = X0 B + Z---------------------
  dim_data <- ncol(y)
  num_design <- nrow(Y0)
  dim_design <- dim_data * p + 1
  Y0 <- build_y0(y, p, p + 1)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
  }
  colnames(Y0) <- name_var
  X0 <- build_design(y, p)
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
  # error for bvharss_spec-----------
  if ((nrow(bayes_spec$init_coef) != dim_design) || (ncol(bayes_spec$init_coef) != dim_data)) {
    stop("Invalid model specification.")
  }
  # for Initial values---------------
  if (all(is.na(bayes_spec$coef_spike)) || all(is.na(bayes_spec$coef_slab))) {
    y_vec <- vectorize_eigen(Y0) # Y: m x 1
    reg_design <- 
      kronecker_eigen(diag(dim_data), X0) %>% # X = Im otimes X0: ms x mk
      qr_eigen() # QR, Q: ms x mk, R: mk x mk
    # SSE = Y^T (I - HAT) Y, HAT = X (X^T X)^(-1) X^T = QQ^T
    sse <- y_vec %*% (diag(dim_data * dim_design) - tcrossprod(reg_design$orthogonal)) %*% t(y_vec)
    # SSE / df * (X^T X)^(-1) = SSE / df * (R^T R)^(-1), df = ms - mk + 1
    ols_var <- diag(sse * reg_design$upper / (dim_data * (num_design - dim_design) + 1))
    bayes_spec$coef_spike <- .1 * sqrt(ols_var) # c0 sqrt(var)
    bayes_spec$coef_slab <- 10 * sqrt(ols_var) # c1 sqrt(var)
  }
  # when cov_spike and cov_slab are not specified
  
  # MCMC-----------------------------
  # ssvs_res <- estimate_bvar_ssvs(
  #   num_iter,
  #   X0,
  #   Y0,
  #   
  # )
  3
}
