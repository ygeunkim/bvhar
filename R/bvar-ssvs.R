#' Fitting Bayesian VAR(p) of SSVS Prior
#' 
#' This function fits BVAR(p) with stochastic search variable selection (SSVS) prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param bayes_spec A BVAR model specification by [set_ssvs()].
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
bvar_ssvs <- function(y, p, bayes_spec = set_ssvs(), include_mean = TRUE) {
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
  if (length(bayes_spec$coef_spike) != dim_data * dim_design) {
    stop("Invalid model specification.")
  }
  if (length(bayes_spec$coef_mixture) != dim_data * dim_design) {
    stop("Invalid model specification.")
  }
  if (length(bayes_spec$cov_shape) != dim_data) {
    stop("Invalid model specification.")
  }
  if (length(bayes_spec$cov_spike) != dim_data * (dim_data - 1) / 2) {
    stop("Invalid model specification.")
  }
  if (length(bayes_spec$cov_mixture) != dim_data * (dim_data - 1) / 2) {
    stop("Invalid model specification.")
  }
  # for Initial values---------------
  3 # temporary
}
