#' Generate Multivariate Normal Random Vector
#' 
#' This function samples n x muti-dimensional normal random matrix.
#' 
#' @param num_sim Number to generate process
#' @param mu Mean vector
#' @param sig Variance matrix
#' @param method Method to compute \eqn{\Sigma^{1/2}}.
#' Choose between `eigen` (spectral decomposition) and `chol` (cholesky decomposition).
#' By default, `eigen`.
#' @details
#' Consider \eqn{x_1, \ldots, x_n \sim N_m (\mu, \Sigma)}.
#' 
#' 1. Lower triangular Cholesky decomposition: \eqn{\Sigma = L L^T}
#' 2. Standard normal generation: \eqn{Z_{i1}, Z_{in} \stackrel{iid}{\sim} N(0, 1)}
#' 3. \eqn{Z_i = (Z_{i1}, \ldots, Z_{in})^T}
#' 4. \eqn{X_i = L Z_i + \mu}
#' @return T x k matrix
#' @export
sim_mnormal <- function(num_sim, mu = rep(0, 5), sig = diag(5), method = c("eigen", "chol")) {
  method <- match.arg(method)
  if (!all.equal(unname(sig), unname(t(sig)))) {
    stop("'sig' must be a symmetric matrix.")
  }
  if (method == "eigen") {
    return( sim_mgaussian(num_sim, mu, sig) )
  }
  sim_mgaussian_chol_export(num_sim, mu, sig)
}

#' Generate Multivariate t Random Vector
#' 
#' This function samples n x multi-dimensional t-random matrix.
#' 
#' @param num_sim Number to generate process.
#' @param df Degrees of freedom.
#' @param mu Location vector
#' @param sig Scale matrix.
#' @param method Method to compute \eqn{\Sigma^{1/2}}.
#' Choose between `eigen` (spectral decomposition) and `chol` (cholesky decomposition).
#' By default, `eigen`.
#' @return T x k matrix
#' @export
sim_mvt <- function(num_sim, df, mu, sig, method = c("eigen", "chol")) {
  method <- match.arg(method)
  if (!all.equal(unname(sig), unname(t(sig)))) {
    stop("'sig' must be a symmetric matrix.")
  }
  if (method == "eigen") {
    return( sim_mstudent(num_sim, df, mu, sig, 1) )
  }
  sim_mstudent(num_sim, df, mu, sig, 2)
}

#' Generate Multivariate Time Series Process Following VAR(p)
#' 
#' This function generates multivariate time series dataset that follows VAR(p).
#' 
#' @param num_sim Number to generated process
#' @param num_burn Number of burn-in
#' @param var_coef VAR coefficient. The format should be the same as the output of [coef()] from [var_lm()]
#' @param var_lag Lag of VAR
#' @param sig_error Variance matrix of the error term. By default, `diag(dim)`.
#' @param init Initial y1, ..., yp matrix to simulate VAR model. Try `matrix(0L, nrow = var_lag, ncol = dim)`.
#' @param method Method to compute \eqn{\Sigma^{1/2}}.
#' Choose between `eigen` (spectral decomposition) and `chol` (cholesky decomposition).
#' By default, `eigen`.
#' @param process Process to generate error term.
#' `gaussian`: Normal distribution (default) or `student`: Multivariate t-distribution.
#' @param t_param `r lifecycle::badge("experimental")` argument for MVT, e.g. DF: 5.
#' @details 
#' 1. Generate \eqn{\epsilon_1, \epsilon_n \sim N(0, \Sigma)}
#' 2. For i = 1, ... n,
#' \deqn{y_{p + i} = (y_{p + i - 1}^T, \ldots, y_i^T, 1)^T B + \epsilon_i}
#' 3. Then the output is \eqn{(y_{p + 1}, \ldots, y_{n + p})^T}
#' 
#' Initial values might be set to be zero vector or \eqn{(I_m - A_1 - \cdots - A_p)^{-1} c}.
#' @return T x k matrix
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
#' @export
sim_var <- function(num_sim, 
                    num_burn, 
                    var_coef, 
                    var_lag, 
                    sig_error = diag(ncol(var_coef)), 
                    init = matrix(0L, nrow = var_lag, ncol = ncol(var_coef)), 
                    method = c("eigen", "chol"),
                    process = c("gaussian", "student"),
                    t_param = 5) {
  method <- match.arg(method)
  process <- match.arg(process)
  process <- switch(process, "gaussian" = 1, "student" = 2)
  dim_data <- ncol(sig_error)
  if (num_sim < 2) {
    stop("Generate more than 1 series")
  }
  if (nrow(var_coef) != dim_data * var_lag + 1 && nrow(var_coef) != dim_data * var_lag) {
    stop("'var_coef' is not VAR coefficient. Check its dimension.")
  }
  if (ncol(var_coef) != dim_data) {
    stop("Wrong 'var_coef' or 'sig_error' format.")
  }
  if (!all.equal(unname(sig_error), unname(t(sig_error)))) {
    stop("'sig_error' must be a symmetric matrix.")
  }
  if (!(nrow(init) == var_lag && ncol(init) == dim_data)) {
    stop("'init' is (var_lag, dim) matrix in order of y1, y2, ..., yp.")
  }
  if (method == "eigen") {
    return( sim_var_eigen(num_sim, num_burn, var_coef, var_lag, sig_error, init, process, t_param) )
  }
  sim_var_chol(num_sim, num_burn, var_coef, var_lag, sig_error, init, process, t_param)
}

#' Generate Normal-IW Random Family
#'
#' This function samples normal inverse-wishart matrices.
#'
#' @param num_sim Number to generate
#' @param mat_mean Mean matrix of MN
#' @param mat_scale_u First scale matrix of MN
#' @param mat_scale Scale matrix of IW
#' @param shape Shape of IW
#' @param u_prec If `TRUE`, use `mat_scale_u` as its inverse. By default, `FALSE`.
#' @details
#' Consider \eqn{(Y_i, \Sigma_i) \sim MIW(M, U, \Psi, \nu)}.
#'
#' 1. Generate upper triangular factor of \eqn{\Sigma_i = C_i C_i^T} in the upper triangular Bartlett decomposition.
#' 2. Standard normal generation: n x k matrix \eqn{Z_i = [z_{ij} \sim N(0, 1)]} in row-wise direction.
#' 3. Lower triangular Cholesky decomposition: \eqn{U = P P^T}
#' 4. \eqn{A_i = M + P Z_i C_i^T}
#' @export
sim_mniw <- function(num_sim, mat_mean, mat_scale_u, mat_scale, shape, u_prec = FALSE) {
  res <-
    sim_mniw_export(num_sim, mat_mean, mat_scale_u, mat_scale, shape, u_prec) |>
    simplify2array() |>
    apply(1, function(x) x)
  names(res) <- c("mn", "iw")
  res
}

#' Generate Multivariate Time Series Process Following VAR(p)
#' 
#' This function generates multivariate time series dataset that follows VAR(p).
#' 
#' @param num_sim Number to generated process
#' @param num_burn Number of burn-in
#' @param vhar_coef VAR coefficient. The format should be the same as the output of [coef()] from [var_lm()]
#' @param week Weekly order of VHAR. By default, `5`.
#' @param month Weekly order of VHAR. By default, `22`.
#' @param sig_error Variance matrix of the error term. By default, `diag(dim)`.
#' @param init Initial y1, ..., yp matrix to simulate VAR model. Try `matrix(0L, nrow = month, ncol = dim)`.
#' @param method Method to compute \eqn{\Sigma^{1/2}}.
#' Choose between `eigen` (spectral decomposition) and `chol` (cholesky decomposition).
#' By default, `eigen`.
#' @param process Process to generate error term.
#' `gaussian`: Normal distribution (default) or `student`: Multivariate t-distribution.
#' @param t_param `r lifecycle::badge("experimental")` argument for MVT, e.g. DF: 5.
#' @details 
#' Let \eqn{M} be the month order, e.g. \eqn{M = 22}.
#' 
#' 1. Generate \eqn{\epsilon_1, \epsilon_n \sim N(0, \Sigma)}
#' 2. For i = 1, ... n,
#' \deqn{y_{M + i} = (y_{M + i - 1}^T, \ldots, y_i^T, 1)^T C_{HAR}^T \Phi + \epsilon_i}
#' 3. Then the output is \eqn{(y_{M + 1}, \ldots, y_{n + M})^T}
#' 
#' 2. For i = 1, ... n,
#' \deqn{y_{p + i} = (y_{p + i - 1}^T, \ldots, y_i^T, 1)^T B + \epsilon_i}
#' 3. Then the output is \eqn{(y_{p + 1}, \ldots, y_{n + p})^T}
#' 
#' Initial values might be set to be zero vector or \eqn{(I_m - A_1 - \cdots - A_p)^{-1} c}.
#' @return T x k matrix
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
#' @export
sim_vhar <- function(num_sim, 
                     num_burn, 
                     vhar_coef, 
                     week = 5L,
                     month = 22L,
                     sig_error = diag(ncol(vhar_coef)), 
                     init = matrix(0L, nrow = month, ncol = ncol(vhar_coef)), 
                     method = c("eigen", "chol"),
                     process = c("gaussian", "student"),
                     t_param = 5) {
  method <- match.arg(method)
  process <- match.arg(process)
  process <- switch(process, "gaussian" = 1, "student" = 2)
  dim_data <- ncol(sig_error)
  if (num_sim < 2) {
    stop("Generate more than 1 series")
  }
  if (nrow(vhar_coef) != 3 * dim_data + 1 && nrow(vhar_coef) != 3 * dim_data) {
    stop("'vhar_coef' is not VHAR coefficient. Check its dimension.")
  }
  if (ncol(vhar_coef) != dim_data) {
    stop("Wrong 'var_coef' or 'sig_error' format.")
  }
  if (!all.equal(unname(sig_error), unname(t(sig_error)))) {
    stop("'sig_error' must be a symmetric matrix.")
  }
  if (!(nrow(init) == month && ncol(init) == dim_data)) {
    stop("'init' is (month, dim) matrix in order of y1, y2, ..., y_month.")
  }
  if (method == "eigen") {
    return( sim_vhar_eigen(num_sim, num_burn, vhar_coef, week, month, sig_error, init, process, t_param) )
  }
  sim_vhar_chol(num_sim, num_burn, vhar_coef, week, month, sig_error, init, process, t_param)
}
