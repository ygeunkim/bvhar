#' Generate Multivariate Normal Random Vector
#' 
#' This function samples n x muti-dimensional normal random matrix.
#' 
#' @param num_sim Number to generate process
#' @param mu Mean vector
#' @param sig Variance matrix
#' @param method Method to compute \eqn{\Sigma^{1/2}}.
#' Choose between `"eigen"` (spectral decomposition) and `"chol"` (cholesky decomposition).
#' By default, `"eigen"`.
#' @export
sim_mnormal <- function(num_sim, mu = rep(0, 5), sig = diag(5), method = c("eigen", "chol")) {
  method <- match.arg(method)
  if (method == "eigen") {
    return( sim_mgaussian(num_sim, mu, sig) )
  } else {
    return( sim_mgaussian_chol(num_sim, mu, sig) )
  }
}

#' Generate Multivariate Time Series Process Following VAR(p)
#' 
#' This function generates multivariate time series dataset that follows VAR(p).
#' 
#' @param num_sim Number to generated process
#' @param num_burn Number of burn-in
#' @param var_coef VAR coefficient. The format should be the same as the output of [coef.varlse()] from [var_lm()]
#' @param var_lag Lag of VAR
#' @param sig_error Variance matrix of the error term. By default, `diag(dim)`.
#' @param init Initial y1, ..., yp matrix to simulate VAR model. Try `matrix(0L, nrow = var_lag, ncol = dim)`.
#' @param method `r lifecycle::badge("experimental")` Method to compute \eqn{\Sigma^{1/2}}.
#' Choose between `"eigen"` (spectral decomposition) and `"chol"` (cholesky decomposition).
#' By default, `"eigen"`.
#' @details 
#' 1. Generate \eqn{\epsilon_1, \epsilon_n \sim N(0, \Sigma)}
#' 2. For i = 1, ... n,
#' \deqn{y_{p + i} = (y_{p + i - 1}^T, \ldots, y_i^T, 1)^T B + \epsilon_i}
#' 3. Then the output is \eqn{(y_{p + 1}, \ldots, y_{n + p})^T}
#' 
#' Initial values might be set to be zero vector or \eqn{(I_m - A_1 - \cdots - A_p)^{-1} c}.
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' @export
sim_var <- function(num_sim, 
                    num_burn, 
                    var_coef, 
                    var_lag, 
                    sig_error = diag(ncol(var_coef)), 
                    init = matrix(0L, nrow = var_lag, ncol = ncol(var_coef)), 
                    method = c("eigen", "chol")) {
  method <- match.arg(method)
  dim_data <- ncol(sig_error)
  if (num_sim < 2) {
    stop("Generate more than 1 series")
  }
  if (nrow(var_coef) != dim_data * var_lag + 1 && nrow(var_lag) != dim_data * var_lag) {
    stop("'var_coef' is not VAR coefficient. Check its dimension.")
  }
  if (ncol(var_coef) != dim_data) {
    stop("Wrong 'var_coef' or 'sig_error' format.")
  }
  if (!(nrow(init) == var_lag && ncol(init) == dim_data)) {
    stop("'init' is (var_lag, dim) matrix in order of y1, y2, ..., yp.")
  }
  if (method == "eigen") {
    return( sim_var_eigen(num_sim, num_burn, var_coef, var_lag, sig_error, init) )
  } else {
    return( sim_var_chol(num_sim, num_burn, var_coef, var_lag, sig_error, init) )
  }
}
