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
