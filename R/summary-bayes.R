#' Summarizing Bayesian Multivariate Time Series Model
#' 
#' `summary` method for `normaliw` class.
#' 
#' @param object `normaliw` object
#' @param n_iter Number to sample Matrix Normal Inverse-Wishart distribution
#' @param ... not used
#' @details 
#' From Minnesota prior, set of coefficient matrices and residual covariance matrix have matrix Normal Inverse-Wishart distribution.
#' 
#' BVAR:
#' 
#' \deqn{(A, \Sigma_e) \sim MNIW(\hat{A}, \hat{V}^{-1}, \hat\Sigma_e, \alpha_0 + s)}
#' where \eqn{\hat{V} = X_\ast^T X_\ast} is the posterior precision of MN.
#' 
#' BVHAR:
#' 
#' \deqn{(\Phi, \Sigma_e) \sim MNIW(\hat\Phi, \hat{V}_H^{-1}, \hat\Sigma_e, d_0 + s)}
#' where \eqn{\hat{V}_H = X_{+}^T X_{+}} is the posterior precision of MN.
#' 
#' @return `summary` for `bvarmn` object returns `summary.bvarmn` [class].
#' \describe{
#'   \item{coefficients}{iter x k x m array: each column of the array indicate the draw for each lag corresponding to that variable}
#'   \item{covmat}{iter x m x m array: each column of teh array indicate the draw for each varable corresponding to that variable}
#' }
#' 
#' @references 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
#' 
#' @order 1
#' @export
summary.normaliw <- function(object, n_iter = 100L, ...) {
  mn_mean <- object$coefficients
  mn_prec <- object$mn_prec
  iw_scale <- object$iw_scale
  nu <- object$iw_shape
  coef_and_sig <- sim_mniw(
    n_iter,
    mn_mean, # mean of MN
    solve(mn_prec), # precision of MN = inverse of precision
    iw_scale, # scale of IW
    nu # shape of IW
  )
  dim_design <- object$df # k or h = 3m + 1 or 3m
  dim_data <- ncol(object$y0)
  coef_mat <- 
    coef_and_sig$mn %>% # k(h) x (n_iter * m)
    array(dim = c(dim_design, dim_data, n_iter))
  cov_mat <- 
    coef_and_sig$iw %>% # m x (n_iter * m)
    array(dim = c(dim_data, dim_data, n_iter))
  # list of 3d array---------
  dimnames(coef_mat) <- list(
    rownames(mn_mean), # row
    colnames(mn_mean), # col
    1:n_iter # 3rd dim
  )
  dimnames(cov_mat) <- list(
    rownames(iw_scale), # row
    colnames(iw_scale), # col
    1:n_iter # 3rd dim
  )
  res <- list(
    names = colnames(object$y0),
    p = object$p,
    m = object$m,
    call = object$call,
    spec = object$spec,
    # posterior------------
    mn_mean = mn_mean,
    mn_prec = mn_prec,
    iw_scale = iw_scale,
    iw_shape = nu,
    # density--------------
    coefficients = coef_mat,
    covmat = cov_mat,
    N = n_iter
  )
  class(res) <- "summary.normaliw"
  res
}

