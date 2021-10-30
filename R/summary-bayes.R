#' Summarizing Bayesian VAR of Minnesota Prior Model
#' 
#' `summary` method for `bvarmn` class.
#' 
#' @param object `bvarmn` object
#' @param n_iter Number to sample Matrix Normal Inverse-Wishart distribution
#' @param ... not used
#' @details 
#' From Minnesota prior, set of coefficient matrices and residual covariance matrix have matrix Normal Inverse-Wishart distribution.
#' 
#' \deqn{(B, \Sigma) \sim MNIW(\hat{B}, \hat{U}, \hat{\Sigma}, \alpha_0 + n + 2)}
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
summary.bvarmn <- function(object, n_iter = 100L, ...) {
  mn_mean <- object$coefficients
  mn_prec <- object$mn_prec
  iw_scale <- object$iw_scale
  nu <- object$iw_shape
  coef_and_sig <- sim_mniw(
    n_iter,
    mn_mean, # mean of MN
    solve(mn_prec), # precision of MN
    iw_scale, # scale of IW
    nu # shape of IW
  )
  dim_design <- object$df
  dim_data <- ncol(object$y0)
  Ahat <- 
    coef_and_sig$mn %>% # k x (n_iter * m)
    array(dim = c(dim_design, dim_data, n_iter))
  Sighat <- 
    coef_and_sig$iw %>% # m x (n_iter * m)
    array(dim = c(dim_data, dim_data, n_iter))
  # list of 3d array---------
  dimnames(Ahat) <- list(
    rownames(mn_mean), # row
    colnames(mn_mean), # col
    1:n_iter # 3rd dim
  )
  dimnames(Sighat) <- list(
    rownames(iw_scale), # row
    colnames(iw_scale), # col
    1:n_iter # 3rd dim
  )
  res <- list(
    names = colnames(object$y0),
    p = object$p,
    m = object$m,
    call = object$call,
    # posterior------------
    mn_mean = mn_mean,
    mn_prec = mn_prec,
    iw_scale = iw_scale,
    iw_shape = nu,
    # density--------------
    coefficients = Ahat,
    covmat = Sighat,
    N = n_iter
  )
  class(res) <- "summary.bvarmn"
  res
}
