#' Summary of \code{\link{bvar_minnesota}}
#' 
#' @param object \code{bvarmn} object
#' @param n_iter Number to sample Matrix Normal Inverse-Wishart distribution
#' @param ... not used
#' @details 
#' From Minnesota prior, set of coefficient matrices and residual covariance matrix have matrix Normal Inverse-Wishart distribution.
#' 
#' \deqn{(B, \Sigma) \sim MNIW(\hat{B}, \hat{U}, \hat{\Sigma}, \alpha_0 + n + 2)}
#' 
#' @return \code{gen_posterior} for \code{bvarmn} object returns \code{minnesota} \link{class}.
#' \describe{
#'   \item{coefficients}{iter x k x m array: each column of the array indicate the draw for each lag corresponding to that variable}
#'   \item{covmat}{iter x m x m array: each column of teh array indicate the draw for each varable corresponding to that variable}
#' }
#' 
#' @references 
#' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
#' 
#' @importFrom mniw rmniw
#' @export
summary.bvarmn <- function(object, n_iter = 100, ...) {
  mn_mean <- object$mn_mean
  mn_prec <- object$mn_prec
  iw_scale <- object$iw_scale
  nu <- object$iw_shape
  b_sig <- rmniw(n = n_iter, Lambda = mn_mean, Omega = mn_prec, Psi = iw_scale, nu = nu)
  Bhat <- b_sig$X
  Sighat <- b_sig$V
  # mniw returns list of 3d array---------
  dimnames(Bhat) <- list(
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
    coefficients = Bhat,
    covmat = Sighat,
    N = n_iter
  )
  class(res) <- "summary.bvarmn"
  res
}
