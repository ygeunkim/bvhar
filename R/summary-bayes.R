#' Generate Coefficient Matrix and Covariance Matrix from Bayesian Model
#' 
#' @param x object
#' @param ... not used
#' 
#' @export
gen_posterior <- function(x, ...) {
  UseMethod("gen_posterior", x)
}

#' Posterior Distribution of Minnesota BVAR(p)
#' 
#' @description 
#' Generates Parameters of Minnesota BVAR \eqn{B, \Sigma_e}.
#' 
#' @param object \code{bvarmn} object
#' @param n number to generate (By default, 100)
#' @param ... not used
#' 
#' @details 
#' From Minnesota prior, set of coefficient matrices and residual covariance matrix have matrix Normal Inverse-Wishart distribution.
#' 
#' \deqn{(B, \Sigma) \sim MNIW(\hat{B}, \hat{U}, \hat{\Sigma}, \alpha_0 + n + 2)}
#' 
#' @references 
#' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
#' 
#' @importFrom mniw rmniw
#' @export
gen_posterior.bvarmn <- function(object, n = 100, ...) {
  mn_mean <- object$mn_mean
  mn_prec <- object$mn_prec
  iw_scale <- object$iw_scale
  nu <- object$a0 + object$obs + 2
  b_sig <- rmniw(n = n, Lambda = mn_mean, Omega = mn_prec, Psi = iw_scale, nu = nu)
  Bhat <- b_sig$X
  Sighat <- b_sig$V
  # mniw returns list of 3d array---------
  Bhat <- lapply(
    1:n,
    function(x) {
      b <- Bhat[,, x]
      rownames(b) <- rownames(mn_mean)
      colnames(b) <- colnames(mn_mean)
      b
    }
  )
  Sighat <- lapply(
    1:n,
    function(x) {
      sig <- Sighat[,, x]
      rownames(sig) <- rownames(iw_scale)
      colnames(sig) <- colnames(iw_scale)
      sig
    }
  )
  res <- list(
    coefficients = Bhat,
    covmat = Sighat
  )
  class(res) <- "minnesota"
  res
}
