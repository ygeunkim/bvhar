#' Generate Minnesota BVAR Parameters
#' 
#' This function generates parameters of BVAR with Minnesota prior.
#' 
#' @param p VAR lag
#' @param sigma standard error for each variable
#' @param lambda tightness of the prior around a random walk or white noise
#' @param delta Persistence (Litterman sets 1 = random walk prior, Default: White noise prior = 0)
#' @param eps very small number
#' @details 
#' Implementing dummy observation constructions,
#' Bańbura et al. (2010) sets Normal-IW prior.
#' \deqn{B \mid \Sigma_e \sim MN(B_0, \Omega_0, \Sigma_e)}
#' \deqn{\Sigma_e \sim IW(S_0, \alpha_0)}
#' @seealso 
#' \code{\link{build_ydummy}} and \code{\link{build_xdummy}} for defining dummy observations
#' @references 
#' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
#' 
#' @importFrom mniw rmniw
#' @export
sim_mncoef <- function(p, sigma, lambda, delta, eps = 1e-04) {
  if (length(sigma) != length(delta)) stop("length of `sigma` and `delta` must be the same as the dimension of the time series")
  # dummy-----------------------------
  Yp <- build_ydummy(p, sigma, lambda, delta)
  Xp <- build_xdummy(p, lambda, sigma, eps)
  # prior-----------------------------
  prior <- minnesota_prior(Xp, Yp)
  mn_mean <- prior$prior_mean
  mn_prec <- prior$prior_prec
  iw_scale <- prior$prior_scale
  iw_shape <- prior$prior_shape
  # random---------------------------
  res <- rmniw(n = 1, Lambda = mn_mean, Omega = mn_prec, Psi = iw_scale, nu = iw_shape)
  list(
    coefficients = res$X,
    covmat = res$V
  )
}

#' Generate Minnesota BVAR Parameters
#' 
#' This function generates parameters of BVAR with Minnesota prior.
#' 
#' @param type Prior mean type to apply. ("VAR" = Original Minnesota. "VHAR" = fills zero in the first block)
#' @param sigma Standard error vector for each variable
#' @param lambda Tightness of the prior around a random walk or white noise
#' @param delta Prior belief about white noise (Litterman sets 1: default)
#' @param daily Same as delta in VHAR type
#' @param weekly Fill the second part in the first block
#' @param monthly Fill the third part in the first block
#' @param eps Very small number
#' @details 
#' Normal-IW family for vector HAR model:
#' \deqn{\Phi \mid \Sigma_e \sim MN(P_0, \Psi_0, \Sigma_e)}
#' \deqn{\Sigma_e \sim IW(U_0, d_0)}
#' @seealso 
#' \code{\link{build_ydummy}}, \code{\link{build_xdummy}}, and \code{\link{build_ydummy_bvhar}} for defining dummy observation.
#' @references 
#' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
#' 
#' Corsi, F. (2008). \emph{A Simple Approximate Long-Memory Model of Realized Volatility}. Journal of Financial Econometrics, 7(2), 174–196. \url{https://doi:10.1093/jjfinec/nbp001}
#' 
#' @importFrom mniw rmniw
#' @export
sim_mnvhar_coef <- function(type = c("VAR", "VHAR"), 
                            sigma, 
                            lambda, 
                            delta, 
                            daily, 
                            weekly, 
                            monthly, 
                            eps = 1e-04) {
  type <- match.arg(type)
  # dummy-----------------------------
  Yh <- switch(
    type,
    "VAR" = {
      if (length(sigma) != length(delta)) stop("length of `sigma` and `delta` must be the same as the dimension of the time series")
      Yh <- build_ydummy(3, sigma, lambda, delta)
      Yh
    },
    "VHAR" = {
      if (length(sigma) != length(daily)) stop("length of `sigma` and `daily` must be the same as the dimension of the time series")
      if (length(sigma) != length(weekly)) stop("length of `sigma` and `weekly` must be the same as the dimension of the time series")
      if (length(sigma) != length(monthly)) stop("length of `sigma` and `monthly` must be the same as the dimension of the time series")
      Yh <- build_ydummy_bvhar(sigma, lambda, daily, weekly, monthly)
      Yh
    }
  )
  Xh <- build_xdummy(3, lambda, sigma, eps)
  # prior-----------------------------
  prior <- minnesota_prior(Xh, Yh)
  mn_mean <- prior$prior_mean
  mn_prec <- prior$prior_prec
  iw_scale <- prior$prior_scale
  iw_shape <- prior$prior_shape
  # random---------------------------
  res <- rmniw(n = 1, Lambda = mn_mean, Omega = mn_prec, Psi = iw_scale, nu = iw_shape)
  list(
    coefficients = res$X,
    covmat = res$V
  )
}

