#' Fit Bayesian BVHAR of Minnesota Prior
#' 
#' @description
#' This function fits BVAR(p) with Minnesota prior.
#' 
#' @param y matrix, Time series data of which columns indicate the variables
#' @param type Prior mean to apply. (VAR = Original Minnesota. VHAR = fills zero in the first block)
#' @param sigma standard error for each variable
#' @param lambda tightness of the prior around a random walk or white noise
#' @param delta prior belief about white noise (Litterman sets 1: default)
#' @param daily same as delta in VHAR type
#' @param weekly fill the second part in the first block
#' @param monthly fill the third part in the first block
#' @param eps very small number
#' 
#' @details 
#' Apply Minnesota prior to Vector HAR: \eqn{\Phi} (VHAR matrices) and \eqn{\Sigma_e} (residual covariance).
#' 
#' \deqn{\Phi \mid \Sigma_e, Y_0 \sim MN(B_0, \Omega_0, \Sigma_e)}
#' \deqn{\Sigma_e \mid Y_0 \sim IW(S_0, \alpha_0)}
#' (MN: \href{https://en.wikipedia.org/wiki/Matrix_normal_distribution}{matrix normal}, IW: \href{https://en.wikipedia.org/wiki/Inverse-Wishart_distribution}{inverse-wishart})
#' 
#' @return \code{bvarmn} object with
#' \item{\code{design}}{\eqn{X_0}}
#' \item{\code{y0}}{\eqn{Y_0}}
#' \item{\code{y}}{raw input}
#' \item{\code{m}}{Dimension of the data}
#' \item{\code{obs}}{Sample size used when training = \code{totobs} - \code{p}}
#' \item{\code{totobs}}{Total number of the observation}
#' \item{\code{process}}{Process: VAR}
#' \item{\code{call}}{Matched call}
#' \item{\code{mn_mean}}{Location of posterior matrix normal distribution}
#' \item{\code{fitted.values}}{Fitted values}
#' \item{\code{residuals}}{Residuals}
#' \item{\code{mn_scale}}{First scale matrix of posterior matrix normal distribution}
#' \item{\code{iw_mean}}{Scale matrix of posterior inverse-wishart distribution}
#' \item{\code{a0}}{\eqn{\alpha_0}: nrow(Dummy observation) - k}
#' 
#' @references 
#' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
#' 
#' Corsi, F. (2008). \emph{A Simple Approximate Long-Memory Model of Realized Volatility}. Journal of Financial Econometrics, 7(2), 174–196. \url{https://doi:10.1093/jjfinec/nbp001}
#' 
#' @order 1
#' @export
bvhar_minnesota <- function(y, type = c("VAR", "VHAR"), sigma, lambda, delta, daily, weekly, monthly, eps = 1e-04) {
  if (!is.matrix(y)) y <- as.matrix(y)
  type <- match.arg(type)
  # if (missing(delta)) delta <- rep(1, ncol(y))
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, 22, 23)
  name_var <- colnames(y)
  colnames(Y0) <- name_var
  X0 <- build_design(y, 22)
  # X1 = X0 %*% t(HARtrans)
  HARtrans <- scale_har(ncol(y))
  X1 <- AAt_eigen(X0, HARtrans)
  name_har <- concatenate_colnames(name_var, c("day", "week", "month")) # in misc-r.R file
  colnames(X1) <- name_har
  # dummy-----------------------------
  Yh <- switch(
    type,
    "VAR" = {
      if (missing(delta)) delta <- rep(1, ncol(y))
      Yh <- build_ydummy(3, sigma, lambda, delta)
      colnames(Yh) <- name_var
      Yh
    },
    "VHAR" = {
      if (missing(daily)) daily <- rep(1, ncol(y))
      if (missing(weekly)) weekly <- rep(1, ncol(y))
      if (missing(monthly)) monthly <- rep(1, ncol(y))
      Yh <- build_ydummy_bvhar(sigma, lambda, daily, weekly, monthly)
      colnames(Yh) <- name_var
      Yh
    }
  )
  # Yh <- build_ydummy(3, sigma, lambda, delta)
  # colnames(Yh) <- name_var
  Xh <- build_xdummy(3, lambda, sigma, eps)
  colnames(Xh) <- name_har
  # Matrix normal---------------------
  posterior <- estimate_bvar_mn(X1, Y0, Xh, Yh)
  Phihat <- posterior$bhat # posterior mean
  colnames(Phihat) <- name_var
  rownames(Phihat) <- name_har
  Uhat <- posterior$mnprec
  colnames(Uhat) <- name_har
  rownames(Uhat) <- name_har
  yhat <- posterior$fitted
  colnames(yhat) <- name_var
  # Inverse-wishart-------------------
  Sighat <- posterior$iwscale
  colnames(Sighat) <- name_var
  rownames(Sighat) <- name_var
  m <- ncol(y)
  a0 <- nrow(Xh) - 3 * m + 1
  # S3--------------------------------
  res <- list(
    design = X0,
    y0 = Y0,
    y = y,
    m = m, # m
    obs = nrow(Y0), # s = n - p
    totobs = nrow(y), # n
    process = "BVHAR",
    call = match.call(),
    HARtrans = HARtrans,
    mn_mean = Phihat,
    fitted.values = yhat,
    residuals = Y0 - yhat,
    mn_prec = Uhat,
    iw_scale = Sighat,
    a0 = a0
  )
  class(res) <- "bvharmn"
  res
}
