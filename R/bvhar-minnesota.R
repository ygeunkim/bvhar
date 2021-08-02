#' Fit Bayesian BVHAR of Minnesota Prior
#' 
#' @description
#' This function fits BVAR(p) with Minnesota prior.
#' 
#' @param y matrix, Time series data of which columns indicate the variables
#' @param sigma standard error for each variable
#' @param lambda tightness of the prior around a random walk or white noise
#' @param delta prior belief about white noise (Litterman sets 1: default)
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
#' \item{\code{mn_scale}}{First scale matrix of posterior matrix normal distribution}
#' \item{\code{iw_mean}}{Scale matrix of posterior inverse-wishart distribution}
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
bvhar_minnesota <- function(y, sigma, lambda, delta, eps = 1e-04) {
  if (!is.matrix(y)) y <- as.matrix(y)
  if (missing(delta)) delta <- rep(1, ncol(y))
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, 22, 23)
  name_var <- colnames(y)
  colnames(Y0) <- name_var
  X0 <- build_design(y, 22)
  # X1 = X0 %*% t(HARtrans)
  HARtrans <- scale_har(ncol(y))
  X1 <- AAt_eigen(X0, HARtrans)
  name_har <- lapply(
    c("day", "week", "month"),
    function(lag) paste(name_var, lag, sep = "_")
  ) %>% 
    unlist() %>% 
    c(., "const")
  colnames(X1) <- name_har
  # dummy-----------------------------
  Yh <- build_ydummy(3, sigma, lambda, delta)
  colnames(Yh) <- name_var
  Xh <- build_xdummy(3, lambda, sigma, eps)
  colnames(Xh) <- name_har
  # Matrix normal---------------------
  posterior <- estimate_bvar_mn(X1, Y0, Xh, Yh)
  Phihat <- posterior$bhat # posterior mean
  colnames(Phihat) <- name_var
  rownames(Phihat) <- name_har
  Uhat <- posterior$mnscale
  colnames(Uhat) <- name_har
  rownames(Uhat) <- name_har
  yhat <- posterior$fitted
  colnames(yhat) <- name_var
  # Inverse-wishart-------------------
  Sighat <- posterior$iwscale
  colnames(Sighat) <- name_var
  rownames(Sighat) <- name_var
  # S3--------------------------------
  res <- list(
    design = X0,
    y0 = Y0,
    y = y,
    m = ncol(y), # m
    obs = nrow(Y0), # s = n - p
    totobs = nrow(y), # n
    process = "BVHAR",
    call = match.call(),
    mn_mean = Phihat,
    fitted.values = yhat,
    # residuals = Y0 - yhat,
    mn_scale = Uhat,
    iw_scale = Sighat
  )
  class(res) <- "bvharmn"
  res
}
