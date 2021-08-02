#' Fit Bayesian VAR(p) of Minnesota Prior
#' 
#' @description
#' This function fits BVAR(p) with Minnesota prior.
#' 
#' @param y matrix, Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param sigma standard error for each variable
#' @param lambda tightness of the prior around a random walk or white noise
#' @param delta prior belief about white noise (Litterman sets 1: default)
#' @param eps very small number
#' 
#' @details 
#' Minnesota prior give prior to parameters \eqn{B} (VAR matrices) and \eqn{\Sigma_e} (residual covariance).
#' 
#' \deqn{B \mid \Sigma_e, Y_0 \sim MN(B_0, \Omega_0, \Sigma_e)}
#' \deqn{\Sigma_e \mid Y_0 \sim IW(S_0, \alpha_0)}
#' (MN: \href{https://en.wikipedia.org/wiki/Matrix_normal_distribution}{matrix normal}, IW: \href{https://en.wikipedia.org/wiki/Inverse-Wishart_distribution}{inverse-wishart})
#' 
#' @return \code{bvarmn} object with
#' \item{\code{design}}{\eqn{X_0}}
#' \item{\code{y0}}{\eqn{Y_0}}
#' \item{\code{y}}{raw input}
#' \item{\code{p}}{lag of VAR: p}
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
#' 
#' @references 
#' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
#' 
#' @order 1
#' @export
bvar_minnesota <- function(y, p, sigma, lambda, delta, eps = 1e-04) {
  if (!is.matrix(y)) y <- as.matrix(y)
  if (missing(delta)) delta <- rep(1, ncol(y))
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, p, p + 1)
  name_var <- colnames(y)
  colnames(Y0) <- name_var
  X0 <- build_design(y, p)
  name_lag <- lapply(
    p:1,
    function(lag) paste(name_var, lag, sep = "_")
  ) %>% 
    unlist() %>% 
    c(., "const")
  colnames(X0) <- name_lag
  # dummy-----------------------------
  Yp <- build_ydummy(p, sigma, lambda, delta)
  colnames(Yp) <- name_var
  Xp <- build_xdummy(p, lambda, sigma, eps)
  colnames(Xp) <- name_lag
  # Matrix normal---------------------
  posterior <- estimate_bvar_mn(X0, Y0, Xp, Yp)
  Bhat <- posterior$bhat # posterior mean
  colnames(Bhat) <- name_var
  rownames(Bhat) <- name_lag
  Uhat <- posterior$mnscale
  colnames(Uhat) <- name_lag
  rownames(Uhat) <- name_lag
  yhat <- posterior$fitted
  colnames(yhat) <- name_var
  # zhat <- Y0 - yhat
  # Inverse-wishart-------------------
  Sighat <- posterior$iwscale
  colnames(Sighat) <- name_var
  rownames(Sighat) <- name_var
  # S3--------------------------------
  res <- list(
    design = X0,
    y0 = Y0,
    y = y,
    p = p, # p
    m = ncol(y), # m
    obs = nrow(Y0), # s = n - p
    totobs = nrow(y), # n
    process = "Minnesota",
    call = match.call(),
    mn_mean = Bhat,
    fitted.values = yhat,
    # residuals = Y0 - yhat,
    mn_scale = Uhat,
    iw_scale = Sighat
  )
  class(res) <- "bvarmn"
  res
}
