#' Fit Bayesian VAR(p) of Minnesota Prior
#' 
#' @description
#' This function fits BVAR(p) with Minnesota prior.
#' 
#' @param y matrix, Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param sigma standard error for each variable
#' @param lambda tightness of the prior around a random walk or white noise
#' @param delta Persistence (Litterman sets 1 = random walk prior, Default: White noise prior = 0)
#' @param eps very small number
#' 
#' @details 
#' Minnesota prior give prior to parameters \eqn{B} (VAR matrices) and \eqn{\Sigma_e} (residual covariance).
#' 
#' \deqn{B \mid \Sigma_e \sim MN(B_0, \Omega_0, \Sigma_e)}
#' \deqn{\Sigma_e \sim IW(S_0, \alpha_0)}
#' (MN: \href{https://en.wikipedia.org/wiki/Matrix_normal_distribution}{matrix normal}, IW: \href{https://en.wikipedia.org/wiki/Inverse-Wishart_distribution}{inverse-wishart})
#' 
#' @return \code{bvar_minnesota} returns an object \code{bvarmn} \link{class}.
#' 
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{design}{\eqn{X_0}}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{y}{Raw input}
#'   \item{p}{Lag of VAR}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = \code{totobs} - \code{p}}
#'   \item{totobs}{Total number of the observation}
#'   \item{process}{Process: Minnesota}
#'   \item{call}{Matched call}
#'   \item{mn_mean}{Location of posterior matrix normal distribution}
#'   \item{fitted.values}{Fitted values}
#'   \item{residuals}{Residuals}
#'   \item{mn_prec}{Precision matrix of posterior matrix normal distribution}
#'   \item{iw_scale}{Scale matrix of posterior inverse-wishart distribution}
#'   \item{a0}{\eqn{\alpha_0}: nrow(Dummy observation) - k}
#' }
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
  if (missing(delta)) delta <- rep(0, ncol(y))
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, p, p + 1)
  name_var <- colnames(y)
  colnames(Y0) <- name_var
  X0 <- build_design(y, p)
  name_lag <- concatenate_colnames(name_var, p:1) # in misc-r.R file
  colnames(X0) <- name_lag
  m <- ncol(y)
  s <- nrow(Y0)
  # dummy-----------------------------
  Yp <- build_ydummy(p, sigma, lambda, delta)
  colnames(Yp) <- name_var
  Xp <- build_xdummy(p, lambda, sigma, eps)
  colnames(Xp) <- name_lag
  # estimate-bvar.cpp-----------------
  posterior <- estimate_bvar_mn(X0, Y0, Xp, Yp)
  # Prior-----------------------------
  B0 <- posterior$prior_mean
  U0 <- posterior$prior_precision
  S0 <- posterior$prior_scale
  a0 <- posterior$prior_shape
  # Matrix normal---------------------
  Bhat <- posterior$bhat # matrix normal mean
  colnames(Bhat) <- name_var
  rownames(Bhat) <- name_lag
  Uhat <- posterior$mnprec # matrix normal precision
  colnames(Uhat) <- name_lag
  rownames(Uhat) <- name_lag
  yhat <- posterior$fitted
  colnames(yhat) <- name_var
  # Inverse-wishart-------------------
  Sighat <- posterior$iwscale # IW scale
  colnames(Sighat) <- name_var
  rownames(Sighat) <- name_var
  # S3--------------------------------
  res <- list(
    design = X0,
    y0 = Y0,
    y = y,
    p = p, # p
    m = m, # m = dimension of Y_t
    df = m * p + 1, # k = m * p + 1
    obs = s, # s = n - p
    totobs = nrow(y), # n = total number of sample size
    process = "BVAR_Minnesota",
    call = match.call(),
    # prior----------------
    prior_mean = B0, # B0
    prior_precision = U0, # U0 = (Omega)^{-1}
    prior_scale = S0, # S0
    prior_shape = a0 + (m + 3), # add (m + 3) for prior mean existence
    # posterior------------
    mn_mean = Bhat,
    fitted.values = yhat,
    residuals = Y0 - yhat,
    mn_prec = Uhat,
    iw_scale = Sighat,
    iw_shape = a0 + s + 2
  )
  class(res) <- "bvarmn"
  res
}

#' See if the Object \code{bvarmn}
#' 
#' This function returns \code{TRUE} if the input is the output of \code{\link{bvar_minnesota}}.
#' 
#' @param x Object
#' 
#' @return \code{TRUE} or \code{FALSE}
#' 
#' @export
is.bvarmn <- function(x) {
  inherits(x, "bvarmn")
}

#' Coefficients Method for \code{bvarmn} object
#' 
#' Matrix Normal mean of Minnesota BVAR
#' 
#' @param object \code{bvarmn} object
#' @param ... not used
#' 
#' @export
coef.bvarmn <- function(object, ...) {
  object$mn_mean
}

#' Residuals Method for \code{bvarmn} object
#' 
#' @param object \code{bvarmn} object
#' @param ... not used
#' 
#' @export
residuals.bvarmn <- function(object, ...) {
  object$residuals
}

#' Fitted Values Method for \code{bvarmn} object
#' 
#' @param object \code{bvarmn} object
#' @param ... not used
#' 
#' @export
fitted.bvarmn <- function(object, ...) {
  object$fitted.values
}
