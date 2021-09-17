#' Fit Bayesian VAR(p) of Minnesota Prior
#' 
#' This function fits BVAR(p) with Minnesota prior.
#' 
#' @param y matrix, Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param sigma standard error for each variable (Default: sd)
#' @param lambda tightness of the prior around a random walk or white noise (Default: .1)
#' @param delta Persistence (Litterman sets 1 = random walk prior, Default: White noise prior = 0)
#' @param eps very small number
#' @param type `r lifecycle::badge("experimental")` add constant term (`"const"`) or not (`"none"`)
#' 
#' @details 
#' Minnesota prior give prior to parameters \eqn{B} (VAR matrices) and \eqn{\Sigma_e} (residual covariance).
#' 
#' \deqn{B \mid \Sigma_e \sim MN(B_0, \Omega_0, \Sigma_e)}
#' \deqn{\Sigma_e \sim IW(S_0, \alpha_0)}
#' (MN: [matrix normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution), IW: [inverse-wishart](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution))
#' 
#' \eqn{\delta_i} are related to the belief to random walk.
#' 
#' * If \eqn{\delta_i = 1} for all i, random walk prior
#' * If \eqn{\delta_i = 0} for all i, white noise prior
#' 
#' \eqn{\lambda} controls the overall tightness of the prior around these two prior beliefs.
#' 
#' * If \eqn{\lambda = 0}, the posterior is equivalent to prior and the data do not influence the estimates.
#' * If \eqn{\lambda = \infty}, the posterior mean becomes OLS estimates (VAR).
#' 
#' \eqn{\sigma_i^2 / \sigma_j^2} in Minnesota moments explain the data scales.
#' 
#' @return `bvar_minnesota` returns an object `bvarmn` [class].
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
#'   \item{type}{include constant term (\code{const}) or not (\code{none})}
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
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
#' 
#' @order 1
#' @export
bvar_minnesota <- function(y, 
                           p, 
                           sigma, 
                           lambda = .1, 
                           delta, 
                           eps = 1e-04, 
                           type = c("const", "none")) {
  if (!all(apply(y, 2, is.numeric))) stop("Every column must be numeric class.")
  if (!is.matrix(y)) y <- as.matrix(y)
  if (missing(sigma)) sigma <- apply(y, 2, sd)
  m <- ncol(y)
  if (missing(delta)) delta <- rep(0, m)
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, p, p + 1)
  name_var <- colnames(y)
  colnames(Y0) <- name_var
  X0 <- build_design(y, p)
  name_lag <- concatenate_colnames(name_var, p:1) # in misc-r.R file
  colnames(X0) <- name_lag
  s <- nrow(Y0)
  k <- m * p + 1
  # dummy-----------------------------
  Yp <- build_ydummy(p, sigma, lambda, delta)
  colnames(Yp) <- name_var
  Xp <- build_xdummy(p, lambda, sigma, eps)
  colnames(Xp) <- name_lag
  # const or none---------------------
  type <- match.arg(type)
  if (type == "none") {
    X0 <- X0[, -k] # exclude 1 column
    Tp <- nrow(Yp)
    Yp <- Yp[-Tp,] # exclude intercept block from Yp (last row)
    Xp <- Xp[-Tp, -k] # exclude intercept block from Xp (last row and last column)
    name_lag <- name_lag[-k] # colnames(X0)
    k <- k - 1 # df = no intercept
  }
  # estimate-bvar.cpp-----------------
  posterior <- estimate_bvar_mn(X0, Y0, Xp, Yp)
  # Prior-----------------------------
  B0 <- posterior$prior_mean
  U0 <- posterior$prior_prec
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
    df = k, # k = m * p + 1
    obs = s, # s = n - p
    totobs = nrow(y), # n = total number of sample size
    process = "BVAR_Minnesota",
    type = type,
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
