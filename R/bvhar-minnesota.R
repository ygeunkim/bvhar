#' Fit Bayesian VHAR of Minnesota Prior
#' 
#' @description
#' This function fits BVHAR with Minnesota prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param mn_type Prior mean type to apply. ("VAR" = Original Minnesota. "VHAR" = fills zero in the first block)
#' @param sigma Standard error vector for each variable
#' @param lambda Tightness of the prior around a random walk or white noise
#' @param delta Prior belief about white noise (Litterman sets 1: default)
#' @param daily Same as delta in VHAR type
#' @param weekly Fill the second part in the first block
#' @param monthly Fill the third part in the first block
#' @param eps Very small number
#' @param include_mean `r lifecycle::badge("experimental")` Add constant term (Default: `TRUE`) or not (`FALSE`)
#' 
#' @details 
#' Apply Minnesota prior to Vector HAR: \eqn{\Phi} (VHAR matrices) and \eqn{\Sigma_e} (residual covariance).
#' 
#' \deqn{\Phi \mid \Sigma_e \sim MN(P_0, \Psi_0, \Sigma_e)}
#' \deqn{\Sigma_e \sim IW(U_0, d_0)}
#' (MN: \href{https://en.wikipedia.org/wiki/Matrix_normal_distribution}{matrix normal}, IW: \href{https://en.wikipedia.org/wiki/Inverse-Wishart_distribution}{inverse-wishart})
#' 
#' Two types of Minnesota priors builds different dummy variables for Y0.
#' `mn_type = "VAR"` constructs dummy Y0 with `p = 3` of [build_ydummy()].
#' The only difference from BVAR is dimension.
#' 
#' On the other hand, dummy Y0 for `mn_type = "VHAR"` has its own function [build_ydummy_bvhar()].
#' It fills the zero matrix in the first block in Bańbura et al. (2010).
#' 
#' @return `bvhar_minnesota` returns an object `bvharmn` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{design}{\eqn{X_0}}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{y}{Raw input}
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
#' @seealso 
#' \code{\link{build_ydummy}}, \code{\link{build_xdummy}}, and \code{\link{build_ydummy_bvhar}} for defining dummy observation.
#' 
#' \code{\link{estimate_bvar_mn}} to compute Minnesota posterior estimates.
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
bvhar_minnesota <- function(y, 
                            mn_type = c("VAR", "VHAR"), 
                            sigma, 
                            lambda, 
                            delta, 
                            daily, 
                            weekly, 
                            monthly, 
                            eps = 1e-04, 
                            include_mean = TRUE) {
  if (!all(apply(y, 2, is.numeric))) stop("Every column must be numeric class.")
  if (!is.matrix(y)) y <- as.matrix(y)
  mn_type <- match.arg(mn_type)
  m <- ncol(y)
  N <- nrow(y)
  num_coef <- 3 * m + 1
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, 22, 23)
  name_var <- colnames(y)
  colnames(Y0) <- name_var
  X0 <- build_design(y, 22)
  HARtrans <- scale_har(m)
  name_har <- concatenate_colnames(name_var, c("day", "week", "month")) # in misc-r.R file
  # dummy-----------------------------
  Yh <- switch(
    mn_type,
    "VAR" = {
      if (missing(delta)) delta <- rep(1, m)
      Yh <- build_ydummy(3, sigma, lambda, delta)
      colnames(Yh) <- name_var
      Yh
    },
    "VHAR" = {
      if (missing(daily)) daily <- rep(1, m)
      if (missing(weekly)) weekly <- rep(1, m)
      if (missing(monthly)) monthly <- rep(1, m)
      Yh <- build_ydummy_bvhar(sigma, lambda, daily, weekly, monthly)
      colnames(Yh) <- name_var
      Yh
    }
  )
  Xh <- build_xdummy(3, lambda, sigma, eps)
  colnames(Xh) <- name_har
  # const or none---------------------
  if (!is.logical(include_mean)) stop("'include_mean' is logical.")
  if (!include_mean) {
    X0 <- X0[, -(22 * m + 1)] # exclude 1 column
    HARtrans <- HARtrans[-num_coef, -(22 * m + 1)] # HARtrans: 3m x 22m matrix
    Th <- nrow(Yh)
    Yh <- Yh[-Th,] # exclude intercept block from Yh (last row)
    Xh <- Xh[-Th, -num_coef] # exclude intercept block from Xh (last row and last column)
    name_har <- name_har[-num_coef] # remove const (row)name
    num_coef <- num_coef - 1 # df = 3 * m
  }
  X1 <- X0 %*% t(HARtrans)
  colnames(X1) <- name_har
  # estimate-bvar.cpp-----------------
  posterior <- estimate_bvar_mn(X1, Y0, Xh, Yh)
  # Prior-----------------------------
  P0 <- posterior$prior_mean
  Psi0 <- posterior$prior_precision
  U0 <- posterior$prior_scale
  d0 <- posterior$prior_shape
  # Matrix normal---------------------
  Phihat <- posterior$bhat # posterior mean
  colnames(Phihat) <- name_var
  rownames(Phihat) <- name_har
  Psihat <- posterior$mnprec
  colnames(Psihat) <- name_har
  rownames(Psihat) <- name_har
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
    m = m, # m
    obs = nrow(Y0), # s = n - p
    totobs = N, # n
    process = ifelse(mn_type == "VAR", "BVHAR_mn_var", "BVHAR_mn_vhar"),
    type = ifelse(include_mean, "const", "none"),
    call = match.call(),
    # HAR------------------
    HARtrans = HARtrans,
    # prior----------------
    prior_mean = P0,
    prior_precision = Psi0,
    prior_scale = U0,
    prior_shape = d0 + (m + 3), # add (m + 3) for prior mean existence
    # posterior-----------
    mn_mean = Phihat,
    fitted.values = yhat,
    residuals = Y0 - yhat,
    mn_prec = Psihat,
    iw_scale = Sighat,
    iw_shape = d0 + N + 2
  )
  class(res) <- "bvharmn"
  res
}
