#' Fit Bayesian VAR(p) of Minnesota Prior
#' 
#' @description
#' This function fits BVAR(p) with Ghosh et al. (2018) nonhierarchical prior.
#' 
#' @param y matrix, Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param U Positive definite matrix. By default, identity matrix of dimension ncol(X0)
#' 
#' @details 
#' In Ghosh et al. (2018), there are many models for BVAR such as hierarchical or non-hierarchical.
#' Among these, this function chooses the most simple non-hierarchical matrix normal prior.
#' 
#' \deqn{B \mid \Sigma_e \sim MN(B_0, \Omega_0, \Sigma_e)}
#' \eqn{\Sigma_e \sim} flat
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
#' \item{\code{mn_scale}}{First scale matrix of posterior matrix normal distribution}
#' \item{\code{iw_mean}}{Scale matrix of posterior inverse-wishart distribution}
#' 
#' @references 
#' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
#' 
#' Ghosh, S., Khare, K., & Michailidis, G. (2018). \emph{High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models}. Journal of the American Statistical Association, 114(526). \url{https://doi:10.1080/01621459.2018.1437043}
#' 
#' @order 1
#' @export
bvar_ghosh <- function(y, p, U) {
  if (!is.matrix(y)) y <- as.matrix(y)
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, p, p + 1)
  name_var <- colnames(y)
  colnames(Y0) <- name_var
  X0 <- build_design(y, p)
  if (missing(U)) U <- diag(ncol(X0)) # identity matrix
  name_lag <- concatenate_colnames(name_var, p:1) # in misc-r.R file
  colnames(X0) <- name_lag
  # Matrix normal---------------------
  posterior <- estimate_bvar_ghosh(X0, Y0, U);
  Bhat <- posterior$bhat # posterior mean
  colnames(Bhat) <- name_var
  rownames(Bhat) <- name_lag
  Uhat <- posterior$mnscale
  colnames(Uhat) <- name_lag
  rownames(Uhat) <- name_lag
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
    process = "Ghosh",
    call = match.call(),
    mn_mean = Bhat,
    # fitted.values = yhat,
    # residuals = Y0 - yhat,
    mn_scale = Uhat,
    iw_scale = Sighat
  )
  class(res) <- "bvarghosh"
  res
}
