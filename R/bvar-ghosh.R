#' Fit Bayesian VAR(p) of Nonhiearchical Matrix Normal Prior
#' 
#' @description
#' This function fits BVAR(p) with Ghosh et al. (2018) nonhierarchical prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param U Positive definite matrix. By default, identity matrix of dimension ncol(X0)
#' 
#' @details 
#' Ghosh et al. (2018) gives flat prior for residual matrix in BVAR.
#' 
#' Under this setting, there are many models such as hierarchical or non-hierarchical.
#' This function chooses the most simple non-hierarchical matrix normal prior in Section 3.1.
#' 
#' \deqn{B \mid \Sigma_e \sim MN(0, U^{-1}, \Sigma_e)}
#' \eqn{\Sigma_e \sim} flat
#' where U: precision matrix.
#' 
#' Then in VAR design equation (Y0 = X0 B + Z),
#' MN mean can be derived by
#' \deqn{\hat{B} = (X_0^T X_0 + U)^{-1} X_0^T Y_0}
#' and the MN scale matrix can be derived by
#' \deqn{\hat\Sigma_e = Y_0^T (I_s - X_0(X_0^T X_0 + U)^{-1} X_0^T) Y_0}
#' and IW shape by \eqn{s - m - 1}.
#' 
#' (MN: \href{https://en.wikipedia.org/wiki/Matrix_normal_distribution}{matrix normal}, IW: \href{https://en.wikipedia.org/wiki/Inverse-Wishart_distribution}{inverse-wishart}).
#' 
#' @return \code{bvar_flat} returns an object \code{bvarghosh} \link{class}.
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
#'   \item{process}{Process: Ghosh}
#'   \item{call}{Matched call}
#'   \item{mn_mean}{Location of posterior matrix normal distribution}
#'   \item{fitted.values}{Fitted values}
#'   \item{residuals}{Residuals}
#'   \item{mn_prec}{Precision matrix of posterior matrix normal distribution}
#'   \item{iw_scale}{Scale matrix of posterior inverse-wishart distribution}
#'   \item{iw_shape}{Shape of posterior inverse-wishart distribution}
#' }
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
bvar_flat <- function(y, p, U) {
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
  posterior <- estimate_mn_flat(X0, Y0, U)
  Bhat <- posterior$bhat # posterior mean
  colnames(Bhat) <- name_var
  rownames(Bhat) <- name_lag
  Uhat <- posterior$mnprec
  colnames(Uhat) <- name_lag
  rownames(Uhat) <- name_lag
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
    p = p, # p
    m = ncol(y), # m
    obs = nrow(Y0), # s = n - p
    totobs = nrow(y), # n
    process = "Ghosh",
    call = match.call(),
    mn_mean = Bhat,
    fitted.values = yhat,
    residuals = Y0 - yhat,
    mn_prec = Uhat,
    iw_scale = Sighat,
    iw_shape = posterior$iwshape
  )
  class(res) <- "bvarghosh"
  res
}

#' See if the Object \code{bvarghosh}
#' 
#' This function returns \code{TRUE} if the input is the output of \code{\link{bvar_ghosh}}.
#' 
#' @param x Object
#' 
#' @return \code{TRUE} or \code{FALSE}
#' 
#' @export
is.bvarghosh <- function(x) {
  inherits(x, "bvarghosh")
}

#' Coefficients Method for \code{bvarghosh} object
#' 
#' Matrix Normal mean of Minnesota BVAR
#' 
#' @param object \code{bvarghosh} object
#' @param ... not used
#' 
#' @export
coef.bvarghosh <- function(object, ...) {
  object$mn_mean
}

#' Residuals Method for \code{bvarghosh} object
#' 
#' @param object \code{bvarghosh} object
#' @param ... not used
#' 
#' @export
residuals.bvarghosh <- function(object, ...) {
  object$residuals
}

#' Fitted Values Method for \code{bvarghosh} object
#' 
#' @param object \code{bvarghosh} object
#' @param ... not used
#' 
#' @export
fitted.bvarghosh <- function(object, ...) {
  object$fitted.values
}
