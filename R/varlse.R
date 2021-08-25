#' Fit Vector Autoregressive Model of Order p Model
#' 
#' @description 
#' This function fits VAR(p) using OLS method
#' @param y Time series data of which columns indicate the variables
#' @param p integer, lags of VAR
#' @details 
#' For VAR(p) model
#' \deqn{Y_{t} = c + B_1 Y_{t - 1} + \cdots + B_p Y_{t - p} + \epsilon_t}
#' the function gives basic values.
#' 
#' @return \code{var_lm} returns an object named \code{varlse} \link{class}.
#' 
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{design}{\eqn{X_0}}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{y}{Raw input}
#'   \item{p}{Lag of VAR}
#'   \item{m}{Dimension of the data}
#'   \item{df}{Numer of Coefficients: mp + 1}
#'   \item{obs}{Sample size used when training = \code{totobs} - \code{p}}
#'   \item{totobs}{Total number of the observation}
#'   \item{process}{Process: VAR}
#'   \item{call}{Matched call}
#'   \item{coefficients}{Coefficient Matrix}
#'   \item{fitted.values}{Fitted response values}
#'   \item{residuals}{Residuals}
#' }
#' 
#' @author Young Geun Kim \email{dudrms33@@g.skku.edu}
#' 
#' @references 
#' Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
#' 
#' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
#' 
#' @seealso 
#' \code{\link{build_y0}} and \code{\link{build_design}} for defining Y0 and X0 matrix,
#' 
#' and \code{\link{estimate_var}} for computing coefficient VAR matrix.
#' 
#' Other package \code{\link[vars]{VAR}} is famous in VAR modeling.
#' @examples 
#' # Perform the function using etf_vix dataset
#' \dontrun{
#'   fit <- var_lm(y = etf_vix, p = 5)
#'   class(fit)
#'   str(fit)
#' }
#' 
#' # Extract coef, fitted values, and residuals
#' \dontrun{
#'   coef(fit)
#'   residuals(fit)
#'   fitted(fit)
#' }
#' 
#' @order 1
#' @export
var_lm <- function(y, p) {
  if (!is.matrix(y)) y <- as.matrix(y)
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, p, p + 1)
  name_var <- colnames(y)
  colnames(Y0) <- name_var
  X0 <- build_design(y, p)
  name_lag <- concatenate_colnames(name_var, p:1) # in misc-r.R file
  colnames(X0) <- name_lag
  # estimate B-----------------------
  var_est <- estimate_var(X0, Y0)
  Bhat <- var_est$bhat
  colnames(Bhat) <- colnames(Y0)
  rownames(Bhat) <- colnames(X0)
  # fitted values and residuals-----
  yhat <- var_est$fitted
  colnames(yhat) <- colnames(Y0)
  zhat <- Y0 - yhat
  # residual Covariance matrix------
  m <- ncol(y)
  k <- m * p + 1
  covmat <- compute_cov(zhat, nrow(Y0), k) # Sighat = z^T %*% z / (s - k)
  colnames(covmat) <- name_var
  rownames(covmat) <- name_var
  # return as new S3 class-----------
  res <- list(
    design = X0,
    y0 = Y0,
    y = y,
    p = p, # p
    m = m, # m
    df = k, # k = m * p + 1
    obs = nrow(Y0), # s = n - p
    totobs = nrow(y), # n
    process = "VAR",
    call = match.call(),
    coefficients = Bhat,
    fitted.values = yhat, # X0 %*% Bhat
    residuals = zhat, # Y0 - X0 %*% Bhat
    covmat = covmat
  )
  class(res) <- "varlse"
  res
}

#' See if the Object \code{varlse}
#' 
#' This function returns \code{TRUE} if the input is the output of \code{\link{var_lm}}.
#' 
#' @param x Object
#' 
#' @return \code{TRUE} or \code{FALSE}
#' 
#' @export
is.varlse <- function(x) {
  inherits(x, "varlse")
}

#' Coefficients Method for \code{varlse} object
#' 
#' @param object \code{varlse} object
#' @param ... not used
#' 
#' @export
coef.varlse <- function(object, ...) {
  object$coefficients
}

#' Residuals Method for \code{varlse} object
#' 
#' @param object \code{varlse} object
#' @param ... not used
#' 
#' @export
residuals.varlse <- function(object, ...) {
  object$residuals
}

#' Fitted Values Method for \code{varlse} object
#' 
#' @param object \code{varlse} object
#' @param ... not used
#' 
#' @export
fitted.varlse <- function(object, ...) {
  object$fitted.values
}

#' Choose the Best VAR based on Information Criteria
#' 
#' This function computes AIC, FPE, BIC, and HQ up to p = \code{lag_max} of VAR model.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param lag_max Maximum Var lag to explore (default = 5)
#' @param parallel Parallel computation using \code{\link[foreach]{foreach}}? By default, \code{FALSE}.
#' 
#' 
#' @return Minimum order and information criteria values
#' 
#' @importFrom foreach foreach %do% %dopar%
#' @export
choose_var <- function(y, lag_max = 5, parallel = FALSE) {
  if (!is.matrix(y)) y <- as.matrix(y)
  var_list <- NULL
  # compute IC-----------------------
  if (parallel) {
    res <- foreach(p = 1:lag_max, .combine = rbind) %dopar% {
      var_list <- var_lm(y, p)
      c(
        "AIC" = AIC(var_list),
        "BIC" = BIC(var_list),
        "HQ" = HQ(var_list),
        "FPE" = FPE(var_list)
      )
    }
  } else {
    res <- foreach(p = 1:lag_max, .combine = rbind) %do% {
      var_list <- var_lm(y, p)
      c(
        "AIC" = AIC(var_list),
        "BIC" = BIC(var_list),
        "HQ" = HQ(var_list),
        "FPE" = FPE(var_list)
      )
    }
  }
  rownames(res) <- 1:lag_max
  # find minimum-----------------------
  list(
    ic = res,
    min_lag = apply(res, 2, which.min)
  )
}

