#' Fit VAR(p)
#' 
#' @description 
#' The function fits VAR(p) using OLS method
#' @param y matrix, Time series data of which columns indicate the variables
#' @param p integer, lags of VAR
#' @details 
#' For VAR(p) model
#' \deqn{Y_{t} = c + B_1 Y_{t - 1} + \cdots + B_p Y_{t - p} + \epsilon_t}
#' the function gives basic values.
#' 
#' @return \code{varlse} \link{class} with
#' \item{\code{design}}{\eqn{X_0}}
#' \item{\code{y0}}{\eqn{Y_0}}
#' \item{\code{y}}{raw input}
#' \item{\code{p}}{lag of VAR: p}
#' \item{\code{m}}{Dimension of the data}
#' \item{\code{obs}}{Sample size used when training = \code{totobs} - \code{p}}
#' \item{\code{totobs}}{Total number of the observation}
#' \item{\code{process}}{Process: VAR}
#' \item{\code{call}}{Matched call}
#' \item{\code{coefficients}}{Coefficient Matrix}
#' \item{\code{fitted.values}}{Fitted response values}
#' \item{\code{residuals}}{Residuals}
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
#' and \code{\link{estimate_var}} for computing coefficient VAR matrix.
#' Other package \code{\link[vars]{VAR}} is famous in VAR modeling.
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
  # return as new S3 class-----------
  res <- list(
    design = X0,
    y0 = Y0,
    y = y,
    p = p, # p
    m = ncol(y), # m
    obs = nrow(Y0), # s = n - p
    totobs = nrow(y), # n
    process = "VAR",
    call = match.call(),
    coefficients = Bhat,
    fitted.values = yhat, # X0 %*% Bhat
    residuals = zhat # Y0 - X0 %*% Bhat
  )
  class(res) <- "varlse"
  res
}

#' Coefficients Method for \code{varlse} object
#' 
#' @param object \code{varlse} object
#' @param ... not used
#' 
#' @export
coefficients.varlse <- function(object, ...) {
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
