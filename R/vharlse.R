#' Fit Vector HAR
#' 
#' @description 
#' This function fits VHAR using OLS method
#' @param y matrix, Time series data of which columns indicate the variables
#' @details 
#' For VHAR model
#' \deqn{Y_{t} = Phi^{(d)} Y_{t - 1} + \Phi^{(w)} Y_{t - 1}^{(w)} + \Phi^{(m)} Y_{t - 1}^{(m)} + \epsilon_t}
#' the function gives basic values.
#' 
#' @return \code{vhar_lm} returns an object named \code{vharlse} \link{class}.
#' 
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{design}{\eqn{X_0}}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{y}{Raw input}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = \code{totobs} - \code{p}}
#'   \item{totobs}{Total number of the observation}
#'   \item{process}{Process: VHAR}
#'   \item{call}{Matched call}
#'   \item{coefficients}{Coefficient Matrix}
#'   \item{fitted.values}{Fitted response values}
#'   \item{residuals}{Residuals}
#' }
#' 
#' @references 
#' Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
#' 
#' Corsi, F. (2008). \emph{A Simple Approximate Long-Memory Model of Realized Volatility}. Journal of Financial Econometrics, 7(2), 174–196. \url{https://doi:10.1093/jjfinec/nbp001}
#' 
#' @order 1
#' @export
vhar_lm <- function(y) {
  if (!is.matrix(y)) y <- as.matrix(y)
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, 22, 23)
  name_var <- colnames(y)
  colnames(Y0) <- name_var
  X0 <- build_design(y, 22)
  name_har <- concatenate_colnames(name_var, c("day", "week", "month")) # in misc-r.R file
  # Y0 = X1 Phi + Z------------------
  # X1 = X0 %*% t(HARtrans)
  # estimate Phi---------------------
  vhar_est <- estimate_har(X0, Y0)
  Phihat <- vhar_est$phihat
  colnames(Phihat) <- name_var
  rownames(Phihat) <- name_har
  # fitted values and residuals-----
  yhat <- vhar_est$fitted
  colnames(yhat) <- colnames(Y0)
  zhat <- Y0 - yhat
  # return as new S3 class-----------
  res <- list(
    design = X0,
    y0 = Y0,
    y = y,
    # p = p, # p
    m = ncol(y), # m
    obs = nrow(Y0), # s = n - p
    totobs = nrow(y), # n
    process = "VHAR",
    call = match.call(),
    HARtrans = vhar_est$HARtrans,
    coefficients = Phihat,
    fitted.values = yhat, # X0 %*% Bhat
    residuals = zhat # Y0 - X0 %*% Bhat
  )
  class(res) <- "vharlse"
  res
}

#' See if the Object \code{vharlse}
#' 
#' This function returns \code{TRUE} if the input is the output of \code{\link{vhar_lm}}.
#' 
#' @param x Object
#' 
#' @return \code{TRUE} or \code{FALSE}
#' 
#' @export
is.vharlse <- function(x) {
  inherits(x, "vharlse")
}

#' Coefficients Method for \code{vharlse} object
#' 
#' @param object varlse
#' @param ... not used
#' 
#' @export
coef.vharlse <- function(object, ...) {
  object$coefficients
}

#' Residuals Method for \code{vharlse} object
#' 
#' @param object varlse
#' @param ... not used
#' 
#' @export
residuals.vharlse <- function(object, ...) {
  object$residuals
}

#' Fitted Values Method for \code{vharlse} object
#' 
#' @param object varlse
#' @param ... not used
#' 
#' @export
fitted.vharlse <- function(object, ...) {
  object$fitted.values
}
