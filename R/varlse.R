#' Fit VAR(p)
#' 
#' @description 
#' \code{var_lm} fits VAR(p) using OLS method
#' @param y matrix, Time series data of which columns indicate the variables
#' @param p integer, lags of VAR
#' @details 
#' For VAR(p) model
#' \deqn{Y_{t} = B_1 Y_{t - 1} + \cdots + B_p Y_{t - p} + \epsilon_t}
#' the function gives basic values.
#' 
#' @return \code{varlse} object
#' 
#' @export
var_lm <- function(y, p) {
  if (!is.matrix(y)) y <- as.matrix(y)
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
