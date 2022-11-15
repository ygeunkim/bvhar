#' Fitting Vector Autoregressive Model of Order p Model
#' 
#' This function fits VAR(p) using OLS method.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p integer, lags of VAR
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @details 
#' This package specifies VAR(p) model as
#' \deqn{Y_{t} = A_1 Y_{t - 1} + \cdots + A_p Y_{t - p} + c + \epsilon_t}
#' 
#' If `include_type = TRUE`, there is \eqn{c} term.
#' Otherwise (`include_type = FALSE`), there is no \eqn{c} term.
#' The function estimates every coefficient matrix \eqn{A_1, \ldots, A_p, c}.
#' 
#' * Response matrix, \eqn{Y_0} in [var_design_formulation]
#' * Design matrix, \eqn{X_0} in [var_design_formulation]
#' * Coefficient matrix is the form of \eqn{A = [A_1, A_2, \ldots, A_p, c]^T}.
#' 
#' Then perform least squares to the following multivariate regression model
#' \deqn{Y_0 = X_0 A + error}
#' 
#' which gives
#' 
#' \deqn{\hat{A} = (X_0^T X_0)^{-1} X_0^T Y_0}
#' @return `var_lm` returns an object named `varlse` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Coefficient Matrix}
#'   \item{fitted.values}{Fitted response values}
#'   \item{residuals}{Residuals}
#'   \item{covmat}{LS estimate for covariance matrix}
#'   \item{df}{Numer of Coefficients: mp + 1 or mp}
#'   \item{p}{Lag of VAR}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Process: VAR}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. doi:[10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' @seealso 
#' * Other package [vars::VAR()] is famous in VAR modeling.
#' * [coef.varlse()], [residuals.varlse()], and [fitted.varlse()]
#' * [summary.varlse()] to summarize VAR model
#' * [predict.varlse()] to forecast the VAR process
#' * [var_design_formulation] for the model design
#' @examples 
#' # Perform the function using etf_vix dataset
#' fit <- var_lm(y = etf_vix, p = 2)
#' class(fit)
#' str(fit)
#' 
#' # Extract coef, fitted values, and residuals
#' coef(fit)
#' head(residuals(fit))
#' head(fitted(fit))
#' @order 1
#' @export
var_lm <- function(y, p, include_mean = TRUE) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, p, p + 1)
  m <- ncol(y)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(m))
  }
  colnames(Y0) <- name_var
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  X0 <- build_design(y, p, include_mean)
  name_lag <- concatenate_colnames(name_var, 1:p, include_mean) # in misc-r.R file
  colnames(X0) <- name_lag
  # estimate B-----------------------
  var_est <- estimate_var(X0, Y0)
  coef_est <- var_est$coef # Ahat
  colnames(coef_est) <- colnames(Y0)
  rownames(coef_est) <- colnames(X0)
  # fitted values and residuals-----
  yhat <- var_est$fitted
  colnames(yhat) <- colnames(Y0)
  zhat <- Y0 - yhat
  # residual Covariance matrix------
  covmat <- compute_cov(zhat, nrow(Y0), ncol(X0)) # Sighat = z^T %*% z / (s - k)
  colnames(covmat) <- name_var
  rownames(covmat) <- name_var
  # return as new S3 class-----------
  res <- list(
    # estimation-----------
    coefficients = coef_est,
    fitted.values = yhat, # X0 %*% Ahat
    residuals = zhat, # Y0 - X0 %*% Ahat
    covmat = covmat,
    # variables------------
    df = ncol(X0), # k = m * p + 1 or m * p
    p = p, # p
    m = m, # m
    obs = nrow(Y0), # s = n - p
    totobs = nrow(y), # n
    # about model---------
    call = match.call(),
    process = "VAR",
    type = ifelse(include_mean, "const", "none"),
    # data----------------
    y0 = Y0,
    design = X0,
    y = y
  )
  class(res) <- c("varlse", "bvharmod")
  res
}
