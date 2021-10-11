#' Fit Vector Autoregressive Model of Order p Model
#' 
#' This function fits VAR(p) using OLS method.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p integer, lags of VAR
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @details 
#' This package specifies VAR(p) model as
#' \deqn{Y_{t} = c + A_1 Y_{t - 1} + \cdots + A_p Y_{t - p} + \epsilon_t}
#' 
#' If `include_type = TRUE`, there is \eqn{c} term.
#' Otherwise (`include_type = FALSE`), there is no \eqn{c} term.
#' The function estimates every coefficient matrix \eqn{c, B_1, \ldots, B_p}.
#' 
#' * [build_y0()] gives response matrix, \eqn{Y_0}.
#' * [build_design()] gives design matrix, \eqn{X_0}.
#' * Coefficient matrix is the form of \eqn{A = [A_1, A_2, \ldots, A_p, c]^T}.
#' 
#' Then perform least squares to the following multivariate regression model
#' 
#' \deqn{Y_0 = X_0 A + error}
#' 
#' which gives
#' 
#' \deqn{\hat{B} = (X_0^T X_0)^{-1} X_0^T Y_0}
#' 
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
#'   \item{obs}{Sample size used when training = \code{totobs} - \code{p}}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Process: VAR}
#'   \item{type}{include constant term (\code{const}) or not (\code{none})}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' 
#' @references 
#' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' @seealso 
#' * [build_y0()] and [build_design()] to define Y0 and X0 matrix.
#' * [estimate_var()] to compute coefficient VAR matrix.
#' * Other package [vars::VAR()] is famous in VAR modeling.
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
var_lm <- function(y, p, include_mean = TRUE) {
  if (!all(apply(y, 2, is.numeric))) stop("Every column must be numeric class.")
  if (!is.matrix(y)) y <- as.matrix(y)
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, p, p + 1)
  name_var <- colnames(y)
  colnames(Y0) <- name_var
  X0 <- build_design(y, p)
  name_lag <- concatenate_colnames(name_var, p:1) # in misc-r.R file
  colnames(X0) <- name_lag
  # const or none--------------------
  if (!is.logical(include_mean)) stop("'include_mean' is logical.")
  m <- ncol(y)
  k <- m * p + 1 # df
  if (!include_mean) {
    X0 <- X0[, -k] # exclude 1 column
    k <- k - 1 # df = no intercept
  }
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
  covmat <- compute_cov(zhat, nrow(Y0), k) # Sighat = z^T %*% z / (s - k)
  colnames(covmat) <- name_var
  rownames(covmat) <- name_var
  # return as new S3 class-----------
  res <- list(
    # estimation-----------
    coefficients = Bhat,
    fitted.values = yhat, # X0 %*% Bhat
    residuals = zhat, # Y0 - X0 %*% Bhat
    covmat = covmat,
    # variables------------
    df = k, # k = m * p + 1 or m * p
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

#' Choose the Best VAR based on Information Criteria
#' 
#' This function computes AIC, FPE, BIC, and HQ up to p = `lag_max` of VAR model.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param lag_max Maximum Var lag to explore (default = 5)
#' @param include_mean `r lifecycle::badge("experimental")` Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param parallel Parallel computation using [foreach::foreach()]? By default, `FALSE`.
#' 
#' @return Minimum order and information criteria values
#' 
#' @importFrom foreach foreach %do% %dopar%
#' @export
choose_var <- function(y, lag_max = 5, include_mean = TRUE, parallel = FALSE) {
  if (!all(apply(y, 2, is.numeric))) stop("Every column must be numeric class.")
  if (!is.matrix(y)) y <- as.matrix(y)
  var_list <- NULL
  if (!is.logical(include_mean)) stop("'include_mean' is logical.")
  # compute IC-----------------------
  if (parallel) {
    res <- foreach(p = 1:lag_max, .combine = rbind) %dopar% {
      var_list <- var_lm(y, p, include_mean = include_mean)
      c(
        "AIC" = AIC(var_list, type = "rss"),
        "BIC" = BIC(var_list, type = "rss"),
        "HQ" = HQ(var_list, type = "rss"),
        "FPE" = FPE(var_list)
      )
    }
  } else {
    res <- foreach(p = 1:lag_max, .combine = rbind) %do% {
      var_list <- var_lm(y, p, include_mean = include_mean)
      c(
        "AIC" = AIC(var_list, type = "rss"),
        "BIC" = BIC(var_list, type = "rss"),
        "HQ" = HQ(var_list, type = "rss"),
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

