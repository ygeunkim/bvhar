#' Fit Vector HAR Model
#' 
#' This function fits VHAR using OLS method.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param include_mean `r lifecycle::badge("experimental")` Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @details 
#' For VHAR model
#' \deqn{Y_{t} = \Phi^{(d)} Y_{t - 1} + \Phi^{(w)} Y_{t - 1}^{(w)} + \Phi^{(m)} Y_{t - 1}^{(m)} + \epsilon_t}
#' the function gives basic values.
#' 
#' @return `vhar_lm` returns an object named `vharlse` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Coefficient Matrix}
#'   \item{fitted.values}{Fitted response values}
#'   \item{residuals}{Residuals}
#'   \item{covmat}{LS estimate for covariance matrix}
#'   \item{df}{Numer of Coefficients: 3m + 1 or 3m}
#'   \item{p}{3 (The number of terms. \code{vharlse} contains this element for usage in other functions.)}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = \code{totobs} - \code{p}}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Process: VHAR}
#'   \item{type}{include constant term (\code{const}) or not (\code{none})}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' 
#' @references 
#' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174–196. [https://doi:10.1093/jjfinec/nbp001](https://doi:10.1093/jjfinec/nbp001)
#' 
#' Baek, C. and Park, M. (2021). *Sparse vector heterogeneous autoregressive modeling for realized volatility*. J. Korean Stat. Soc. 50, 495–510. [https://doi.org/10.1007/s42952-020-00090-5](https://doi.org/10.1007/s42952-020-00090-5)
#' 
#' @seealso 
#' [scale_har()] to linear transformation for VHAR,
#' 
#' [estimate_har()] to compute coefficient VHAR matrix,
#' 
#' and [estimate_har_none()] to compute coefficient VHAR matrix without constant term.
#' @examples 
#' # Perform the function using etf_vix dataset
#' \dontrun{
#'   fit <- vhar_lm(y = etf_vix)
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
vhar_lm <- function(y, include_mean = TRUE) {
  if (!all(apply(y, 2, is.numeric))) stop("Every column must be numeric class.")
  if (!is.matrix(y)) y <- as.matrix(y)
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, 22, 23)
  name_var <- colnames(y)
  colnames(Y0) <- name_var
  X0 <- build_design(y, 22)
  name_har <- concatenate_colnames(name_var, c("day", "week", "month")) # in misc-r.R file
  # const or none--------------------
  if (!is.logical(include_mean)) stop("'include_mean' is logical.")
  m <- ncol(y)
  num_coef <- 3 * m + 1
  if (!include_mean) {
    X0 <- X0[, -(22 * m + 1)] # exclude 1 column
    name_har <- name_har[-num_coef] # remove const (row)name
    num_coef <- num_coef - 1 # df = 3 * m
  }
  # Y0 = X1 Phi + Z------------------
  # X1 = X0 %*% t(HARtrans)
  # estimate Phi---------------------
  type <- ifelse(include_mean, "const", "none")
  vhar_est <- switch(
    type,
    "const" = {
      estimate_har(X0, Y0)
    },
    "none" = {
      estimate_har_none(X0, Y0)
    }
  )
  Phihat <- vhar_est$phihat
  colnames(Phihat) <- name_var
  rownames(Phihat) <- name_har
  # fitted values and residuals-----
  yhat <- vhar_est$fitted
  colnames(yhat) <- colnames(Y0)
  zhat <- Y0 - yhat
  # residual Covariance matrix------
  covmat <- compute_cov(zhat, nrow(Y0), num_coef) # Sighat = z^T %*% z / (s - (3m + 1))
  colnames(covmat) <- name_var
  rownames(covmat) <- name_var
  # return as new S3 class-----------
  res <- list(
    # estimation---------------
    coefficients = Phihat,
    fitted.values = yhat, # X1 %*% Phihat
    residuals = zhat, # Y0 - X1 %*% Phihat
    covmat = covmat,
    # variables---------------
    df = num_coef, # nrow(Phihat) = 3 * m + 1 or 3 * m
    p = 3, # add for other function (df = 3m + 1 = mp + 1)
    m = ncol(y), # m
    obs = nrow(Y0), # s = n - 22
    totobs = nrow(y), # n
    # about model------------
    call = match.call(),
    process = "VHAR",
    type = type,
    # data-------------------
    HARtrans = vhar_est$HARtrans,
    y0 = Y0,
    design = X0,
    y = y
  )
  class(res) <- c("vharlse", "bvharmod")
  res
}
