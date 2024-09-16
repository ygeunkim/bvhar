#' Fitting Vector Autoregressive Model of Order p Model
#' 
#' This function fits VAR(p) using OLS method.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p Lag of VAR (Default: 1)
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param method Method to solve linear equation system.
#' (`nor`: normal equation (default), `chol`: Cholesky, and `qr`: HouseholderQR)
#' @details 
#' This package specifies VAR(p) model as
#' 
#' \deqn{Y_{t} = A_1 Y_{t - 1} + \cdots + A_p Y_{t - p} + c + \epsilon_t}
#' 
#' If `include_type = TRUE`, there is constant term.
#' The function estimates every coefficient matrix.
#' @return `var_lm()` returns an object named `varlse` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Coefficient Matrix}
#'   \item{fitted.values}{Fitted response values}
#'   \item{residuals}{Residuals}
#'   \item{covmat}{LS estimate for covariance matrix}
#'   \item{df}{Numer of Coefficients}
#'   \item{p}{Lag of VAR}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Process: VAR}
#'   \item{type}{include constant term (`const`) or not (`none`)}
#'   \item{design}{Design matrix}
#'   \item{y}{Raw input}
#'   \item{y0}{Multivariate response matrix}
#'   \item{method}{Solving method}
#'   \item{call}{Matched call}
#' }
#' It is also a `bvharmod` class.
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
#' @seealso 
#' * [summary.varlse()] to summarize VAR model
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
var_lm <- function(y, p = 1, include_mean = TRUE, method = c("nor", "chol", "qr")) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  method <- match.arg(method)
  method_fit <- switch(method, "nor" = 1, "chol" = 2, "qr" = 3)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(ncol(y)))
  }
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  name_lag <- concatenate_colnames(name_var, 1:p, include_mean)
  res <- estimate_var(y, p, include_mean, method_fit)
  colnames(res$y) <- name_var
  colnames(res$y0) <- name_var
  colnames(res$design) <- name_lag
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_lag
  colnames(res$covmat) <- name_var
  rownames(res$covmat) <- name_var
  # return as new S3 class-----------
  res$method <- method
  res$call <- match.call()
  class(res) <- c("varlse", "olsmod", "bvharmod")
  res
}