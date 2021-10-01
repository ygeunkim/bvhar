#' Summarizing Vector Autoregressive Model
#' 
#' `summary` method for `varlse` class.
#' 
#' @param object \code{varlse} object
#' @param ... not used
#' 
#' @return \code{summary.varlse} \link{class} additionaly computes the following
#' \item{\code{names}}{Variable names}
#' \item{\code{totobs}}{Total number of the observation}
#' \item{\code{obs}}{Sample size used when training = \code{totobs} - \code{p}}
#' \item{\code{coefficients}}{Coefficient Matrix}
#' \item{\code{call}}{Matched call}
#' \item{\code{process}}{Process: VAR}
#' \item{\code{covmat}}{Covariance matrix of the residuals}
#' \item{\code{corrmat}}{Correlation matrix of the residuals}
#' \item{\code{roots}}{Roots of characteristic polynomials}
#' \item{\code{is_stable}}{Whether the process is stable or not based on \code{roots}}
#' \item{\code{ic}}{Information criteria vector}
#' \itemize{
#'     \item{\code{AIC}} - AIC
#'     \item{\code{BIC}} - BIC
#'     \item{\code{HQ}} - HQ
#'     \item{\code{FPE}} - FPE
#' }
#' 
#' @references 
#' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' @importFrom stats cor
#' @order 1
#' @export
summary.varlse <- function(object, ...) {
  var_name <- colnames(object$y0)
  cov_resid <- object$covmat
  # split the matrix for the print: B1, ..., Bp
  bhat_mat <- split_coef(object)
  if (object$type == "const") bhat_mat$intercept <- object$coefficients[object$df,]
  log_lik <- logLik(object)
  res <- list(
    names = var_name,
    totobs = object$totobs,
    obs = object$obs,
    p = object$p,
    coefficients = bhat_mat,
    call = object$call,
    process = object$process,
    type = object$type,
    covmat = cov_resid,
    corrmat = cor(object$residuals),
    roots = stableroot(object),
    is_stable = is.stable(object),
    log_lik = log_lik,
    ic = c(
      AIC = AIC(log_lik),
      BIC = BIC(log_lik),
      HQ = HQ(log_lik)
    )
  )
  class(res) <- "summary.varlse"
  res
}
