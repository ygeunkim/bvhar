#' Summarizing Vector HAR Model
#' 
#' `summary` method for `vharlse` class.
#' 
#' @param object `vharlse` object
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
#' Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174–196. [https://doi:10.1093/jjfinec/nbp001](https://doi:10.1093/jjfinec/nbp001)
#' 
#' Baek, C. and Park, M. (2021). *Sparse vector heterogeneous autoregressive modeling for realized volatility*. J. Korean Stat. Soc. 50, 495–510. [https://doi.org/10.1007/s42952-020-00090-5](https://doi.org/10.1007/s42952-020-00090-5)
#' 
#' @importFrom stats cor
#' @order 1
#' @export
summary.vharlse <- function(object, ...) {
  vhar_name <- colnames(object$y0)
  cov_resid <- object$covmat
  # split the matrix for the print: Phi(d), Phi(w), Phi(m)
  phihat_mat <- switch(
    object$type,
    "const" = {
      split.data.frame(object$coefficients[-(3 * object$m + 1),], gl(3, object$m)) %>% 
        lapply(t)
    },
    "none" = {
      split.data.frame(object$coefficients, gl(3, object$m)) %>% 
        lapply(t)
    }
  )
  names(phihat_mat) <- c("day", "week", "month")
  if (object$type == "const") phihat_mat$intercept <- object$coefficients[3 * object$m + 1,]
  log_lik <- logLik(object)
  res <- list(
    names = vhar_name,
    totobs = object$totobs,
    obs = object$obs,
    p = object$p,
    coefficients = phihat_mat,
    call = object$call,
    process = object$process,
    type = object$type,
    covmat = cov_resid,
    corrmat = cor(object$residuals),
    log_lik = log_lik,
    ic = c(
      AIC = AIC(log_lik),
      BIC = BIC(log_lik),
      HQ = HQ(log_lik)
    )
  )
  class(res) <- "summary.vharlse"
  res
}
