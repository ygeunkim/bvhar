#' Roots of characteristic polynomial
#' @param x object
#' @param ... not used
#' 
#' @export
stableroot <- function(x, ...) {
  UseMethod("stableroot", x)
}

#' Characteristic polynomial roots for VAR(p)
#' 
#' @param x \code{varlse} object
#' @param ... not used
#' 
#' @export
stableroot.varlse <- function(x, ...) {
  m <- x$m
  p <- x$p
  rbind(
    t(x$coefficients[-(m * p + 1),]), # without const term
    cbind(
      diag(m * (p - 1)),
      matrix(0L, nrow = m * (p - 1), ncol = m)
    )
  ) %>% 
    eigen() %>% 
    .$values %>% 
    Mod()
}

#' Stability of the process
#' 
#' @param x object
#' @param ... not used
#' 
#' @export
is.stable <- function(x, ...) {
  UseMethod("is.stable", x)
}

#' Stability of VAR(p)
#' 
#' @param x \code{varlse} object
#' @param ... not used
#' 
#' @export
is.stable.varlse <- function(x, ...) {
  all(stableroot(x) < 1)
}

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
  bhat_mat <- 
    switch(
      object$type,
      "const" = {
        split.data.frame(object$coefficients[-(object$m + object$p + 1),], gl(object$p, object$m)) %>% 
          lapply(t)
      },
      "none" = {
        split.data.frame(object$coefficients, gl(object$p, object$m)) %>% 
          lapply(t)
      }
    )
  if (object$type == "const") bhat_mat$intercept <- object$coefficients[object$m * object$p + 1,]
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
