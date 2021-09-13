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

#' AIC of VAR(p)
#' 
#' Compute AIC of VAR(p)
#' @param object fit
#' @param ... not used
#' 
#' @importFrom stats AIC
#' @export
AIC.varlse <- function(object, ...) {
  SIG <- object$covmat # crossprod(COV) / (s - k)
  m <- object$m
  k <- object$df
  s <- object$obs
  sig_det <- det(SIG) * ((s - k) / s)^m # det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  log(sig_det) + 2 / s * m * k # penalty = (2 / s) * m * k
}

#' FPE
#' @param object model fit
#' @param ... not used
#' 
#' @export
FPE <- function(object, ...) {
  UseMethod("FPE", object)
}

#' FPE of VAR(p)
#' 
#' @param object \code{varlse} object
#' @param ... not used
#' 
#' @export
FPE.varlse <- function(object, ...) {
  SIG <- object$covmat # SIG = crossprod(resid) / (s - k), FPE = ((s + k) / (s - k))^m * det(crossprod(resid) / s)
  m <- object$m
  k <- object$df
  s <- object$obs
  ((s + k) / s)^m * det(SIG) # FPE = ((s + k) / (s - k))^m * det = ((s + k) / s)^m * det(crossprod(resid) / (s - k))
}

#' BIC of VAR(p)
#' 
#' Compute BIC of VAR(p)
#' @param object \code{varlse} object
#' @param ... not used
#' 
#' @importFrom stats BIC
#' @export
BIC.varlse <- function(object, ...) {
  SIG <- object$covmat # crossprod(COV) / (s - k)
  m <- object$m
  k <- object$df
  s <- object$obs
  sig_det <- det(SIG) * ((s - k) / s)^m # det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  log(sig_det) + log(s) / s * m * k # penalty = (log(s) / s) * m * k
}

#' HQ
#' 
#' @param object model fit
#' @param ... not used
#' 
#' @export
HQ <- function(object, ...) {
  UseMethod("HQ", object)
}

#' HQ of VAR(p)
#' 
#' @param object \code{varlse} object
#' @param ... not used
#' 
#' @export
HQ.varlse <- function(object, ...) {
  SIG <- object$covmat # crossprod(COV) / (s - k)
  m <- object$m
  k <- object$df
  s <- object$obs
  sig_det <- det(SIG) * ((s - k) / s)^m # det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  log(sig_det) + 2 * log(log(s)) / s * m * k # penalty = (2 * log(log(s)) / s) * m * k
}

#' Summary of varlse
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
#' Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
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
    ic = c(
      AIC = AIC(object),
      BIC = BIC(object),
      HQ = HQ(object),
      FPE = FPE(object)
    )
  )
  class(res) <- "summary.varlse"
  res
}
