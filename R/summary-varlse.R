#' Roots of characteristic polynomial
#' @param x object
#' @param ... not used
stableroot <- function(x, ...) {
  UseMethod("stableroot", x)
}

#' @describeIn Characteristic polynomial roots for VAR(p)
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

#' Stability
#' @param x object
#' @param ... not used
is.stable <- function(x, ...) {
  UseMethod("is.stable", x)
}

#' @describeIn Stability of VAR(p)
is.stable.varlse <- function(x, ...) {
  all(stableroot(x) < 1)
}

#' AIC of VAR(p)
#' 
#' Compute AIC of VAR(p)
#' @param object fit
#' @param ... not used
#' 
#' @export
AIC.varlse <- function(object, ...) {
  COV <- object$resid
  m <- object$m
  p <- object$p
  s <- object$obs
  sig_det <- det(crossprod(COV) / s)
  log(sig_det) + 2 / s * m * (p * m + 1)
}

#' FPE
#' @param object model fit
#' @param ... not used
FPE <- function(object, ...) {
  UseMethod("FPE", object)
}

#' @describeIn FPE of VAR(p)
FPE.varlse <- function(object, ...) {
  COV <- object$resid
  m <- object$m
  k <- m * object$p + 1
  s <- object$obs
  sig_det <- det(crossprod(COV) / s)
  ((s + k) / (s - k))^m * sig_det
}

#' BIC of VAR(p)
#' 
#' Compute BIC of VAR(p)
#' @param object varlse
#' @param ... not used
#' @export
BIC.varlse <- function(object, ...) {
  COV <- object$resid
  m <- object$m
  p <- object$p
  s <- object$obs
  sig_det <- det(crossprod(COV) / s)
  log(sig_det) + log(s) / s * m * (p * m + 1)
}

#' HQ
#' @param object model fit
#' @param ... not used
HQ <- function(object, ...) {
  UseMethod("HQ", object)
}

#' @describeIn HQ of VAR(p)
HQ.varlse <- function(object, ...) {
  COV <- object$resid
  m <- object$m
  p <- object$p
  s <- object$obs
  sig_det <- det(crossprod(COV) / s)
  log(sig_det) + 2 * log(log(s)) / s * m * (p * m + 1)
}

#' Summary of varlse
#' 
#' @param object varlse
#' @param ... not used
#' 
#' @export
summary.varlse <- function(object, ...) {
  var_name <- colnames(object$y0)
  cov_resid <- compute_var(object$residuals, object$obs, object$m * object$p + 1)
  colnames(cov_resid) <- var_name
  rownames(cov_resid) <- var_name
  # split the matrix for the print: B1, ..., Bp
  bhat_mat <- 
    split.data.frame(object$coefficients[-(object$m + object$p + 1),], gl(object$p, object$m)) %>% 
    lapply(t)
  bhat_mat$intercept <- object$coefficients[object$m * object$p + 1,]
  res <- list(
    names = var_name,
    totobs = object$totobs,
    obs = object$obs,
    coefficients = bhat_mat,
    call = object$call,
    process = object$process,
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
