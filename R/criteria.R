#' Akaike's Information Criterion of Multivariate Time Series Model
#' 
#' Compute AIC of VAR(p), VHAR, BVAR(p), and BVHAR
#' 
#' @param object Model fit
#' @param ... not used
#' @details 
#' Let \eqn{\tilde{\Sigma}_e} be the MLE
#' and let \eqn{\hat{\Sigma}_e} be the unbiased estimator (from [compute_cov()] and the member named `covmat`) for \eqn{\Sigma_e}.
#' Note that
#' 
#' \deqn{\tilde{\Sigma}_e = \frac{s - k}{n} \hat{\Sigma}_e}
#' 
#' Then
#' 
#' \deqn{AIC(p) = \log \det \Sigma_e + \frac{2}{s}(\text{number of freely estimated parameters})}
#' 
#' where the number of freely estimated parameters is \eqn{pm^2}.
#' 
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' @importFrom stats AIC
#' @export
AIC.varlse <- function(object, ...) {
  SIG <- object$covmat # crossprod(COV) / (s - k)
  m <- object$m
  k <- object$df
  s <- object$obs
  sig_det <- det(SIG) * ((s - k) / s)^m # det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  log(sig_det) + 2 / s * object$p * m^2 # penalty = (2 / s) * p * m^2
}

#' @rdname AIC.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @importFrom stats AIC
#' @export
AIC.vharlse <- function(object, ...) {
  SIG <- object$covmat
  m <- object$m
  k <- object$df
  s <- object$obs
  sig_det <- det(SIG) * ((s - k) / s)^m
  log(sig_det) + 2 / s * 3 * m^2
}

#' Final Prediction Error Criterion
#' 
#' Generic function that computes FPE criterion.
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @export
FPE <- function(object, ...) {
  UseMethod("FPE", object)
}

#' Final Prediction Error Criterion of Multivariate Time Series Model
#' 
#' Compute FPE of VAR(p), VHAR, BVAR(p), and BVHAR
#' 
#' @param object Model fit
#' @param ... not used
#' @details 
#' Let \eqn{\tilde{\Sigma}_e} be the MLE
#' and let \eqn{\hat{\Sigma}_e} be the unbiased estimator (from [compute_cov()] and the member named `covmat`) for \eqn{\Sigma_e}.
#' Note that
#' 
#' \deqn{\tilde{\Sigma}_e = \frac{s - k}{n} \hat{\Sigma}_e}
#' 
#' Then
#' 
#' \deqn{FPE(p) = (\frac{s + k}{s - k})^m \det \tilde{\Sigma}_e}
#' 
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' @export
FPE.varlse <- function(object, ...) {
  SIG <- object$covmat # SIG = crossprod(resid) / (s - k), FPE = ((s + k) / (s - k))^m * det(crossprod(resid) / s)
  m <- object$m
  k <- object$df
  s <- object$obs
  ((s + k) / s)^m * det(SIG) # FPE = ((s + k) / (s - k))^m * det = ((s + k) / s)^m * det(crossprod(resid) / (s - k))
}

#' @rdname FPE.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @export
FPE.vharlse <- function(object, ...) {
  SIG <- object$covmat
  m <- object$m
  k <- object$df
  s <- object$obs
  ((s + k) / s)^m * det(SIG)
}

#' Bayesian Information Criterion of Multivariate Time Series Model
#' 
#' Compute BIC of VAR(p), VHAR, BVAR(p), and BVHAR
#' 
#' @param object Model fit
#' @param ... not used
#' @details 
#' Let \eqn{\tilde{\Sigma}_e} be the MLE
#' and let \eqn{\hat{\Sigma}_e} be the unbiased estimator (from [compute_cov()] and the member named `covmat`) for \eqn{\Sigma_e}.
#' Note that
#' 
#' \deqn{\tilde{\Sigma}_e = \frac{s - k}{n} \hat{\Sigma}_e}
#' 
#' Then
#' 
#' \deqn{BIC(p) = \log \det \Sigma_e + \frac{\log s}{s}(\text{number of freely estimated parameters})}
#' 
#' where the number of freely estimated parameters is \eqn{pm^2}.
#' 
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' @importFrom stats BIC
#' @export
BIC.varlse <- function(object, ...) {
  SIG <- object$covmat # crossprod(COV) / (s - k)
  m <- object$m
  k <- object$df
  s <- object$obs
  sig_det <- det(SIG) * ((s - k) / s)^m # det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  log(sig_det) + log(s) / s * object$p * m^2 # penalty = (log(s) / s) * p * m^2
}

#' @rdname BIC.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @importFrom stats BIC
#' @export
BIC.vharlse <- function(object, ...) {
  SIG <- object$covmat
  m <- object$m
  k <- object$df
  s <- object$obs
  sig_det <- det(SIG) * ((s - k) / s)^m
  log(sig_det) + log(s) / s * 3 * m^2
}

#' Hannan-Quinn Criterion
#' 
#' Generic function that computes HQ criterion.
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @export
HQ <- function(object, ...) {
  UseMethod("HQ", object)
}

#' Hannan-Quinn Criterion of Multivariate Time Series Model
#' 
#' Compute HQ of VAR(p), VHAR, BVAR(p), and BVHAR
#' 
#' @param object Model fit
#' @param ... not used
#' @details 
#' Let \eqn{\tilde{\Sigma}_e} be the MLE
#' and let \eqn{\hat{\Sigma}_e} be the unbiased estimator (from [compute_cov()] and the member named `covmat`) for \eqn{\Sigma_e}.
#' Note that
#' 
#' \deqn{\tilde{\Sigma}_e = \frac{s - k}{n} \hat{\Sigma}_e}
#' 
#' Then
#' 
#' \deqn{HQ(p) = \log \det \Sigma_e + \frac{2 \log \log s}{s}(\text{number of freely estimated parameters})}
#' 
#' where the number of freely estimated parameters is \eqn{pm^2}.
#' 
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' @export
HQ.varlse <- function(object, ...) {
  SIG <- object$covmat # crossprod(COV) / (s - k)
  m <- object$m
  k <- object$df
  s <- object$obs
  sig_det <- det(SIG) * ((s - k) / s)^m # det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  log(sig_det) + 2 * log(log(s)) / s * object$p * m^2 # penalty = (2 * log(log(s)) / s) * p * m^2
}

#' @rdname HQ.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @export
HQ.vharlse <- function(object, ...) {
  SIG <- object$covmat
  m <- object$m
  k <- object$df
  s <- object$obs
  sig_det <- det(SIG) * ((s - k) / s)^m
  log(sig_det) + 2 * log(log(s)) / s * 3 * m^2
}
