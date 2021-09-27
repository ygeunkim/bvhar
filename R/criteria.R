#' Extract Log-Likelihood of Multivariate Time Series Model
#' 
#' Compute log-likelihood function value of VAR(p), VHAR, BVAR(p), and BVHAR
#' 
#' @param object Model fit
#' @param ... not used
#' @details 
#' Consider \eqn{Y_0} matrix from [build_y0()].
#' Let \eqn{n} be the total number of sample,
#' let \eqn{m} be the dimension of the time series,
#' let \eqn{p} be the order of the model,
#' and let \eqn{s = n - p}.
#' Likelihood of VAR(p) has
#' 
#' \deqn{Y_0 \mid B, \Sigma_e \sim MN(X_0 B, I_s, \Sigma_e)}
#' 
#' where \eqn{X_0} from [build_design()],
#' and MN is [matrix normal distribution](https://en.wikipedia.org/wiki/Matrix_normal_distribution).
#' 
#' Then log-likelihood of vector autoregressive model family is specified by
#' 
#' \deqn{\log p(Y_0 \mid B, \Sigma_e) = - \frac{sm}{2} \log 2\pi - \frac{s}{2} \log \det \Sigma_e - \frac{1}{2} tr( (Y_0 - X_0 B) \Sigma_e^{-1} (Y_0 - X_0 B)^T )}
#' 
#' In addition, recall that the OLS estimator for the matrix coefficient matrix is the same as MLE under the Gaussian assumption.
#' MLE for \eqn{\Sigma_e} has different denominator, \eqn{s}.
#' 
#' \deqn{\hat{B} = \hat{B}^{LS} = \hat{B}^{ML} = (X_0^T X_0)^{-1} X_0^T Y_0}
#' \deqn{\hat\Sigma_e = \frac{1}{s - k} (Y_0 - X_0 \hat{B})^T (Y_0 - X_0 \hat{B})}
#' \deqn{\tilde\Sigma_e = \frac{1}{s} (Y_0 - X_0 \hat{B})^T (Y_0 - X_0 \hat{B}) = \frac{s - k}{s} \hat\Sigma_e}
#' 
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' @seealso [var_lm()]
#' 
#' @importFrom stats logLik
#' @importFrom mniw dMNorm
#' 
#' @export
logLik.varlse <- function(object, ...) {
  obs <- object$obs
  k <- object$df
  m <- object$m
  cov_mle <- object$covmat * (obs - k) / obs # MLE = (s - k) / s * LS
  log_lik <- dMNorm(
    X = object$y0,
    Lambda = object$fitted.values,
    SigmaR = diag(obs),
    SigmaC = cov_mle,
    log = TRUE
  )
  class(log_lik) <- "logLik"
  attr(log_lik, "df") <- k * m + m^2 # cf, mk + m if iid
  attr(log_lik, "nobs") <- obs
  log_lik
}

#' @rdname logLik.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' @details 
#' In case of VHAR, just consider the linear relationship.
#' 
#' @references Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174–196. [https://doi:10.1093/jjfinec/nbp001](https://doi:10.1093/jjfinec/nbp001)
#' 
#' @seealso [vhar_lm()]
#' 
#' @importFrom stats logLik
#' @importFrom mniw dMNorm
#' @export
logLik.vharlse <- function(object, ...) {
  obs <- object$obs
  k <- object$df
  m <- object$m
  cov_mle <- object$covmat * (obs - k) / obs
  log_lik <- dMNorm(
    X = object$y0,
    Lambda = object$fitted.values,
    SigmaR = diag(obs),
    SigmaC = cov_mle,
    log = TRUE
  )
  class(log_lik) <- "logLik"
  attr(log_lik, "df") <- k * m + m^2
  attr(log_lik, "nobs") <- obs
  log_lik
}

#' @rdname logLik.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' @details 
#' While frequentist models use OLS and MLE for coefficient and covariance matrices, Bayesian models implement posterior means.
#' 
#' @references 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
#' 
#' @seealso [bvar_minnesota()]
#' 
#' @importFrom stats logLik
#' @importFrom mniw dMNorm
#' @export
logLik.bvarmn <- function(object, ...) {
  obs <- object$obs
  k <- object$df
  m <- object$m
  posterior_cov <- object$iw_scale / (object$iw_shape - m - 1)
  log_lik <- dMNorm(
    X = object$y0,
    Lambda = object$fitted.values,
    SigmaR = diag(obs),
    SigmaC = posterior_cov,
    log = TRUE
  )
  class(log_lik) <- "logLik"
  attr(log_lik, "df") <- k * m + m^2
  attr(log_lik, "nobs") <- obs
  log_lik
}

#' @rdname logLik.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @references Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). [https://doi:10.1080/01621459.2018.1437043](https://doi:10.1080/01621459.2018.1437043)
#' 
#' @seealso [bvar_flat()]
#' 
#' @importFrom stats logLik
#' @importFrom mniw dMNorm
#' @export
logLik.bvarflat <- function(object, ...) {
  obs <- object$obs
  k <- object$df
  m <- object$m
  posterior_cov <- object$iw_scale / (object$iw_shape - m - 1)
  log_lik <- dMNorm(
    X = object$y0,
    Lambda = object$fitted.values,
    SigmaR = diag(obs),
    SigmaC = posterior_cov,
    log = TRUE
  )
  class(log_lik) <- "logLik"
  attr(log_lik, "df") <- k * m + m^2
  attr(log_lik, "nobs") <- obs
  log_lik
}

#' @rdname logLik.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @seealso [bvhar_minnesota()]
#' 
#' @importFrom stats logLik
#' @importFrom mniw dMNorm
#' @export
logLik.bvharmn <- function(object, ...) {
  obs <- object$obs
  k <- object$df
  m <- object$m
  posterior_cov <- object$iw_scale / (object$iw_shape - m - 1)
  log_lik <- dMNorm(
    X = object$y0,
    Lambda = object$fitted.values,
    SigmaR = diag(obs),
    SigmaC = posterior_cov,
    log = TRUE
  )
  class(log_lik) <- "logLik"
  attr(log_lik, "df") <- k * m + m^2
  attr(log_lik, "nobs") <- obs
  log_lik
}

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
#' \deqn{\tilde{\Sigma}_e = \frac{s - k}{s} \hat{\Sigma}_e}
#' 
#' Then
#' 
#' \deqn{AIC(p) = \log \det \Sigma_e + \frac{2}{s}(\text{number of freely estimated parameters})}
#' 
#' where the number of freely estimated parameters is \eqn{mk}, i.e. \eqn{pm^2} or \eqn{pm^2 + m}.
#' 
#' @references
#' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' Akaike, H. (1969). *Fitting autoregressive models for prediction*. Ann Inst Stat Math 21, 243–247. [https://doi.org/10.1007/BF02532251](https://doi.org/10.1007/BF02532251)
#' 
#' Akaike, H. (1971). *Autoregressive model fitting for control*. Ann Inst Stat Math 23, 163–180. [https://doi.org/10.1007/BF02479221](https://doi.org/10.1007/BF02479221)
#' 
#' Akaike H. (1998). *Information Theory and an Extension of the Maximum Likelihood Principle*. In: Parzen E., Tanabe K., Kitagawa G. (eds) Selected Papers of Hirotugu Akaike. Springer Series in Statistics (Perspectives in Statistics). Springer, New York, NY. [https://doi.org/10.1007/978-1-4612-1694-0_15](https://doi.org/10.1007/978-1-4612-1694-0_15)
#' 
#' Akaike H. (1974). *A new look at the statistical model identification*. IEEE Transactions on Automatic Control, vol. 19, no. 6, pp. 716-723. doi: [10.1109/TAC.1974.1100705](https://ieeexplore.ieee.org/document/1100705).
#' 
#' @importFrom stats AIC
#' 
#' @export
AIC.varlse <- function(object, ...) {
  SIG <- object$covmat # crossprod(COV) / (s - k)
  m <- object$m
  k <- object$df
  s <- object$obs
  sig_det <- det(SIG) * ((s - k) / s)^m # det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  log(sig_det) + 2 / s * m * k # penalty = (2 / s) * number of freely estimated parameters
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
  log(sig_det) + 2 / s * m * k
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
#' @references 
#' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' Gideon Schwarz. (1978). *Estimating the Dimension of a Model*. Ann. Statist. 6 (2) 461 - 464. [https://doi.org/10.1214/aos/1176344136](https://doi.org/10.1214/aos/1176344136)
#' 
#' @importFrom stats BIC
#' 
#' @export
BIC.varlse <- function(object, ...) {
  SIG <- object$covmat # crossprod(COV) / (s - k)
  m <- object$m
  k <- object$df
  s <- object$obs
  sig_det <- det(SIG) * ((s - k) / s)^m # det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  log(sig_det) + log(s) / s * m * k # replace 2 / s with log(s) / s
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
  log(sig_det) + log(s) / s * m * k
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
#' @references
#' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' Hannan, E.J. and Quinn, B.G. (1979). *The Determination of the Order of an Autoregression*. Journal of the Royal Statistical Society: Series B (Methodological), 41: 190-195. [https://doi.org/10.1111/j.2517-6161.1979.tb01072.x](https://doi.org/10.1111/j.2517-6161.1979.tb01072.x)
#' 
#' Quinn, B.G. (1980). *Order Determination for a Multivariate Autoregression*. Journal of the Royal Statistical Society: Series B (Methodological), 42: 182-185. [https://doi.org/10.1111/j.2517-6161.1980.tb01116.x](https://doi.org/10.1111/j.2517-6161.1980.tb01116.x)
#' 
#' @export
HQ.varlse <- function(object, ...) {
  SIG <- object$covmat # crossprod(COV) / (s - k)
  m <- object$m
  k <- object$df
  s <- object$obs
  sig_det <- det(SIG) * ((s - k) / s)^m # det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
  log(sig_det) + 2 * log(log(s)) / s * m * k # replace log(s) / s with log(log(s)) / s
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
  log(sig_det) + 2 * log(log(s)) / s * m * k
}
