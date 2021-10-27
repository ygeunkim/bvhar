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
#' @export
logLik.varlse <- function(object, ...) {
  obs <- object$obs
  k <- object$df
  m <- object$m
  cov_mle <- object$covmat * (obs - k) / obs # MLE = (s - k) / s * LS
  zhat <- object$residuals
  log_lik <- 
    -obs * m / 2 * log(2 * pi) - 
    obs / 2 * log(det(cov_mle)) - 
    sum(
      diag(
        zhat %*% solve(cov_mle) %*% t(zhat)
      )
    ) / 2
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
#' @export
logLik.vharlse <- function(object, ...) {
  obs <- object$obs
  k <- object$df
  m <- object$m
  cov_mle <- object$covmat * (obs - k) / obs
  zhat <- object$residuals
  log_lik <- 
    -obs * m / 2 * log(2 * pi) - 
    obs / 2 * log(det(cov_mle)) - 
    sum(
      diag(
        zhat %*% solve(cov_mle) %*% t(zhat)
      )
    ) / 2
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
#' @export
logLik.bvarmn <- function(object, ...) {
  obs <- object$obs
  k <- object$df
  m <- object$m
  posterior_cov <- object$iw_scale / (object$iw_shape - m - 1)
  zhat <- object$residuals
  log_lik <- 
    -obs * m / 2 * log(2 * pi) - 
    obs / 2 * log(det(posterior_cov)) - 
    sum(
      diag(
        zhat %*% solve(posterior_cov) %*% t(zhat)
      )
    ) / 2
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
#' @export
logLik.bvarflat <- function(object, ...) {
  obs <- object$obs
  k <- object$df
  m <- object$m
  posterior_cov <- object$iw_scale / (object$iw_shape - m - 1)
  zhat <- object$residuals
  log_lik <- 
    -obs * m / 2 * log(2 * pi) - 
    obs / 2 * log(det(posterior_cov)) - 
    sum(
      diag(
        zhat %*% solve(posterior_cov) %*% t(zhat)
      )
    ) / 2
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
#' @export
logLik.bvharmn <- function(object, ...) {
  obs <- object$obs
  k <- object$df
  m <- object$m
  posterior_cov <- object$iw_scale / (object$iw_shape - m - 1)
  zhat <- object$residuals
  log_lik <- 
    -obs * m / 2 * log(2 * pi) - 
    obs / 2 * log(det(posterior_cov)) - 
    sum(
      diag(
        zhat %*% solve(posterior_cov) %*% t(zhat)
      )
    ) / 2
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
#' @param type `r lifecycle::badge("experimental")` Only for `varlse` or `vharlse`, choose to compute based on log-likelihood (`loglik`: default) or determinant of Sigma (`rss`)
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
AIC.varlse <- function(object, type = c("loglik", "rss"), ...) {
  type <- match.arg(type)
  switch(
    type,
    "loglik" = {
      object %>% 
        logLik() %>% 
        AIC()
    },
    "rss" = {
      SIG <- object$covmat # crossprod(COV) / (s - k)
      m <- object$m
      k <- object$df
      s <- object$obs
      sig_det <- det(SIG) * ((s - k) / s)^m # det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
      log(sig_det) + 2 / s * m * k # penalty = (2 / s) * number of freely estimated parameters
    }
  )
}

#' @rdname AIC.varlse
#' 
#' @param object Model fit
#' @param type `r lifecycle::badge("experimental")` Only for `varlse` or `vharlse`, choose to compute based on log-likelihood (`loglik`: default) or determinant of Sigma (`rss`)
#' @param ... not used
#' 
#' @importFrom stats AIC
#' @export
AIC.vharlse <- function(object, type = c("loglik", "rss"), ...) {
  type <- match.arg(type)
  switch(
    type,
    "loglik" = {
      object %>% 
        logLik() %>% 
        AIC()
    },
    "rss" = {
      SIG <- object$covmat
      m <- object$m
      k <- object$df
      s <- object$obs
      sig_det <- det(SIG) * ((s - k) / s)^m
      log(sig_det) + 2 / s * m * k
    }
  )
}

#' @rdname AIC.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @importFrom stats AIC
#' @export
AIC.bvarmn <- function(object, ...) {
  object %>% 
    logLik() %>% 
    AIC()
}

#' @rdname AIC.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @importFrom stats AIC
#' @export
AIC.bvarflat <- function(object, ...) {
  object %>% 
    logLik() %>% 
    AIC()
}

#' @rdname AIC.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @importFrom stats AIC
#' @export
AIC.bvharmn <- function(object, ...) {
  object %>% 
    logLik() %>% 
    AIC()
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
#' @param type `r lifecycle::badge("experimental")` Only for `varlse` or `vharlse`, choose to compute based on log-likelihood (`loglik`: default) or determinant of Sigma (`rss`)
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
BIC.varlse <- function(object, type = c("loglik", "rss"), ...) {
  type <- match.arg(type)
  switch(
    type,
    "loglik" = {
      object %>% 
        logLik() %>% 
        BIC()
    },
    "rss" = {
      SIG <- object$covmat # crossprod(COV) / (s - k)
      m <- object$m
      k <- object$df
      s <- object$obs
      sig_det <- det(SIG) * ((s - k) / s)^m # det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
      log(sig_det) + log(s) / s * m * k # replace 2 / s with log(s) / s
    }
  )
}

#' @rdname BIC.varlse
#' 
#' @param object Model fit
#' @param type `r lifecycle::badge("experimental")` Only for `varlse` or `vharlse`, choose to compute based on log-likelihood (`loglik`: default) or determinant of Sigma (`rss`)
#' @param ... not used
#' 
#' @importFrom stats BIC
#' @export
BIC.vharlse <- function(object, type = c("loglik", "rss"), ...) {
  type <- match.arg(type)
  switch(
    type,
    "loglik" = {
      object %>% 
        logLik() %>% 
        BIC()
    },
    "rss" = {
      SIG <- object$covmat
      m <- object$m
      k <- object$df
      s <- object$obs
      sig_det <- det(SIG) * ((s - k) / s)^m
      log(sig_det) + log(s) / s * m * k
    }
  )
}

#' @rdname BIC.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @importFrom stats BIC
#' @export
BIC.bvarmn <- function(object, ...) {
  object %>% 
    logLik() %>% 
    BIC()
}

#' @rdname BIC.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @importFrom stats BIC
#' @export
BIC.bvarflat <- function(object, ...) {
  object %>% 
    logLik() %>% 
    BIC()
}

#' @rdname BIC.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @importFrom stats BIC
#' @export
BIC.bvharmn <- function(object, ...) {
  object %>% 
    logLik() %>% 
    BIC()
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

#' @rdname HQ
#' 
#' @details 
#' The formula is
#' 
#' \deqn{HQ = -2 \log p(y \mid \hat\theta) + k \log\log(n)}
#' 
#' whic can be computed by
#' `AIC(object, ..., k = 2 * log(log(nobs(object))))` with [stats::AIC()].
#' 
#' @references Hannan, E.J. and Quinn, B.G. (1979). *The Determination of the Order of an Autoregression*. Journal of the Royal Statistical Society: Series B (Methodological), 41: 190-195. [https://doi.org/10.1111/j.2517-6161.1979.tb01072.x](https://doi.org/10.1111/j.2517-6161.1979.tb01072.x)
#' 
#' @importFrom stats AIC nobs
#' @export
HQ.logLik <- function(object, ...) {
  AIC(object, k = 2 * log(log(nobs(object))))
}

#' Hannan-Quinn Criterion of Multivariate Time Series Model
#' 
#' Compute HQ of VAR(p), VHAR, BVAR(p), and BVHAR
#' 
#' @param object Model fit
#' @param type `r lifecycle::badge("experimental")` Only for `varlse` or `vharlse`, choose to compute based on log-likelihood (`loglik`: default) or determinant of Sigma (`rss`)
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
HQ.varlse <- function(object, type = c("loglik", "rss"), ...) {
  type <- match.arg(type)
  switch(
    type,
    "loglik" = {
      object %>% 
        logLik() %>% 
        HQ()
    },
    "rss" = {
      SIG <- object$covmat # crossprod(COV) / (s - k)
      m <- object$m
      k <- object$df
      s <- object$obs
      sig_det <- det(SIG) * ((s - k) / s)^m # det(crossprod(resid) / s) = det(SIG) * (s - k)^m / s^m
      log(sig_det) + 2 * log(log(s)) / s * m * k # replace log(s) / s with log(log(s)) / s
    }
  )
}

#' @rdname HQ.varlse
#' 
#' @param object Model fit
#' @param type `r lifecycle::badge("experimental")` Only for `varlse` or `vharlse`, choose to compute based on log-likelihood (`loglik`: default) or determinant of Sigma (`rss`)
#' @param ... not used
#' 
#' @export
HQ.vharlse <- function(object, ...) {
  type <- match.arg(type)
  switch(
    type,
    "loglik" = {
      object %>% 
        logLik() %>% 
        HQ()
    },
    "rss" = {
      SIG <- object$covmat
      m <- object$m
      k <- object$df
      s <- object$obs
      sig_det <- det(SIG) * ((s - k) / s)^m
      log(sig_det) + 2 * log(log(s)) / s * m * k
    }
  )
}

#' @rdname HQ.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @export
HQ.bvarmn <- function(object, ...) {
  object %>% 
    logLik() %>% 
    HQ()
}

#' @rdname HQ.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @export
HQ.bvarflat <- function(object, ...) {
  object %>% 
    logLik() %>% 
    HQ()
}

#' @rdname HQ.varlse
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @export
HQ.bvharmn <- function(object, ...) {
  object %>% 
    logLik() %>% 
    AIC()
}

#' Deviance Information Criterion
#' 
#' Generic function that computes DIC.
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @export
compute_dic <- function(object, ...) {
  UseMethod("compute_dic", object)
}

#' Deviance Information Criterion of Multivariate Time Series Model
#' 
#' Compute DIC of BVAR and BVHAR
#' 
#' @param object Model fit
#' @param n_iter Number to sample
#' @param ... not used
#' @details 
#' Deviance information criteria (DIC) is
#' 
#' \deqn{- 2 \log p(y \mid \hat\theta_{bayes}) + 2 p_{DIC}}
#' 
#' where \eqn{p_{DIC}} is the effective number of parameters defined by
#' 
#' \deqn{p_{DIC} = 2 ( \log p(y \mid \hat\theta_{bayes}) - E_{post} \log p(y \mid \theta) )}
#' 
#' Random sampling from posterior distribution gives its computation, \eqn{\theta_i \sim \theta \mid y, i = 1, \ldots, M}
#' 
#' \deqn{p_{DIC}^{computed} = 2 ( \log p(y \mid \hat\theta_{bayes}) - \frac{1}{M} \sum_i \log p(y \mid \theta_i) )}
#' 
#' @references 
#' Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). *Bayesian data analysis*. Chapman and Hall/CRC. [http://www.stat.columbia.edu/~gelman/book/](http://www.stat.columbia.edu/~gelman/book/)
#' 
#' Spiegelhalter, D.J., Best, N.G., Carlin, B.P. and Van Der Linde, A. (2002). *Bayesian measures of model complexity and fit*. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 64: 583-639. [https://doi.org/10.1111/1467-9868.00353](https://doi.org/10.1111/1467-9868.00353)
#' 
#' @export
compute_dic.bvarmn <- function(object, n_iter = 100L, ...) {
  rand_gen <- summary(object, n_iter = n_iter)
  bmat_gen <- rand_gen$coefficients
  covmat_gen <- rand_gen$covmat
  log_lik <- 
    object %>% 
    logLik() %>% 
    as.numeric()
  obs <- object$obs
  m <- object$m
  post_mean <- 
    lapply(
      1:n_iter,
      function(i) {
        zhat <- object$y0 - object$design %*% bmat_gen[,, i]
        posterior_cov <- covmat_gen[,, i] / (object$iw_shape - m - 1)
        -obs * m / 2 * log(2 * pi) - 
          obs / 2 * log(det(posterior_cov)) - 
          sum(
            diag(
              zhat %*% solve(posterior_cov) %*% t(zhat)
            )
          ) / 2
      }
    ) %>% 
    unlist()
  eff_num <- 2 * (log_lik - mean(post_mean))
  -2 * log_lik + 2 * eff_num
}

#' Extracting Log of Marginal Likelihood
#' 
#' Compute log of marginal likelihood of Bayesian Fit
#' 
#' @param object Model fit
#' @param ... not used
#' 
#' @export
compute_logml <- function(object, ...) {
  UseMethod("compute_logml", object)
}

#' @rdname compute_logml
#' 
#' @param object Model fit
#' @param ... not used
#' @details 
#' Closed form of Marginal Likelihood of BVAR can be derived by
#' 
#' \deqn{p(Y_0) = \pi^{-ms / 2} \frac{\Gamma_m ((\alpha_0 + s) / 2)}{\Gamma_m (\alpha_0 / 2)} \det(\Omega_0)^{-m / 2} \det(S_0)^{\alpha_0 / 2} \det(\hat{V})^{- m / 2} \det(\hat{\Sigma}_e)^{-(\alpha_0 + s) / 2}}
#' 
#' @importFrom CholWishart lmvgamma
#' @export
compute_logml.bvarmn <- function(object, ...) {
  dim_data <- object$m # m
  prior_shape <- object$prior_shape # alpha0
  num_obs <- object$obs # s
  # constant term-------------
  const_term <- - dim_data * num_obs / 2 * log(pi) + lmvgamma((prior_shape + num_obs) / 2, dim_data) - lmvgamma(prior_shape / 2, dim_data)
  # compute log ML-----------
  const_term - dim_data / 2 * log(
    det(object$prior_precision)
  ) + prior_shape / 2 * log(
    det(object$prior_scale)
  ) - dim_data / 2 * log(
    det(object$mn_prec)
  ) - (prior_shape + num_obs) / 2 * log(
    det(object$iw_scale)
  )
}

#' @rdname compute_logml
#' 
#' @param object Model fit
#' @param ... not used
#' @details 
#' Closed form of Marginal Likelihood of BVHAR can be derived by
#' 
#' \deqn{p(Y_0) = \pi^{-ms_0 / 2} \frac{\Gamma_m ((d_0 + s) / 2)}{\Gamma_m (d_0 / 2)} \det(P_0)^{-m / 2} \det(U_0)^{d_0 / 2} \det(\hat{V}_{HAR})^{- m / 2} \det(\hat{\Sigma}_e)^{-(d_0 + s) / 2}}
#' 
#' @importFrom CholWishart lmvgamma
#' @export
compute_logml.bvharmn <- function(object, ...) {
  dim_data <- object$m # m
  prior_shape <- object$prior_shape # d0
  num_obs <- object$obs # s
  # constant term-------------
  const_term <- - dim_data * num_obs / 2 * log(pi) + lmvgamma((prior_shape + num_obs) / 2, dim_data) - lmvgamma(prior_shape / 2, dim_data)
  # compute log ML------------
  const_term - dim_data / 2 * log(
    det(object$prior_precision)
  ) + prior_shape / 2 * log(
    det(object$prior_scale)
  ) - dim_data / 2 * log(
    det(object$mn_prec)
  ) - (prior_shape + num_obs) / 2 * log(
    det(object$iw_scale)
  )
}

