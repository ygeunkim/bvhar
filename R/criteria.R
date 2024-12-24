#' @rdname var_lm
#' @param object A `varlse` object
#' @param ... not used
#' @details 
#' Consider the response matrix \eqn{Y_0}.
#' Let \eqn{T} be the total number of sample,
#' let \eqn{m} be the dimension of the time series,
#' let \eqn{p} be the order of the model,
#' and let \eqn{n = T - p}.
#' Likelihood of VAR(p) has
#' 
#' \deqn{Y_0 \mid B, \Sigma_e \sim MN(X_0 B, I_s, \Sigma_e)}
#' 
#' where \eqn{X_0} is the design matrix,
#' and MN is [matrix normal distribution](https://en.wikipedia.org/wiki/Matrix_normal_distribution).
#' 
#' Then log-likelihood of vector autoregressive model family is specified by
#' 
#' \deqn{\log p(Y_0 \mid B, \Sigma_e) = - \frac{nm}{2} \log 2\pi - \frac{n}{2} \log \det \Sigma_e - \frac{1}{2} tr( (Y_0 - X_0 B) \Sigma_e^{-1} (Y_0 - X_0 B)^T )}
#' 
#' In addition, recall that the OLS estimator for the matrix coefficient matrix is the same as MLE under the Gaussian assumption.
#' MLE for \eqn{\Sigma_e} has different denominator, \eqn{n}.
#' 
#' \deqn{\hat{B} = \hat{B}^{LS} = \hat{B}^{ML} = (X_0^T X_0)^{-1} X_0^T Y_0}
#' \deqn{\hat\Sigma_e = \frac{1}{s - k} (Y_0 - X_0 \hat{B})^T (Y_0 - X_0 \hat{B})}
#' \deqn{\tilde\Sigma_e = \frac{1}{s} (Y_0 - X_0 \hat{B})^T (Y_0 - X_0 \hat{B}) = \frac{s - k}{s} \hat\Sigma_e}
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

#' @rdname vhar_lm
#' @param object A `vharlse` object
#' @param ... not used
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

#' @rdname bvar_minnesota
#' @param object A `bvarmn` object
#' @param ... not used
#' @export
logLik.bvarmn <- function(object, ...) {
  obs <- object$obs
  k <- object$df
  m <- object$m
  posterior_cov <- object$covmat / (object$iw_shape - m - 1)
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

#' @rdname bvar_flat
#' @param object A `bvarflat` object
#' @param ... not used
#' @export
logLik.bvarflat <- function(object, ...) {
  obs <- object$obs
  k <- object$df
  m <- object$m
  posterior_cov <- object$covmat / (object$iw_shape - m - 1)
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

#' @rdname bvhar_minnesota
#' @param object A `bvharmn` object
#' @param ... not used
#' @export
logLik.bvharmn <- function(object, ...) {
  obs <- object$obs
  k <- object$df
  m <- object$m
  posterior_cov <- object$covmat / (object$iw_shape - m - 1)
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

#' @rdname var_lm
#' @param object A `varlse` object
#' @param ... not used
#' @details 
#' Let \eqn{\tilde{\Sigma}_e} be the MLE
#' and let \eqn{\hat{\Sigma}_e} be the unbiased estimator (`covmat`) for \eqn{\Sigma_e}.
#' Note that
#' 
#' \deqn{\tilde{\Sigma}_e = \frac{n - k}{n} \hat{\Sigma}_e}
#' 
#' Then
#' 
#' \deqn{AIC(p) = \log \det \Sigma_e + \frac{2}{n}(\text{number of freely estimated parameters})}
#' 
#' where the number of freely estimated parameters is \eqn{mk}, i.e. \eqn{pm^2} or \eqn{pm^2 + m}.
#' @references
#' Akaike, H. (1969). *Fitting autoregressive models for prediction*. Ann Inst Stat Math 21, 243-247.
#' 
#' Akaike, H. (1971). *Autoregressive model fitting for control*. Ann Inst Stat Math 23, 163-180.
#' 
#' Akaike H. (1974). *A new look at the statistical model identification*. IEEE Transactions on Automatic Control, vol. 19, no. 6, pp. 716-723.
#' 
#' Akaike H. (1998). *Information Theory and an Extension of the Maximum Likelihood Principle*. In: Parzen E., Tanabe K., Kitagawa G. (eds) Selected Papers of Hirotugu Akaike. Springer Series in Statistics (Perspectives in Statistics). Springer, New York, NY.
#' @importFrom stats AIC
#' @export
AIC.varlse <- function(object, ...) {
  object |> 
    logLik() |> 
    AIC()
}

#' @rdname vhar_lm
#' @param object A `vharlse` object
#' @param ... not used
#' @export
AIC.vharlse <- function(object, ...) {
  object |> 
    logLik() |> 
    AIC()
}

#' @rdname bvar_minnesota
#' @param object A `bvarmn` object
#' @param ... not used
#' @export
AIC.bvarmn <- function(object, ...) {
  object |> 
    logLik() |> 
    AIC()
}

#' @rdname bvar_flat
#' @param object A `bvarflat` object
#' @param ... not used
#' @export
AIC.bvarflat <- function(object, ...) {
  object |> 
    logLik() |> 
    AIC()
}

#' @rdname bvhar_minnesota
#' @param object A `bvharmn` object
#' @param ... not used
#' @export
AIC.bvharmn <- function(object, ...) {
  object |> 
    logLik() |> 
    AIC()
}

#' Final Prediction Error Criterion
#' 
#' Compute FPE of VAR(p) and VHAR
#' 
#' @param object Model fit
#' @param ... not used
#' @return FPE value.
#' @export
FPE <- function(object, ...) {
  UseMethod("FPE", object)
}

#' @rdname FPE
#' @details 
#' Let \eqn{\tilde{\Sigma}_e} be the MLE
#' and let \eqn{\hat{\Sigma}_e} be the unbiased estimator (`covmat`) for \eqn{\Sigma_e}.
#' Note that
#' 
#' \deqn{\tilde{\Sigma}_e = \frac{n - k}{T} \hat{\Sigma}_e}
#' 
#' Then
#' 
#' \deqn{FPE(p) = (\frac{n + k}{n - k})^m \det \tilde{\Sigma}_e}
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
#' @export
FPE.varlse <- function(object, ...) {
  compute_fpe(object)
}

#' @rdname FPE
#' @export
FPE.vharlse <- function(object, ...) {
  compute_fpe(object)
}

#' @rdname var_lm
#' @details 
#' Let \eqn{\tilde{\Sigma}_e} be the MLE
#' and let \eqn{\hat{\Sigma}_e} be the unbiased estimator (`covmat`) for \eqn{\Sigma_e}.
#' Note that
#' 
#' \deqn{\tilde{\Sigma}_e = \frac{n - k}{T} \hat{\Sigma}_e}
#' 
#' Then
#' 
#' \deqn{BIC(p) = \log \det \Sigma_e + \frac{\log n}{n}(\text{number of freely estimated parameters})}
#' 
#' where the number of freely estimated parameters is \eqn{pm^2}.
#' @references Gideon Schwarz. (1978). *Estimating the Dimension of a Model*. Ann. Statist. 6 (2) 461 - 464.
#' @importFrom stats BIC
#' @export
BIC.varlse <- function(object, ...) {
  object |> 
    logLik() |> 
    BIC()
}

#' @rdname vhar_lm
#' @export
BIC.vharlse <- function(object, ...) {
  object |> 
    logLik() |> 
    BIC()
}

#' @rdname bvar_minnesota
#' @export
BIC.bvarmn <- function(object, ...) {
  object |> 
    logLik() |> 
    BIC()
}

#' @rdname bvar_flat
#' @export
BIC.bvarflat <- function(object, ...) {
  object |> 
    logLik() |> 
    BIC()
}

#' @rdname bvhar_minnesota
#' @export
BIC.bvharmn <- function(object, ...) {
  object |> 
    logLik() |> 
    BIC()
}

#' Hannan-Quinn Criterion
#' 
#' Compute HQ of VAR(p), VHAR, BVAR(p), and BVHAR
#' 
#' @param object A `logLik` object or Model fit
#' @param ... not used
#' @return HQ value.
#' @export
HQ <- function(object, ...) {
  UseMethod("HQ", object)
}

#' @rdname HQ
#' @details 
#' The formula is
#' 
#' \deqn{HQ = -2 \log p(y \mid \hat\theta) + k \log\log(T)}
#' 
#' which can be computed by
#' `AIC(object, ..., k = 2 * log(log(nobs(object))))` with [stats::AIC()].
#' 
#' @references Hannan, E.J. and Quinn, B.G. (1979). *The Determination of the Order of an Autoregression*. Journal of the Royal Statistical Society: Series B (Methodological), 41: 190-195.
#' @importFrom stats AIC nobs
#' @export
HQ.logLik <- function(object, ...) {
  AIC(object, k = 2 * log(log(nobs(object))))
}

#' @rdname HQ
#' @details 
#' Let \eqn{\tilde{\Sigma}_e} be the MLE
#' and let \eqn{\hat{\Sigma}_e} be the unbiased estimator (`covmat`) for \eqn{\Sigma_e}.
#' Note that
#' 
#' \deqn{\tilde{\Sigma}_e = \frac{n - k}{T} \hat{\Sigma}_e}
#' 
#' Then
#' 
#' \deqn{HQ(p) = \log \det \Sigma_e + \frac{2 \log \log n}{n}(\text{number of freely estimated parameters})}
#' 
#' where the number of freely estimated parameters is \eqn{pm^2}.
#' @references
#' Hannan, E.J. and Quinn, B.G. (1979). *The Determination of the Order of an Autoregression*. Journal of the Royal Statistical Society: Series B (Methodological), 41: 190-195.
#' 
#' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
#' 
#' Quinn, B.G. (1980). *Order Determination for a Multivariate Autoregression*. Journal of the Royal Statistical Society: Series B (Methodological), 42: 182-185.
#' @export
HQ.varlse <- function(object, ...) {
  object |> 
    logLik() |> 
    HQ()
}

#' @rdname HQ
#' @export
HQ.vharlse <- function(object, ...) {
  object |> 
    logLik() |> 
    HQ()
}

#' @rdname HQ
#' @export
HQ.bvarmn <- function(object, ...) {
  object |> 
    logLik() |> 
    HQ()
}

#' @rdname HQ
#' @export
HQ.bvarflat <- function(object, ...) {
  object |> 
    logLik() |> 
    HQ()
}

#' @rdname HQ
#' @export
HQ.bvharmn <- function(object, ...) {
  object |> 
    logLik() |> 
    AIC()
}

#' Deviance Information Criterion of Multivariate Time Series Model
#' 
#' Compute DIC of BVAR and BVHAR.
#' 
#' @param object Model fit
#' @param ... not used
#' @return DIC value.
#' @export
compute_dic <- function(object, ...) {
  UseMethod("compute_dic", object)
}

#' @rdname compute_dic
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
#' Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). *Bayesian data analysis*. Chapman and Hall/CRC.
#' 
#' Spiegelhalter, D.J., Best, N.G., Carlin, B.P. and Van Der Linde, A. (2002). *Bayesian measures of model complexity and fit*. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 64: 583-639.
#' @export
compute_dic.bvarmn <- function(object, n_iter = 100L, ...) {
  rand_gen <- summary(object, n_iter = n_iter)
  bmat_gen <- rand_gen$coefficients
  covmat_gen <- rand_gen$covmat
  log_lik <- 
    object |> 
    logLik() |> 
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
    ) |> 
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
#' @return log likelihood of Minnesota prior model.
#' @references Giannone, D., Lenza, M., & Primiceri, G. E. (2015). *Prior Selection for Vector Autoregressions*. Review of Economics and Statistics, 97(2).
#' @export
compute_logml <- function(object, ...) {
  UseMethod("compute_logml", object)
}

#' @rdname compute_logml
#' @details 
#' Closed form of Marginal Likelihood of BVAR can be derived by
#' 
#' \deqn{p(Y_0) = \pi^{-mn / 2} \frac{\Gamma_m ((\alpha_0 + n) / 2)}{\Gamma_m (\alpha_0 / 2)} \det(\Omega_0)^{-m / 2} \det(S_0)^{\alpha_0 / 2} \det(\hat{V})^{- m / 2} \det(\hat{\Sigma}_e)^{-(\alpha_0 + n) / 2}}
#' @export
compute_logml.bvarmn <- function(object, ...) {
  dim_data <- object$m # m
  prior_shape <- object$prior_shape # alpha0
  num_obs <- object$obs # s
  # constant term-------------
  const_term <- - dim_data * num_obs / 2 * log(pi) + log_mgammafn((prior_shape + num_obs) / 2, dim_data) - log_mgammafn(prior_shape / 2, dim_data)
  # compute log ML-----------
  const_term + dim_data / 2 * log(
    det(object$prior_precision) # precision = scale^(-1)
  ) + prior_shape / 2 * log(
    det(object$prior_scale)
  ) - dim_data / 2 * log(
    det(object$mn_prec)
  ) - (prior_shape + num_obs) / 2 * log(
    det(object$covmat)
  )
}

#' @rdname compute_logml
#' @details 
#' Closed form of Marginal Likelihood of BVHAR can be derived by
#' 
#' \deqn{p(Y_0) = \pi^{-ms_0 / 2} \frac{\Gamma_m ((d_0 + n) / 2)}{\Gamma_m (d_0 / 2)} \det(P_0)^{-m / 2} \det(U_0)^{d_0 / 2} \det(\hat{V}_{HAR})^{- m / 2} \det(\hat{\Sigma}_e)^{-(d_0 + n) / 2}}
#' @export
compute_logml.bvharmn <- function(object, ...) {
  dim_data <- object$m # m
  prior_shape <- object$prior_shape # d0
  num_obs <- object$obs # s
  # constant term-------------
  const_term <- - dim_data * num_obs / 2 * log(pi) + log_mgammafn((prior_shape + num_obs) / 2, dim_data) - log_mgammafn(prior_shape / 2, dim_data)
  # compute log ML------------
  const_term + dim_data / 2 * log(
    det(object$prior_precision) # precision = scale^(-1)
  ) + prior_shape / 2 * log(
    det(object$prior_scale)
  ) - dim_data / 2 * log(
    det(object$mn_prec)
  ) - (prior_shape + num_obs) / 2 * log(
    det(object$covmat)
  )
}
