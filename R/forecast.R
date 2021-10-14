#' Forecasting Multivariate Time Series
#' 
#' Forecasts multivariate time series using given model.
#' 
#' @param object Model object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' @section n-step ahead forecasting VAR(p):
#' See pp35 of Lütkepohl (2007).
#' Consider h-step ahead forecasting (e.g. n + 1, ... n + h).
#' 
#' Let \eqn{y_{(n)}^T = (y_n^T, ..., y_{n - p + 1}^T, 1)}.
#' Then one-step ahead (point) forecasting:
#' \deqn{\hat{y}_{n + 1}^T = y_{(n)}^T \hat{B}}
#' 
#' Recursively, let \eqn{\hat{y}_{(n + 1)}^T = (\hat{y}_{n + 1}^T, y_n^T, ..., y_{n - p + 2}^T, 1)}.
#' Then two-step ahead (point) forecasting:
#' \deqn{\hat{y}_{n + 2}^T = \hat{y}_{(n + 1)}^T \hat{B}}
#' 
#' Similarly, h-step ahead (point) forecasting:
#' \deqn{\hat{y}_{n + h}^T = \hat{y}_{(n + h - 1)}^T \hat{B}}
#' 
#' How about confident region?
#' Confidence interval at h-period is
#' \deqn{y_{k,t}(h) \pm z_(\alpha / 2) \sigma_k (h)}
#' 
#' Joint forecast region of \eqn{100(1-\alpha)}% can be computed by
#' \deqn{\{ (y_{k, 1}, y_{k, h}) \mid y_{k, n}(i) - z_{(\alpha / 2h)} \sigma_n(i) \le y_{n, i} \le y_{k, n}(i) + z_{(\alpha / 2h)} \sigma_k(i), i = 1, \ldots, h \}}
#' See the pp41 of Lütkepohl (2007).
#' 
#' To compute covariance matrix, it needs VMA representation:
#' \deqn{Y_{t}(h) = c + \sum_{i = h}^{\infty} W_{i} \epsilon_{t + h - i} = c + \sum_{i = 0}^{\infty} W_{h + i} \epsilon_{t - i}}
#' 
#' Then
#' 
#' \deqn{\Sigma_y(h) = MSE [ y_t(h) ] = \sum_{i = 0}^{h - 1} W_i \Sigma_{\epsilon} W_i^T = \Sigma_y(h - 1) + W_{h - 1} \Sigma_{\epsilon} W_{h - 1}^T}
#' 
#' @return `predbvhar` [class] with the following components:
#' \describe{
#'   \item{process}{object$process}
#'   \item{forecast}{forecast matrix}
#'   \item{se}{standard error matrix}
#'   \item{lower}{lower confidence interval}
#'   \item{upper}{upper confidence interval}
#'   \item{lower_joint}{lower CI adjusted (Bonferroni)}
#'   \item{upper_joint}{upper CI adjusted (Bonferroni)}
#'   \item{y}{object$y}
#' }
#' 
#' @references 
#' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing. [https://doi.org/10.1007/978-3-540-27752-1](https://doi.org/10.1007/978-3-540-27752-1)
#' 
#' @importFrom stats qnorm
#' @order 1
#' @export
predict.varlse <- function(object, n_ahead, level = .05, ...) {
  pred_res <- forecast_var(object, n_ahead)
  colnames(pred_res) <- colnames(object$y0)
  SE <- 
    compute_covmse(object, n_ahead) %>% # concatenated matrix
    split.data.frame(gl(n_ahead, object$m)) %>% # list of forecast MSE covariance matrix
    sapply(diag) %>% 
    t() %>% # extract only diagonal element to compute CIs
    sqrt()
  colnames(SE) <- colnames(object$y0)
  z_quant <- qnorm(level / 2, lower.tail = FALSE)
  z_bonferroni <- qnorm(level / (2 * n_ahead), lower.tail = FALSE)
  res <- list(
    process = object$process,
    forecast = pred_res,
    se = SE,
    lower = pred_res - z_quant * SE,
    upper = pred_res + z_quant * SE,
    lower_joint = pred_res - z_bonferroni * SE,
    upper_joint = pred_res + z_bonferroni * SE,
    y = object$y
  )
  class(res) <- "predbvhar"
  res
}

#' @rdname predict.varlse
#' 
#' @param object `vharlse` object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' @section n-step ahead forecasting VHAR:
#' Let \eqn{T_{HAR}} is VHAR linear transformation matrix constructed by [scale_har()].
#' Since VHAR is the linearly transformed VAR(22),
#' let \eqn{y_{(n)}^T = (y_n^T, y_{n - 1}^T, ..., y_{n - 21}^T, 1)}.
#' 
#' Then one-step ahead (point) forecasting:
#' \deqn{\hat{y}_{n + 1}^T = y_{(n)}^T T_{HAR} \hat{\Phi}}
#' 
#' Recursively, let \eqn{\hat{y}_{(n + 1)}^T = (\hat{y}_{n + 1}^T, y_n^T, ..., y_{n - 20}^T, 1)}.
#' Then two-step ahead (point) forecasting:
#' \deqn{\hat{y}_{n + 2}^T = \hat{y}_{(n + 1)}^T T_{HAR} \hat{\Phi}}
#' 
#' and h-step ahead (point) forecasting:
#' \deqn{\hat{y}_{n + h}^T = \hat{y}_{(n + h - 1)}^T T_{HAR} \hat{\Phi}}
#' 
#' @references 
#' Baek, C. and Park, M. (2021). *Sparse vector heterogeneous autoregressive modeling for realized volatility*. J. Korean Stat. Soc. 50, 495–510. [https://doi.org/10.1007/s42952-020-00090-5](https://doi.org/10.1007/s42952-020-00090-5)
#' 
#' @importFrom stats qnorm
#' @order 1
#' @export
predict.vharlse <- function(object, n_ahead, level = .05, ...) {
  pred_res <- forecast_vhar(object, n_ahead)
  colnames(pred_res) <- colnames(object$y0)
  SE <- 
    compute_covmse_har(object, n_ahead) %>% # concatenated matrix
    split.data.frame(gl(n_ahead, object$m)) %>% # list of forecast MSE covariance matrix
    sapply(diag) %>% 
    t() %>% # extract only diagonal element to compute CIs
    sqrt()
  colnames(SE) <- colnames(object$y0)
  z_quant <- qnorm(level / 2, lower.tail = FALSE)
  z_bonferroni <- qnorm(level / (2 * n_ahead), lower.tail = FALSE)
  res <- list(
    process = object$process,
    forecast = pred_res,
    se = SE,
    lower = pred_res - z_quant * SE,
    upper = pred_res + z_quant * SE,
    lower_joint = pred_res - z_bonferroni * SE,
    upper_joint = pred_res + z_bonferroni * SE,
    y = object$y
  )
  class(res) <- "predbvhar"
  res
}

#' @rdname predict.varlse
#' 
#' @param object Model object
#' @param n_ahead step to forecast
#' @param n_iter Number to sample residual matrix from inverse-wishart distribution. By default, 100.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' 
#' @section n-step ahead forecasting BVAR(p) with minnesota prior:
#' Point forecasts are computed by posterior mean of the parameters.
#' See Section 3 of Bańbura et al. (2010).
#' 
#' Let \eqn{\hat{B}} be the posterior MN mean
#' and let \eqn{\hat{V}} be the posterior MN precision.
#' 
#' Then predictive posterior for each step
#' 
#' \deqn{y_{n + 1} \mid \Sigma_e, y \sim N( vec(y_{(n)}^T \hat{B}), \Sigma_e \otimes (1 + y_{(n)}^T \hat{V}^{-1} y_{(n)}) )}
#' \deqn{y_{n + 2} \mid \Sigma_e, y \sim N( vec(\hat{y}_{(n + 1)}^T \hat{B}), \Sigma_e \otimes (1 + \hat{y}_{(n + 1)}^T \hat{V}^{-1} \hat{y}_{(n + 1)}) )}
#' and recursively,
#' \deqn{y_{n + h} \mid \Sigma_e, y \sim N( vec(\hat{y}_{(n + h - 1)}^T \hat{B}), \Sigma_e \otimes (1 + \hat{y}_{(n + h - 1)}^T \hat{V}^{-1} \hat{y}_{(n + h - 1)}) )}
#' 
#' @references 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
#' 
#' Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). *Bayesian data analysis*. Chapman and Hall/CRC. [http://www.stat.columbia.edu/~gelman/book/](http://www.stat.columbia.edu/~gelman/book/)
#' 
#' @importFrom stats qnorm
#' @importFrom mniw riwish
#' @order 1
#' @export
predict.bvarmn <- function(object, n_ahead, n_iter = 100L, level = .05, ...) {
  pred_res <- forecast_bvarmn(object, n_ahead)
  # Point forecasting (Posterior mean)--------------
  pred_mean <- pred_res$posterior_mean
  colnames(pred_mean) <- colnames(object$y0)
  # Standard error----------------------------------
  pred_variance <- pred_res$posterior_var_closed
  sig_rand <- riwish(n = n_iter, Psi = object$iw_scale, nu = object$iw_shape)
  # Compute CI--------------------------------------
  ci_simul <- 
    lapply(
      1:n_iter,
      function(i) {
        lapply(
          pred_variance,
          function(v) {
            sqrt( diag(sig_rand[,, i]) * v ) # kronecker(sig_rand[,, i], v) %>% diag()
          }
        ) %>% 
          do.call(rbind, .)
      }
    ) %>% 
    simplify2array() %>% 
    apply(1:2, mean)
  colnames(ci_simul) <- colnames(object$y0)
  z_quant <- qnorm(level / 2, lower.tail = FALSE)
  z_bonferroni <- qnorm(level / (2 * n_ahead), lower.tail = FALSE)
  res <- list(
    process = object$process,
    forecast = pred_mean,
    se = ci_simul,
    lower = pred_mean - z_quant * ci_simul,
    upper = pred_mean + z_quant * ci_simul,
    lower_joint = pred_mean - z_bonferroni * ci_simul,
    upper_joint = pred_mean + z_bonferroni * ci_simul,
    y = object$y
  )
  class(res) <- "predbvhar"
  res
}

#' @rdname predict.varlse
#' 
#' @param object Model object
#' @param n_ahead step to forecast
#' @param n_iter Number to sample residual matrix from inverse-wishart distribution. By default, 100.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' @section n-step ahead forecasting BVHAR:
#' Let \eqn{\hat\Phi} be the posterior MN mean
#' and let \eqn{\hat\Psi} be the posterior MN precision.
#' 
#' Then predictive posterior for each step
#' 
#' \deqn{y_{n + 1} \mid \Sigma_e, y \sim N( vec(y_{(n)}^T \tilde{T}^T \hat\Phi), \Sigma_e \otimes (1 + y_{(n)}^T \tilde{T} \hat\Psi^{-1} \tilde{T} y_{(n)}) )}
#' \deqn{y_{n + 2} \mid \Sigma_e, y \sim N( vec(y_{(n + 1)}^T \tilde{T}^T \hat\Phi), \Sigma_e \otimes (1 + y_{(n + 1)}^T \tilde{T} \hat\Psi^{-1} \tilde{T} y_{(n + 1)}) )}
#' and recursively,
#' \deqn{y_{n + h} \mid \Sigma_e, y \sim N( vec(y_{(n + h - 1)}^T \tilde{T}^T \hat\Phi), \Sigma_e \otimes (1 + y_{(n + h - 1)}^T \tilde{T} \hat\Psi^{-1} \tilde{T} y_{(n + h - 1)}) )}
#' 
#' @importFrom stats qnorm
#' @importFrom mniw riwish
#' @order 1
#' @export
predict.bvharmn <- function(object, n_ahead, n_iter = 100L, level = .05, ...) {
  pred_res <- forecast_bvharmn(object, n_ahead)
  # Point forecasting (Posterior mean)--------------
  pred_mean <- pred_res$posterior_mean
  colnames(pred_mean) <- colnames(object$y0)
  # Standard error----------------------------------
  pred_variance <- pred_res$posterior_var_closed
  sig_rand <- riwish(n = n_iter, Psi = object$iw_scale, nu = object$iw_shape)
  # Compute CI--------------------------------------
  ci_simul <- 
    lapply(
      1:n_iter,
      function(i) {
        lapply(
          pred_variance,
          function(v) {
            sqrt( diag(sig_rand[,, i]) * v ) # kronecker(sig_rand[,, i], v) %>% diag()
          }
        ) %>% 
          do.call(rbind, .)
      }
    ) %>% 
    simplify2array() %>% 
    apply(1:2, mean)
  colnames(ci_simul) <- colnames(object$y0)
  z_quant <- qnorm(level / 2, lower.tail = FALSE)
  z_bonferroni <- qnorm(level / (2 * n_ahead), lower.tail = FALSE)
  # return-----------------------------------------
  res <- list(
    process = object$process,
    forecast = pred_mean,
    se = ci_simul,
    lower = pred_mean - z_quant * ci_simul,
    upper = pred_mean + z_quant * ci_simul,
    lower_joint = pred_mean - z_bonferroni * ci_simul,
    upper_joint = pred_mean + z_bonferroni * ci_simul,
    y = object$y
  )
  class(res) <- "predbvhar"
  res
}

#' @rdname predict.varlse
#' 
#' @param object Model object
#' @param n_ahead step to forecast
#' @param n_iter Number to sample residual matrix from inverse-wishart distribution. By default, 100.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' 
#' @references 
#' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). [https://doi:10.1080/01621459.2018.1437043](https://doi:10.1080/01621459.2018.1437043)
#' 
#' @importFrom stats qnorm
#' @importFrom mniw riwish
#' @order 1
#' @export
predict.bvarflat <- function(object, n_ahead, n_iter = 100L, level = .05, ...) {
  pred_res <- forecast_bvarmn_flat(object, n_ahead)
  # Point forecasting (Posterior mean)--------------
  pred_mean <- pred_res$posterior_mean
  colnames(pred_mean) <- colnames(object$y0)
  # Standard error----------------------------------
  pred_variance <- pred_res$posterior_var_closed
  sig_rand <- riwish(n = n_iter, Psi = object$iw_scale, nu = object$iw_shape)
  # Compute CI--------------------------------------
  ci_simul <- 
    lapply(
      1:n_iter,
      function(i) {
        lapply(
          pred_variance,
          function(v) {
            sqrt( diag(sig_rand[,, i]) * v ) # kronecker(sig_rand[,, i], v) %>% diag()
          }
        ) %>% 
          do.call(rbind, .)
      }
    ) %>% 
    simplify2array() %>% 
    apply(1:2, mean)
  colnames(ci_simul) <- colnames(object$y0)
  z_quant <- qnorm(level / 2, lower.tail = FALSE)
  z_bonferroni <- qnorm(level / (2 * n_ahead), lower.tail = FALSE)
  # return-----------------------------------------
  res <- list(
    process = object$process,
    forecast = pred_mean,
    se = ci_simul,
    lower = pred_mean - z_quant * ci_simul,
    upper = pred_mean + z_quant * ci_simul,
    lower_joint = pred_mean - z_bonferroni * ci_simul,
    upper_joint = pred_mean + z_bonferroni * ci_simul,
    y = object$y
  )
  class(res) <- "predbvhar"
  res
}

#' Print Method for \code{predbvhar} object
#' @rdname predict.varlse
#' @param x `predbvhar` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.predbvhar <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  print(x$forecast)
  invisible(x)
}

#' @rdname predict.varlse
#' @param x \code{predbvhar} object
#' @param ... not used
#' @order 3
#' @export
knit_print.predbvhar <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "predbvhar",
  knit_print.predbvhar,
  envir = asNamespace("knitr")
)

