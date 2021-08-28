#' Predict Method for \code{varlse} object
#' 
#' Forecasting VAR(p)
#' 
#' @param object \code{varlse} object
#' @param n.ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' @details 
#' n-step ahead forecasting using VAR(p) recursively.
#' See pp35 of Lütkepohl (2007).
#' Consider h-step forecasting (e.g. n + 1, ... n + h).
#' 
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
#' @return \code{predbvhar} \link{class} with the following components:
#' \describe{
#'   \item{process}{object class: varlse}
#'   \item{forecast}{forecast matrix}
#'   \item{se}{standard error matrix}
#'   \item{lower}{lower confidence interval}
#'   \item{upper}{upper confidence interval}
#'   \item{lower_joint}{lower CI adjusted (Bonferroni)}
#'   \item{upper_joint}{upper CI adjusted (Bonferroni)}
#'   \item{y}{varlse$y}
#' }
#' 
#' @references 
#' Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
#' @order 1
#' @export
predict.varlse <- function(object, n.ahead, level = .05, ...) {
  pred_res <- forecast_var(object, n.ahead)
  colnames(pred_res) <- colnames(object$y0)
  SE <- 
    compute_covmse(object, n.ahead) %>% # concatenated matrix
    split.data.frame(gl(n.ahead, object$m)) %>% # list of forecast MSE covariance matrix
    sapply(diag) %>% 
    sqrt() %>% 
    t() # extract only diagonal element to compute CIs
  colnames(SE) <- colnames(object$y0)
  z_quant <- qnorm(level / 2, lower.tail = FALSE)
  z_bonferroni <- qnorm(level / (2 * n.ahead), lower.tail = FALSE)
  res <- list(
    process = "varlse",
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

#' Predict Method for \code{vharlse} object
#' 
#' Forecasting VHAR
#' 
#' @param object \code{vharlse} object
#' @param n.ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' @details 
#' n-step ahead forecasting using VHAR recursively
#' @return \code{predbvhar} \link{class} with the following components:
#' \describe{
#'   \item{process}{object class: vharlse}
#'   \item{forecast}{forecast matrix}
#'   \item{se}{standard error matrix}
#'   \item{lower}{lower confidence interval}
#'   \item{upper}{upper confidence interval}
#'   \item{lower_joint}{lower CI adjusted (Bonferroni)}
#'   \item{upper_joint}{upper CI adjusted (Bonferroni)}
#'   \item{y}{vharlse$y}
#' }
#' 
#' @order 1
#' @export
predict.vharlse <- function(object, n.ahead, level = .05, ...) {
  pred_res <- forecast_vhar(object, n.ahead)
  colnames(pred_res) <- colnames(object$y0)
  SE <- 
    compute_covmse_har(object, n.ahead) %>% # concatenated matrix
    split.data.frame(gl(n.ahead, object$m)) %>% # list of forecast MSE covariance matrix
    sapply(diag) %>% 
    sqrt() %>% 
    t() # extract only diagonal element to compute CIs
  colnames(SE) <- colnames(object$y0)
  z_quant <- qnorm(level / 2, lower.tail = FALSE)
  z_bonferroni <- qnorm(level / (2 * n.ahead), lower.tail = FALSE)
  res <- list(
    process = "vharlse",
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

#' Predict Method for \code{bvarmn} object
#' 
#' Forecasting for Minnesota BVAR
#' 
#' @param object \code{bvarmn} object
#' @param n.ahead step to forecast
#' @param n_iter Number to sample residual matrix from inverse-wishart distribution. By default, 100.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' 
#' @details 
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
#' @return \code{predbvhar} \link{class} with the following components:
#' \describe{
#'   \item{process}{object class: bvarmn}
#'   \item{forecast}{forecast matrix}
#'   \item{se}{standard error matrix}
#'   \item{lower}{lower confidence interval}
#'   \item{upper}{upper confidence interval}
#'   \item{lower_joint}{lower CI adjusted (Bonferroni)}
#'   \item{upper_joint}{upper CI adjusted (Bonferroni)}
#'   \item{y}{bvarmn$y}
#' }
#' 
#' @references 
#' Litterman, R. B. (1986). \emph{Forecasting with Bayesian Vector Autoregressions: Five Years of Experience}. Journal of Business & Economic Statistics, 4(1), 25. \url{https://doi:10.2307/1391384}
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). \emph{Large Bayesian vector auto regressions}. Journal of Applied Econometrics, 25(1). \url{https://doi:10.1002/jae.1137}
#' 
#' @importFrom mniw riwish
#' @order 1
#' @export
predict.bvarmn <- function(object, n.ahead, n_iter = 100L, level = .05, ...) {
  pred_res <- forecast_bvarmn(object, n.ahead)
  # Point forecasting (Posterior mean)--------------
  pred_mean <- pred_res$posterior_mean
  colnames(pred_mean) <- colnames(object$y0)
  # Standard error----------------------------------
  pred_variance <- pred_res$posterior_var_closed
  prior_sigmean <- object$prior_scale / (object$prior_shape - object$m - 1)
  # Compute CI--------------------------------------
  ci_simul <- 
    lapply(
      pred_variance,
      function(v) {
        kronecker(prior_sigmean, v) %>% diag()
      }
    ) %>% 
    do.call(rbind, .) %>% 
    sqrt()
  colnames(ci_simul) <- colnames(object$y0)
  z_quant <- qnorm(level / 2, lower.tail = FALSE)
  z_bonferroni <- qnorm(level / (2 * n.ahead), lower.tail = FALSE)
  res <- list(
    process = "bvarmn",
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

#' Predict Method for \code{bvharmn} object
#' 
#' Forecasting BVHAR
#' 
#' @param object \code{vharlse} object
#' @param n.ahead step to forecast
#' @param n_iter Number to sample residual matrix from inverse-wishart distribution. By default, 100.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' @details 
#' n-step ahead forecasting using VHAR recursively.
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
#' @return \code{predbvhar} \link{class} with the following components:
#' \describe{
#'   \item{process}{object class: bvharmn}
#'   \item{forecast}{forecast matrix}
#'   \item{se}{standard error matrix}
#'   \item{lower}{lower confidence interval}
#'   \item{upper}{upper confidence interval}
#'   \item{lower_joint}{lower CI adjusted (Bonferroni)}
#'   \item{upper_joint}{upper CI adjusted (Bonferroni)}
#'   \item{y}{bvharmn$y}
#' }
#' 
#' @importFrom mniw riwish
#' @order 1
#' @export
predict.bvharmn <- function(object, n.ahead, n_iter = 100L, level = .05, ...) {
  pred_res <- forecast_bvharmn(object, n.ahead)
  # Point forecasting (Posterior mean)--------------
  pred_mean <- pred_res$posterior_mean
  colnames(pred_mean) <- colnames(object$y0)
  # Standard error----------------------------------
  pred_variance <- pred_res$posterior_var_closed
  prior_sigmean <- object$prior_scale / (object$prior_shape - object$m - 1)
  # Compute CI--------------------------------------
  ci_simul <- 
    lapply(
      pred_variance,
      function(v) {
        kronecker(prior_sigmean, v) %>% diag()
      }
    ) %>% 
    do.call(rbind, .) %>% 
    sqrt()
  colnames(ci_simul) <- colnames(object$y0)
  z_quant <- qnorm(level / 2, lower.tail = FALSE)
  z_bonferroni <- qnorm(level / (2 * n.ahead), lower.tail = FALSE)
  # return-----------------------------------------
  res <- list(
    process = "bvharmn",
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

#' Predict Method for \code{bvarghosh} object
#' 
#' Point forecasting for Nonhierarchical Ghosh BVAR(p)
#' 
#' @param object \code{bvarghosh} object
#' @param n.ahead step to forecast
#' @param n_iter Number to sample residual matrix from inverse-wishart distribution. By default, 100.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' 
#' @details 
#' n-step ahead forecasting using BVAR recursively
#' @return \code{predbvhar} \link{class} with the following components:
#' \describe{
#'   \item{process}{object class: bvarghosh}
#'   \item{forecast}{forecast matrix}
#'   \item{y}{bvarflat$y}
#' }
#' 
#' @importFrom mniw riwish
#' @order 1
#' @export
predict.bvarflat <- function(object, n.ahead, n_iter = 100L, level = .05, ...) {
  pred_res <- forecast_bvarmn_flat(object, n.ahead)
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
            kronecker(sig_rand[,, i], v) %>% diag()
          }
        ) %>% 
          do.call(rbind, .)
      }
    ) %>% 
    simplify2array() %>% 
    sqrt() %>% 
    apply(1:2, mean)
  colnames(ci_simul) <- colnames(object$y0)
  z_quant <- qnorm(level / 2, lower.tail = FALSE)
  z_bonferroni <- qnorm(level / (2 * n.ahead), lower.tail = FALSE)
  # return-----------------------------------------
  res <- list(
    process = "bvarflat",
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

#' See if the Object \code{predbvhar}
#' 
#' This function returns \code{TRUE}
#' if the input is the output of \code{\link{predict.varlse}}, \code{\link{predict.vharlse}}, \code{\link{predict.bvarmn}}, \code{\link{predict.bvharmn}}, and \code{\link{predict.bvarghosh}}.
#' 
#' @param x object
#' 
#' @return \code{TRUE} or \code{FALSE}
#' 
#' @export
is.predbvhar <- function(x) {
  inherits(x, "predbvhar")
}

#' Print Method for \code{predbvhar} object
#' @rdname print.predbvhar
#' @param x \code{predbvhar} object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.predbvhar <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  print(x$forecast)
  invisible(x)
}

#' @rdname print.predbvhar
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

