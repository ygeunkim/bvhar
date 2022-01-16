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
#' Let \eqn{T_{HAR}} is VHAR linear transformation matrix (See [var_design_formulation]).
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
#' \deqn{y_{n + 1} \mid \Sigma_e, y \sim N( vec(y_{(n)}^T A), \Sigma_e \otimes (1 + y_{(n)}^T \hat{V}^{-1} y_{(n)}) )}
#' \deqn{y_{n + 2} \mid \Sigma_e, y \sim N( vec(\hat{y}_{(n + 1)}^T A), \Sigma_e \otimes (1 + \hat{y}_{(n + 1)}^T \hat{V}^{-1} \hat{y}_{(n + 1)}) )}
#' and recursively,
#' \deqn{y_{n + h} \mid \Sigma_e, y \sim N( vec(\hat{y}_{(n + h - 1)}^T A), \Sigma_e \otimes (1 + \hat{y}_{(n + h - 1)}^T \hat{V}^{-1} \hat{y}_{(n + h - 1)}) )}
#' 
#' See [bvar_predictive_density] how to generate the predictive distribution.
#' 
#' @references 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
#' 
#' Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). *Bayesian data analysis*. Chapman and Hall/CRC. [http://www.stat.columbia.edu/~gelman/book/](http://www.stat.columbia.edu/~gelman/book/)
#' 
#' Karlsson, S. (2013). *Chapter 15 Forecasting with Bayesian Vector Autoregression*. Handbook of Economic Forecasting, 2, 791–897. doi:[10.1016/b978-0-444-62731-5.00015-4](https://doi.org/10.1016/B978-0-444-62731-5.00015-4)
#' 
#' @importFrom stats quantile
#' @order 1
#' @export
predict.bvarmn <- function(object, n_ahead, n_iter = 100L, level = .05, ...) {
  pred_res <- forecast_bvar(object, n_ahead, n_iter)
  # Point forecasting (Posterior mean)--------------
  pred_mean <- pred_res$posterior_mean
  var_names <- colnames(object$y0)
  colnames(pred_mean) <- var_names
  # Predictive distribution-------------------------
  dim_data <- ncol(pred_mean)
  y_distn <- 
    pred_res$predictive %>% 
    array(dim = c(n_ahead, dim_data, n_iter)) # 3d array: h x m x B
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  # Standard error----------------------------------
  est_se <- apply(y_distn, c(1, 2), sd)
  colnames(est_se) <- var_names
  # result------------------------------------------
  res <- list(
    process = object$process,
    forecast = pred_mean,
    se = est_se,
    lower = lower_quantile,
    upper = upper_quantile,
    lower_joint = lower_quantile,
    upper_joint = upper_quantile,
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
#' \deqn{y_{n + 1} \mid \Sigma_e, y \sim N( vec(y_{(n)}^T \tilde{T}^T \Phi), \Sigma_e \otimes (1 + y_{(n)}^T \tilde{T} \hat\Psi^{-1} \tilde{T} y_{(n)}) )}
#' \deqn{y_{n + 2} \mid \Sigma_e, y \sim N( vec(y_{(n + 1)}^T \tilde{T}^T \Phi), \Sigma_e \otimes (1 + y_{(n + 1)}^T \tilde{T} \hat\Psi^{-1} \tilde{T} y_{(n + 1)}) )}
#' and recursively,
#' \deqn{y_{n + h} \mid \Sigma_e, y \sim N( vec(y_{(n + h - 1)}^T \tilde{T}^T \Phi), \Sigma_e \otimes (1 + y_{(n + h - 1)}^T \tilde{T} \hat\Psi^{-1} \tilde{T} y_{(n + h - 1)}) )}
#' 
#' See [bvar_predictive_density] how to generate the predictive distribution.
#' 
#' @importFrom stats quantile
#' @order 1
#' @export
predict.bvharmn <- function(object, n_ahead, n_iter = 100L, level = .05, ...) {
  pred_res <- forecast_bvharmn(object, n_ahead, n_iter)
  # Point forecasting (Posterior mean)--------------
  pred_mean <- pred_res$posterior_mean
  var_names <- colnames(object$y0)
  colnames(pred_mean) <- var_names
  # Predictive distribution-------------------------
  dim_data <- ncol(pred_mean)
  y_distn <- 
    pred_res$predictive %>% 
    array(dim = c(n_ahead, dim_data, n_iter)) # 3d array: h x m x B
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  # Standard error----------------------------------
  est_se <- apply(y_distn, c(1, 2), sd)
  colnames(est_se) <- var_names
  # result------------------------------------------
  res <- list(
    process = object$process,
    forecast = pred_mean,
    se = est_se,
    lower = lower_quantile,
    upper = upper_quantile,
    lower_joint = lower_quantile,
    upper_joint = upper_quantile,
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
#' @importFrom stats quantile
#' @order 1
#' @export
predict.bvarflat <- function(object, n_ahead, n_iter = 100L, level = .05, ...) {
  pred_res <- forecast_bvar(object, n_ahead, n_iter)
  # Point forecasting (Posterior mean)--------------
  pred_mean <- pred_res$posterior_mean
  var_names <- colnames(object$y0)
  colnames(pred_mean) <- var_names
  # Predictive distribution-------------------------
  dim_data <- ncol(pred_mean)
  y_distn <- 
    pred_res$predictive %>% 
    array(dim = c(n_ahead, dim_data, n_iter)) # 3d array: h x m x B
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  # Standard error----------------------------------
  est_se <- apply(y_distn, c(1, 2), sd)
  colnames(est_se) <- var_names
  # result------------------------------------------
  res <- list(
    process = object$process,
    forecast = pred_mean,
    se = est_se,
    lower = lower_quantile,
    upper = upper_quantile,
    lower_joint = lower_quantile,
    upper_joint = upper_quantile,
    y = object$y
  )
  class(res) <- "predbvhar"
  res
}
