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
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
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
    t() # extract only diagonal element to compute CIs
  SE <- sqrt(SE)
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
#' Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174–196.
#' 
#' Baek, C. and Park, M. (2021). *Sparse vector heterogeneous autoregressive modeling for realized volatility*. J. Korean Stat. Soc. 50, 495–510.
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
    t() # extract only diagonal element to compute CIs
  SE <- sqrt(SE)
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
#' @param object Model object
#' @param n_ahead step to forecast
#' @param n_iter Number to sample residual matrix from inverse-wishart distribution. By default, 100.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
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
#' @references 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1).
#' 
#' Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). *Bayesian data analysis*. Chapman and Hall/CRC.
#' 
#' Karlsson, S. (2013). *Chapter 15 Forecasting with Bayesian Vector Autoregression*. Handbook of Economic Forecasting, 2, 791–897.
#' 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25.
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
    # forecast = apply(y_distn, c(1, 2), mean),
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
    # forecast = apply(y_distn, c(1, 2), mean),
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
#' @references Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526).
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

#' @rdname predict.varlse
#' 
#' @param object Model object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' @section n-step ahead forecasting VAR(p) with SSVS and Horseshoe:
#' The process of the computing point estimate is the same.
#' However, predictive interval is achieved from each Gibbs sampler sample.
#' 
#' \deqn{y_{n + 1} \mid A, \Sigma_e, y \sim N( vec(y_{(n)}^T A), \Sigma_e )}
#' \deqn{y_{n + h} \mid A, \Sigma_e, y \sim N( vec(\hat{y}_{(n + h - 1)}^T A), \Sigma_e )}
#' @references George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580.
#' @importFrom posterior as_draws_matrix
#' @importFrom stats quantile
#' @order 1
#' @export
predict.bvarssvs <- function(object, n_ahead, level = .05, ...) {
  num_chains <- object$chain
  pred_res <- forecast_bvarssvs(
    num_chains,
    object$p,
    n_ahead,
    object$y0,
    object$df,
    as_draws_matrix(object$alpha_record),
    as_draws_matrix(object$eta_record),
    as_draws_matrix(object$psi_record)
  )
  dim_data <- object$m
  var_names <- colnames(object$y0)
  # Predictive distribution------------------------------------
  num_step <- nrow(object$alpha_record) / num_chains
  y_distn <-
    pred_res %>% 
    array(dim = c(n_ahead * num_chains, dim_data, num_step))
  pred_mean <- apply(y_distn, c(1, 2), mean)
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  est_se <- apply(y_distn, c(1, 2), sd)
  if (num_chains > 1) {
    pred_mean <- split.data.frame(pred_mean, gl(num_chains, n_ahead))
    pred_mean <- Reduce("+", pred_mean) / num_chains
    lower_quantile <- split.data.frame(lower_quantile, gl(num_chains, n_ahead))
    lower_quantile <- Reduce("+", lower_quantile) / num_chains
    upper_quantile <- split.data.frame(upper_quantile, gl(num_chains, n_ahead))
    upper_quantile <- Reduce("+", upper_quantile) / num_chains
    est_se <- split.data.frame(est_se, gl(num_chains, n_ahead))
    est_se <- Reduce("+", est_se) / num_chains
  }
  colnames(pred_mean) <- var_names
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  colnames(est_se) <- var_names
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
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' @section n-step ahead forecasting VHAR with SSVS and Horseshoe:
#' The process of the computing point estimate is the same.
#' However, predictive interval is achieved from each Gibbs sampler sample.
#' 
#' \deqn{y_{n + 1} \mid \Sigma_e, y \sim N( vec(y_{(n)}^T \tilde{T}^T \Phi), \Sigma_e \otimes (1 + y_{(n)}^T \tilde{T} \hat\Psi^{-1} \tilde{T} y_{(n)}) )}
#' \deqn{y_{n + h} \mid \Sigma_e, y \sim N( vec(y_{(n + h - 1)}^T \tilde{T}^T \Phi), \Sigma_e \otimes (1 + y_{(n + h - 1)}^T \tilde{T} \hat\Psi^{-1} \tilde{T} y_{(n + h - 1)}) )}
#' 
#' @references George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580.
#' @importFrom posterior as_draws_matrix
#' @importFrom stats quantile
#' @order 1
#' @export
predict.bvharssvs <- function(object, n_ahead, level = .05, ...) {
  num_chains <- object$chain
  pred_res <- forecast_bvharssvs(
    num_chains,
    object$month,
    n_ahead,
    object$y0,
    object$HARtrans,
    as_draws_matrix(object$phi_record),
    as_draws_matrix(object$eta_record),
    as_draws_matrix(object$psi_record)
  )
  dim_data <- object$m
  var_names <- colnames(object$y0)
  # Predictive distribution------------------------------------
  num_step <- nrow(object$phi_record) / num_chains
  y_distn <-
    pred_res %>%
    # pred_res$predictive %>%
    array(dim = c(n_ahead * num_chains, dim_data, num_step))
  pred_mean <- apply(y_distn, c(1, 2), mean)
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  est_se <- apply(y_distn, c(1, 2), sd)
  if (num_chains > 1) {
    pred_mean <- split.data.frame(pred_mean, gl(num_chains, n_ahead))
    pred_mean <- Reduce("+", pred_mean) / num_chains
    lower_quantile <- split.data.frame(lower_quantile, gl(num_chains, n_ahead))
    lower_quantile <- Reduce("+", lower_quantile) / num_chains
    upper_quantile <- split.data.frame(upper_quantile, gl(num_chains, n_ahead))
    upper_quantile <- Reduce("+", upper_quantile) / num_chains
    est_se <- split.data.frame(est_se, gl(num_chains, n_ahead))
    est_se <- Reduce("+", est_se) / num_chains
  }
  colnames(pred_mean) <- var_names
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  colnames(est_se) <- var_names
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
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' @importFrom posterior as_draws_matrix
#' @importFrom stats quantile
#' @order 1
#' @export
predict.bvarhs <- function(object, n_ahead, level = .05, ...) {
  num_chains <- object$chain
  pred_res <- forecast_bvarhs(
    num_chains,
    object$p,
    n_ahead,
    object$y0,
    object$df,
    as_draws_matrix(object$alpha_record),
    as.numeric(as_draws_matrix(object$sigma_record))
  )
  dim_data <- object$m
  var_names <- colnames(object$y0)
  # Predictive distribution------------------------------------
  num_step <- nrow(object$alpha_record) / num_chains
  y_distn <-
    pred_res %>%
    array(dim = c(n_ahead * num_chains, dim_data, num_step))
  pred_mean <- apply(y_distn, c(1, 2), mean)
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  est_se <- apply(y_distn, c(1, 2), sd)
  if (num_chains > 1) {
    pred_mean <- split.data.frame(pred_mean, gl(num_chains, n_ahead))
    pred_mean <- Reduce("+", pred_mean) / num_chains
    lower_quantile <- split.data.frame(lower_quantile, gl(num_chains, n_ahead))
    lower_quantile <- Reduce("+", lower_quantile) / num_chains
    upper_quantile <- split.data.frame(upper_quantile, gl(num_chains, n_ahead))
    upper_quantile <- Reduce("+", upper_quantile) / num_chains
    est_se <- split.data.frame(est_se, gl(num_chains, n_ahead))
    est_se <- Reduce("+", est_se) / num_chains
  }
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  colnames(pred_mean) <- var_names
  colnames(est_se) <- var_names
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
  class(res) <- c("predsp", "predbvhar")
  res
}

#' @rdname predict.varlse
#' 
#' @param object Model object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' @importFrom posterior as_draws_matrix
#' @importFrom stats quantile
#' @order 1
#' @export
predict.bvharhs <- function(object, n_ahead, level = .05, ...) {
  num_chains <- object$chain
  pred_res <- forecast_bvharhs(
    num_chains,
    object$month,
    n_ahead,
    object$y0,
    object$HARtrans,
    as_draws_matrix(object$phi_record),
    as.numeric(as_draws_matrix(object$sigma_record))
  )
  dim_data <- object$m
  var_names <- colnames(object$y0)
  # Predictive distribution------------------------------------
  num_step <- nrow(object$phi_record) / num_chains
  y_distn <-
    pred_res %>% 
    # pred_res$predictive %>%
    array(dim = c(n_ahead * num_chains, dim_data, num_step))
  pred_mean <- apply(y_distn, c(1, 2), mean)
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  est_se <- apply(y_distn, c(1, 2), sd)
  if (num_chains > 1) {
    pred_mean <- split.data.frame(pred_mean, gl(num_chains, n_ahead))
    pred_mean <- Reduce("+", pred_mean) / num_chains
    lower_quantile <- split.data.frame(lower_quantile, gl(num_chains, n_ahead))
    lower_quantile <- Reduce("+", lower_quantile) / num_chains
    upper_quantile <- split.data.frame(upper_quantile, gl(num_chains, n_ahead))
    upper_quantile <- Reduce("+", upper_quantile) / num_chains
    est_se <- split.data.frame(est_se, gl(num_chains, n_ahead))
    est_se <- Reduce("+", est_se) / num_chains
  }
  colnames(pred_mean) <- var_names
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  colnames(est_se) <- var_names
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
  class(res) <- c("predsp", "predbvhar")
  res
}

#' @rdname predict.varlse
#' @param object Model object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' @importFrom posterior as_draws_matrix
#' @order 1
#' @export
predict.bvarsv <- function(object, n_ahead, level = .05, ...) {
  dim_data <- object$m
  num_chains <- object$chain
  alpha_record <- as_draws_matrix(object$alpha_record)
  is_stable <- apply(
    alpha_record,
    1,
    function(x) {
      all(
        matrix(x, ncol = object$m) %>%
          compute_stablemat() %>%
          eigen() %>%
          .$values %>%
          Mod() < 1
      )
    }
  )
  if (any(!is_stable)) {
    warning("Some alpha records are unstable, so add burn-in")
  }
  if (object$type == "const") {
    alpha_record <- cbind(alpha_record, as_draws_matrix(object$c_record))
  }
  pred_res <- forecast_bvarsv(
    num_chains,
    object$p,
    n_ahead,
    object$y0,
    alpha_record,
    as_draws_matrix(object$h_record),
    as_draws_matrix(object$a_record),
    as_draws_matrix(object$sigh_record),
    sample.int(.Machine$integer.max, size = num_chains),
    object$type == "const"
  )
  var_names <- colnames(object$y0)
  # Predictive distribution------------------------------------
  num_draw <- nrow(object$alpha_record) # concatenate multiple chains
  y_distn <-
    pred_res %>% 
    unlist() %>% 
    array(dim = c(n_ahead, dim_data, num_draw))
  pred_mean <- apply(y_distn, c(1, 2), mean)
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  est_se <- apply(y_distn, c(1, 2), sd)
  colnames(pred_mean) <- var_names
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  colnames(est_se) <- var_names
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
  res$object <- object
  class(res) <- c("predsv", "predbvhar")
  res
}

#' @rdname predict.varlse
#' @param object Model object
#' @param n_ahead step to forecast
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param ... not used
#' @importFrom posterior as_draws_matrix
#' @order 1
#' @export
predict.bvharsv <- function(object, n_ahead, level = .05, ...) {
  dim_data <- object$m
  num_chains <- object$chain
  phi_record <- as_draws_matrix(object$phi_record)
  is_stable <- apply(
    phi_record,
    1,
    function(x) {
      coef <- t(object$HARtrans[1:(object$p * dim_data), 1:(object$month * dim_data)]) %*% matrix(x, ncol = object$m)
      all(
        coef %>% 
          compute_stablemat() %>% 
          eigen() %>% 
          .$values %>% 
          Mod() < 1
      )
    }
  )
  if (any(!is_stable)) {
    warning("Some phi records are unstable, so add burn-in")
  }
  if (object$type == "const") {
    phi_record <- cbind(phi_record, as_draws_matrix(object$c_record))
  }
  pred_res <- forecast_bvharsv(
    num_chains,
    object$month,
    n_ahead,
    object$y0,
    object$HARtrans,
    phi_record,
    as_draws_matrix(object$h_record),
    as_draws_matrix(object$a_record),
    as_draws_matrix(object$sigh_record),
    sample.int(.Machine$integer.max, size = num_chains),
    object$type == "const"
  )
  var_names <- colnames(object$y0)
  # Predictive distribution------------------------------------
  num_draw <- nrow(object$alpha_record) # concatenate multiple chains
  y_distn <-
    pred_res %>% 
    unlist() %>% 
    array(dim = c(n_ahead, dim_data, num_draw))
  pred_mean <- apply(y_distn, c(1, 2), mean)
  lower_quantile <- apply(y_distn, c(1, 2), quantile, probs = level / 2)
  upper_quantile <- apply(y_distn, c(1, 2), quantile, probs = (1 - level / 2))
  est_se <- apply(y_distn, c(1, 2), sd)
  colnames(pred_mean) <- var_names
  colnames(lower_quantile) <- var_names
  colnames(upper_quantile) <- var_names
  colnames(est_se) <- var_names
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
  res$object <- object
  class(res) <- c("predsv", "predbvhar")
  res
}
