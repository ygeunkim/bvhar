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
#' 
#' Confidence interval at h-period is
#' \deqn{y_{k,t}(h) \pm z_(\alpha / 2) \sigma_k (h)}
#' 
#' Joint forecast region of \eqn{100(1-\alpha)}\% can be computed by
#' \deqn{\left\{ (y_{k, 1}, y_{k, h}) \mid y_{k, n}(i) - z_{(\alpha / 2h)} \sigma_n(i) \le y_{n, i} \le y_{k, n}(i) + z_{(\alpha / 2h)} \sigma_k(i), i = 1, \ldots, h \right\}}
#' See the pp41 of Lütkepohl (2007).
#' 
#' To compute covariance matrix, it needs VMA representation:
#' \deqn{Y_{t}(h) = c + \sum_{i = h}^{\infty} W_{i} \epsilon_{t + h - i} = c + \sum_{i = 0}^{\infty} W_{h + i} \epsilon_{t - i}}
#' 
#' Then
#' 
#' \deqn{\Sigma_y(h) = MSE \left[ y_t(h) \right] = \sum_{i = 0}^{h - 1} W_i \Sigma_{\epsilon} W_i^T = \Sigma_y(h - 1) + W_{h - 1} \Sigma_{\epsilon} W_{h - 1}^T}
#' 
#' @return matrix
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
#' @param ... not used
#' @details 
#' n-step ahead forecasting using VHAR recursively
#' @return matrix
#' @order 1
#' @export
predict.vharlse <- function(object, n.ahead, ...) {
  pred_res <- forecast_vhar(object, n.ahead)
  colnames(pred_res) <- colnames(object$y0)
  res <- list(
    process = "vharlse",
    forecast = pred_res,
    y = object$y
  )
  class(res) <- "predbvhar"
  res
}

#' Predict Method for \code{bvarmn} object
#' 
#' Point forecasting for Minnesota BVAR
#' 
#' @param object \code{bvarmn} object
#' @param n.ahead step to forecast
#' @param ... not used
#' 
#' @details 
#' n-step ahead forecasting using BVAR recursively
#' @return matrix
#' 
#' @order 1
#' @export
predict.bvarmn <- function(object, n.ahead, ...) {
  pred_res <- forecast_bvarmn(object, n.ahead)
  colnames(pred_res) <- colnames(object$y0)
  res <- list(
    process = "bvarmn",
    forecast = pred_res,
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
#' @param ... not used
#' @details 
#' n-step ahead forecasting using VHAR recursively
#' @return matrix
#' 
#' @order 1
#' @export
predict.bvharmn <- function(object, n.ahead, ...) {
  pred_res <- forecast_bvharmn(object, n.ahead)
  colnames(pred_res) <- colnames(object$y0)
  res <- list(
    process = "bvharmn",
    forecast = pred_res,
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
#' @param ... not used
#' 
#' @details 
#' n-step ahead forecasting using BVAR recursively
#' @return matrix
#' 
#' @export
predict.bvarghosh <- function(object, n.ahead, ...) {
  pred_res <- forecast_bvarghosh(object, n.ahead)
  colnames(pred_res) <- colnames(object$y0)
  res <- list(
    process = "bvarghosh",
    forecast = pred_res,
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

