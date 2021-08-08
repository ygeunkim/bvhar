#' Predict Method for \code{varlse} object
#' 
#' Forecasting VAR(p)
#' 
#' @param object \code{varlse} object
#' @param n.ahead step to forecast
#' @param ... not used
#' @details 
#' n-step ahead forecasting using VAR(p) recursively.
#' See pp35 of Lütkepohl (2007).
#' 
#' @return matrix
#' 
#' @references 
#' Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
#' @order 1
#' @export
predict.varlse <- function(object, n.ahead, ...) {
  res <- forecast_var(object, n.ahead)
  colnames(res) <- colnames(object$y0)
  class(res) <- c("matrix", "array", "predvarlse")
  res
}

# predict.varlse <- function(object, n.ahead, ...) {
#   # input required-------------------------------
#   Y0 <- object$y0
#   bhat <- object$coefficients
#   m <- object$m
#   p <- object$p
#   k <- m * p + 1
#   # vectorize last p observations and add constant
#   h_vec <- t(last(Y0, n = p)[p:1,])[1:(m * p)]
#   h_vec <- c(h_vec, 1)
#   # forecasting (point)---------------------------
#   # y(n + 1)^T = [y(n)^T, ..., y(n - p + 1)^T, 1] %*% Bhat
#   res <- h_vec %*% bhat
#   if (n.ahead == 1) return(res)
#   # recursively----------------------
#   for (i in 1:(n.ahead - 1)) {
#     # y(n + 2)^T = [yhat(n + 1)^T, y(n)^T, ... y(n - p + 2)^T, 1] %*% Bhat
#     h_vec <- c(c(last(res)), h_vec[-c((k - m):(k - 1))])
#     res <- rbind(res, h_vec %*% bhat)
#   }
#   class(res) <- c("matrix", "array", "predvarlse")
#   res
# }

#' Forecast Region for VAR(p)
#' 
#' Compute forecast region VAR(p) model
#' @param object predvarlse object
#' @param level Confidence level
#' @param ... not used
#' 
#' @details 
#' For h-step forecasting (e.g. n + 1, ... n + h),
#' joint forecast region of \eqn{100(1-\alpha)}\% can be computed by
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
#' @references 
#' Lütkepohl, H. (2007). \emph{New Introduction to Multiple Time Series Analysis}. Springer Publishing. \url{https://doi.org/10.1007/978-3-540-27752-1}
#' @order 2
#' @export
forecast_region <- function(object, h, level = .05, ...) {
  h <- nrow(object) # h-step
  z <- qnorm(level / (2 * h), lower.tail = FALSE) # + z(alpha / 2h)
  z
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
#' 
#' @importFrom data.table last
#' @export
predict.vharlse <- function(object, n.ahead, ...) {
  res <- forecast_var(object, n.ahead)
  colnames(res) <- colnames(object$y0)
  class(res) <- c("matrix", "array", "predvarlse")
  res
}

# predict.vharlse <- function(object, n.ahead, ...) {
#   # input required-------------------------------
#   Y0 <- object$y0
#   phihat <- object$coefficients
#   m <- object$m
#   HARtrans <- scale_har(m)
#   # p <- object$p
#   # k <- m * p + 1
#   # vectorize last p observations and add constant
#   h_vec <- t(last(Y0, n = 22)[22:1,])[1:(m * 22)]
#   h_vec <- c(h_vec, 1)
#   # forecasting (point)---------------------------
#   # y(n + 1)^T = [y(n)^T, ..., y(n - p + 1)^T, 1] %*% t(HARtrans) %*% Phihat
#   res <- h_vec %*% t(HARtrans) %*% phihat
#   if (n.ahead == 1) return(res)
#   # recursively----------------------
#   for (i in 1:(n.ahead - 1)) {
#     # y(n + 2)^T = [yhat(n + 1)^T, y(n)^T, ... y(n - p + 2)^T, 1] %*% t(HARtrans) %*% Phihat
#     # remove the last m observation (except the last 1)
#     h_vec <- c(c(last(res)), h_vec[-c((21 * m + 1):(22 * m))])
#     res <- rbind(res, h_vec %*% t(HARtrans) %*% phihat)
#   }
#   res
# }

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
#' @importFrom data.table last
#' @export
predict.bvarmn <- function(object, n.ahead, ...) {
  # input required-------------------------------
  Y0 <- object$y0
  bhat <- object$mn_mean
  m <- object$m
  p <- object$p
  k <- m * p + 1
  # vectorize last p observations and add constant
  h_vec <- t(last(Y0, n = p)[p:1,])[1:(m * p)]
  h_vec <- c(h_vec, 1)
  # forecasting (point)---------------------------
  # y(n + 1)^T = [y(n)^T, ..., y(n - p + 1)^T, 1] %*% Bhat
  res <- h_vec %*% bhat
  if (n.ahead == 1) return(res)
  # recursively----------------------
  for (i in 1:(n.ahead - 1)) {
    # y(n + 2)^T = [yhat(n + 1)^T, y(n)^T, ... y(n - p + 2)^T, 1] %*% Bhat
    h_vec <- c(c(last(res)), h_vec[-c((k - m):(k - 1))])
    res <- rbind(res, h_vec %*% bhat)
  }
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
#' @importFrom data.table last
#' @export
predict.bvharmn <- function(object, n.ahead, ...) {
  # input required-------------------------------
  Y0 <- object$y0
  phihat <- object$mn_mean
  m <- object$m
  HARtrans <- scale_har(m)
  # vectorize last p observations and add constant
  h_vec <- t(last(Y0, n = 22)[22:1,])[1:(m * 22)]
  h_vec <- c(h_vec, 1)
  # forecasting (point)---------------------------
  # y(n + 1)^T = [y(n)^T, ..., y(n - p + 1)^T, 1] %*% t(HARtrans) %*% Phihat
  res <- h_vec %*% t(HARtrans) %*% phihat
  if (n.ahead == 1) return(res)
  # recursively----------------------
  for (i in 1:(n.ahead - 1)) {
    # y(n + 2)^T = [yhat(n + 1)^T, y(n)^T, ... y(n - p + 2)^T, 1] %*% t(HARtrans) %*% Phihat
    # remove the last m observation (except the last 1)
    h_vec <- c(c(last(res)), h_vec[-c((21 * m + 1):(22 * m))])
    res <- rbind(res, h_vec %*% t(HARtrans) %*% phihat)
  }
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
#' @importFrom data.table last
#' @export
predict.bvarghosh <- function(object, n.ahead, ...) {
  # input required-------------------------------
  Y0 <- object$y0
  bhat <- object$mn_mean
  m <- object$m
  p <- object$p
  k <- m * p + 1
  # vectorize last p observations and add constant
  h_vec <- t(last(Y0, n = p)[p:1,])[1:(m * p)]
  h_vec <- c(h_vec, 1)
  # forecasting (point)---------------------------
  # y(n + 1)^T = [y(n)^T, ..., y(n - p + 1)^T, 1] %*% Bhat
  res <- h_vec %*% bhat
  if (n.ahead == 1) return(res)
  # recursively----------------------
  for (i in 1:(n.ahead - 1)) {
    # y(n + 2)^T = [yhat(n + 1)^T, y(n)^T, ... y(n - p + 2)^T, 1] %*% Bhat
    h_vec <- c(c(last(res)), h_vec[-c((k - m):(k - 1))])
    res <- rbind(res, h_vec %*% bhat)
  }
  res
}
