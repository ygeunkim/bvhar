#' Predict Method for \code{varlse} object
#' 
#' Forecasting VAR(p)
#' 
#' @param object \code{varlse} object
#' @param n.ahead step to forecast
#' @param ... not used
#' @details 
#' n-step ahead forecasting using VAR(p) recursively
#' @return matrix
#' 
#' @importFrom data.table last
#' @export
predict.varlse <- function(object, n.ahead, ...) {
  # input required-------------------------------
  Y0 <- object$y0
  bhat <- object$coefficients
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
  # input required-------------------------------
  Y0 <- object$y0
  phihat <- object$coefficients
  m <- object$m
  HARtrans <- scale_har(m)
  # p <- object$p
  # k <- m * p + 1
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
