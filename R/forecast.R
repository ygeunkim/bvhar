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
  pred_res <- forecast_var(object, n.ahead)
  colnames(pred_res) <- colnames(object$y0)
  res <- list(
    process = "varlse",
    forecast = pred_res,
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

