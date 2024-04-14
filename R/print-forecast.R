#' Print Method for `predbvhar` object
#' @rdname predict
#' @param x `predbvhar` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.predbvhar <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  print(x$forecast)
  invisible(x)
}

#' @rdname predict
#' @exportS3Method knitr::knit_print
knit_print.predbvhar <- function(x, ...) {
  print(x)
}

#' @rdname forecast_roll
#' @param x `bvharcv` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharcv <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  print(x$forecast)
  invisible(x)
}

#' @rdname forecast_roll
#' @exportS3Method knitr::knit_print
knit_print.bvharcv <- function(x, ...) {
  print(x)
}
