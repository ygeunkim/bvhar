#' Print Method for `predbvhar` object
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
#' @param x `predbvhar` object
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
#' @param x `bvharcv` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharcv <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvharcv",
  knit_print.bvharcv,
  envir = asNamespace("knitr")
)
