#' @rdname choose_bvar
#' @param x `bvharemp` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharemp <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  print(x$spec)
  invisible(x)
}

#' @rdname choose_bvar
#' @param x `bvharemp` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharemp <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvharemp",
  knit_print.bvharemp,
  envir = asNamespace("knitr")
)
