#' @rdname varshares
#' @param x `bvharspillover` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharspillover <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(paste0(
    "Variance shares by ",
    x$process,
    ":\n"
  ))
  cat("shocks (i) -> variables (j)\n")
  cat("========================\n")
  print(
    x$connect,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname varshares
#' @param x `bvharspillover` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharspillover <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvharspillover",
  knit_print.bvharspillover,
  envir = asNamespace("knitr")
)

#' @rdname summary.bvharspillover
#' @param x `summary.bvharspillover` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.summary.bvharspillover <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Connectedness Table:\n")
  cat("shocks (i) -> variables (j)\n")
  cat("========================\n")
  print(
    cbind(x$connect, "net" = c(x$net, NA)),
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname summary.bvharspillover
#' @param x `summary.bvharspillover` object
#' @param ... not used
#' @order 3
#' @export
knit_print.summary.bvharspillover <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "summary.bvharspillover",
  knit_print.summary.bvharspillover,
  envir = asNamespace("knitr")
)