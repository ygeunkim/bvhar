#' @rdname spillover
#' @param x `bvharspillover` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharspillover <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Directional spillovers:\n")
  cat("variables (i) <- shocks (j)\n")
  cat("========================\n")
  print(
    x$connect,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\n*Lower right corner: Total spillover\n")
  cat("------------------------\n")
  cat("Net spillovers:\n")
  print(
    x$net,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\nNet pairwise spillovers:\n")
  print(
    x$net_pairwise,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname spillover
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
