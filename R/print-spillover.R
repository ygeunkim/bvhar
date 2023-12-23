#' @rdname spillover
#' @param x `bvharspillover` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharspillover <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(paste0(
    "Variance shares as the fractions of n_ahead error variances in forecasting* using ",
    x$process,
    ":\n"
  ))
  # cat("shocks (i) -> variables (j)\n")
  cat("variables (i) <- shocks (j)\n")
  cat("========================\n")
  print(
    x$connect,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("------------------------\n")
  cat("*Own vairance shares (diagonal): error caused by i-th shocks when forecasting i-th variable\n")
  cat("*Cross variance shares = Spillovers (ij-th element): error caused by j-th shocks when forecasting i-th variable")
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

#' @rdname summary.bvharspillover
#' @param x `summary.bvharspillover` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.summary.bvharspillover <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Directional spillovers:\n")
  # cat("shocks (i) -> variables (j)\n")
  cat("variables (i) <- shocks (j)\n")
  cat("========================\n")
  print(
    # cbind(x$connect, "net" = c(x$net, NA)),
    x$connect,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\n*Lower right corner: Total spillover\n")
  cat("------------------------\n")
  cat("Net spillovers:\n")
  print(
    # cbind(x$connect, "net" = c(x$net, NA)),
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