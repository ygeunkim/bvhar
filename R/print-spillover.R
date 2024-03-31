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
#' @exportS3Method knitr::knit_print
knit_print.bvharspillover <- function(x, ...) {
  print(x)
}

#' @rdname dynamic_spillover
#' @param x `bvhardynsp` object
#' @param digits digit option to print
#' @param ... not used
#' @importFrom utils head
#' @order 2
#' @export
print.bvhardynsp <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Dynamics of spillover effect:\n")
  cat(sprintf("Forecast using %s\n", x$process))
  cat(sprintf("Forecast step: %d\n", x$ahead))
  cat("========================\n")
  cat("Total spillovers:\n")
  cat(sprintf("# A vector: %d\n", length(x$tot)))
  print(
    head(x$tot),
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("------------------------\n")
  # cat("Directional spillovers:\n")
  # print(
  #   x$directional,
  #   digits = digits,
  #   print.gap = 2L,
  #   quote = FALSE
  # )
  cat("To spillovers:\n")
  print(
    x$to,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("------------------------\n")
  cat("From spillovers:\n")
  print(
    x$from,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("------------------------\n")
  cat("Net spillovers:\n")
  print(
    x$net,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname dynamic_spillover
#' @exportS3Method knitr::knit_print
knit_print.bvhardynsp <- function(x, ...) {
  print(x)
}
