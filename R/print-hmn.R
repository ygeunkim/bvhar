#' @rdname bvar_minnesota
#' @param x `hmnmod` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.hmnmod <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep = "\n", collapse = "\n"), "\n\n",
    sep = ""
  )
  if (x$spec$process == "BVAR") {
    cat(sprintf("BVAR(%i) with Hierarchical Prior\n", x$p))
  } else {
    cat("BVHAR with Hierarchical Prior\n")
  }
  cat("Fitted by Metropolis algorithm\n")
  cat(paste0("Total number of iteration: ", x$iter, "\n"))
  cat(paste0("Number of burn-in: ", x$burn, "\n"))
  if (x$thin > 1) {
    cat(paste0("Thinning: ", x$thin, "\n"))
  }
  cat("====================================================\n\n")
  cat("Hyperparameter Selection:\n")
  print(
    x$hyperparam,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\n--------------------------------------------------\n")
  cat("Coefficients ~ Matrix Normal Record:\n")
  print(
    x$alpha_record,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\nSigma ~ Inverse-Wishart Record:\n")
  print(
    x$sigma_record,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname bvar_minnesota
#' @exportS3Method knitr::knit_print
knit_print.hmnmod <- function(x, ...) {
  print(x)
}
