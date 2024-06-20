#' @rdname summary.normaliw
#' @param x `summary.normaliw` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.summary.normaliw <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep = "\n", collapse = "\n"), "\n\n",
    sep = ""
  )
  # Model description----------------
  process_type <- ifelse(x$spec$process == "BVAR", "BVAR", "BVHAR")
  prior_type <- switch(process_type,
    "BVAR" = {
      x$spec$prior
    },
    "BVHAR" = {
      gsub(pattern = ".*_", replacement = "", x = x$spec$prior)
    }
  )
  if (process_type == "BVAR") {
    cat(sprintf("BVAR(%i) with %s Prior\n", x$p, prior_type))
  } else {
    cat(sprintf("BVHAR with %s-type Minnesota Prior\n", prior_type))
  }
  cat("====================================================\n")
  cat("Phi ~ Matrix Normal (Mean, Precision, Scale = Sigma)\n")
  cat("Sigma ~ Inverse-Wishart (IW Scale, IW df)\n")
  cat("\n\nConjugate MCMC:\n")
  cat("====================================================\n")
  cat(paste0("Total number of iteration: ", x$iter, "\n"))
  cat(paste0("Number of burn-in: ", x$burn, "\n"))
  if (x$thin > 1) {
    cat(paste0("Thinning: ", x$thin, "\n"))
  }
  cat("====================================================\n\n")
  cat("Parameter record:\n")
  print(
    x$param,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  # cat("\nCoefficients (A):\n")
  # cat(
  #   utils::capture.output(str(x$coefficients))[1:5],
  #   sep = "\n"
  # )
  # cat("\nCovariance Matrix (Sigma):\n")
  # cat(
  #   utils::capture.output(str(x$covmat))[1:5],
  #   sep = "\n"
  # )
  invisible(x)
}

#' @rdname summary.normaliw
#' @exportS3Method knitr::knit_print
knit_print.summary.normaliw <- function(x, ...) {
  print(x)
}
