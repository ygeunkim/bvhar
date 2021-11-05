#' @rdname summary.bvharmod
#' @param x `summary.bvharmod` object
#' @param digits digit option to print
#' @param ... not used
#' @importFrom utils str
#' @order 2
#' @export
print.summary.bvharmod <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # Model description----------------
  process_type <- ifelse(x$spec$process == "BVAR", "BVAR", "BVHAR")
  prior_type <- switch(
    process_type,
    "BVAR" = {
      x$spec$prior
    },
    "BVHAR" = {
      gsub(pattern = ".*_", replacement = "", x = x$spec$prior)
    }
  )
  if (process_type == "BVAR") {
    cat(sprintf("BVAR(%i) with %s Prior\n", x$p, x$spec$prior))
  } else {
    cat(sprintf("BVHAR with %s-type Minnesota Prior\n", prior_type))
  }
  cat("====================================================\n")
  cat("Phi ~ Matrix Normal (Mean, Precision, Scale = Sigma)\n")
  cat("Sigma ~ Inverse-Wishart (IW Scale, IW df)\n")
  # density--------------------------------
  cat("\n\nAbout the Posterior Density:\n")
  cat("====================================================\n")
  cat("Number of iteration:\n")
  print.default(
    x$N,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\nCoefficients (A):\n")
  cat(
    utils::capture.output(str(x$coefficients))[1:5],
    sep = "\n"
  )
  cat("\nCovariance Matrix (Sigma):\n")
  cat(
    utils::capture.output(str(x$covmat))[1:5],
    sep = "\n"
  )
  invisible(x)
}

#' @rdname summary.bvharmod
#' @param x `summary.bvharmod` object
#' @param ... not used
#' @order 3
#' @export
knit_print.summary.bvharmn <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "summary.bvharmn",
  knit_print.summary.bvharmn,
  envir = asNamespace("knitr")
)
