#' @rdname var_lm
#' @param x `varlse` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.varlse <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # split the matrix for the print: B1, ..., Bp
  bhat_mat <- split_coef(x)
  cat(sprintf("VAR(%i) Estimation using least squares\n", x$p))
  cat("====================================================\n\n")
  for (i in 1:(x$p)) {
    cat(sprintf("LSE for A%i:\n", i))
    # B1, ..., Bp--------------------
    print.default(
      bhat_mat[[i]],
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n\n")
  }
  # const term----------------------
  if (x$type == "const") {
    intercept <- x$coefficients[x$df,]
    cat("LSE for constant:\n")
    print.default(
      intercept,
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n\n")
  }
  cat("--------------------------------------------------\n")
  cat("*_j of the Coefficient matrix: corresponding to the j-th VAR lag\n\n")
  invisible(x)
}

#' @rdname var_lm
#' @param x \code{varlse} object
#' @param ... not used
#' @order 3
#' @export
knit_print.varlse <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "varlse",
  knit_print.varlse,
  envir = asNamespace("knitr")
)

#' @rdname summary.varlse
#' @param x \code{summary.varlse} object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.summary.varlse <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # for VAR(p)--------------------------------------------------------------------
  cat(sprintf("VAR(p = %i)\n", x$p))
  cat("====================================================\n")
  # variables--------------------------
  cat("Variables: ")
  cat(paste(x$names, collapse = ", "))
  if (x$type == "const") {
    cat("\nwith const added\n")
  }
  cat(
    paste("Number of variables:", "m =", length(x$names))
  )
  cat("\n")
  # obs num----------------------------
  cat(
    paste("Observation size:", "n =", x$totobs)
  )
  cat("\n")
  cat(
    paste("Number of sample used for fitting:", "s = n - p =", x$obs)
  )
  # stability--------------------------
  cat("\n\nCharacteristic polynomial roots:\n")
  print(x$roots)
  cat(
    paste("The process is", ifelse(x$is_stable, "stable ***", "not stable"))
  )
  cat("\n====================================================\n")
  for (i in 1:(x$p)) {
    cat(sprintf("LSE for A%i:\n", i))
    # print Bi----------------------
    print.default(
      x$coefficients[[i]],
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n\n")
  }
  if (x$type == "const") {
    cat("LSE for constant:\n")
    # print c-------------------------
    print.default(
      x$coefficients$intercept,
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n====================================================\n")
  }
  # cov and corr-----------------------------
  cat("LS Estimate for Covariance matrix:\n")
  print(x$covmat)
  cat("\nLS Estimate for Correlation matrix:\n")
  print(x$corrmat)
  # information criteria--------------------
  cat("\n====================================================\n")
  cat("log-likelihood:\n")
  print(x$log_lik)
  cat("Information criteria:\n")
  print(x$ic)
}

#' @rdname summary.varlse
#' @param x \code{summary.varlse} object
#' @param ... not used
#' @order 3
#' @export
knit_print.summary.varlse <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "summary.varlse",
  knit_print.summary.varlse,
  envir = asNamespace("knitr")
)
