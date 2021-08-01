#' @rdname var_lm
#' @export
print.varlse <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # split the matrix for the print: B1, ..., Bp
  bhat_mat <- 
    split.data.frame(x$coefficients[-(x$m * x$p + 1),], gl(x$p, x$m)) %>% 
    lapply(t)
  # const term
  intercept <- x$coefficients[x$m * x$p + 1,]
  cat(sprintf("VAR(%i) Estimation using least squares\n", p))
  cat("====================================================\n\n")
  for (i in 1:p) {
    cat(sprintf("LSE for B%i:\n", i))
    # B1, ..., Bp--------------------
    print.default(
      bhat_mat[[i]],
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n\n")
  }
  cat("LSE for constant:\n")
  # c-------------------------------
  print.default(
    intercept,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\n\n--------------------------------------------------\n")
  cat("*_j of the Coefficient matrix: j-th observation is the first observation corresponding to the coefficient\n\n")
  invisible(x)
}

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
#' @export
print.summary.varlse <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # for VAR(p)--------------------------------------------------------------------
  cat(sprintf("VAR(p = %i)\n", p))
  cat("====================================================\n")
  # variables--------------------------
  cat("Variables: ")
  cat(paste(x$names, collapse = ", "))
  cat("\nwith const added\n")
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
    paste("Number of sample used for fitting:", "n - p =", x$obs)
  )
  # stability--------------------------
  cat("\n\nCharacteristic polynomial roots:\n")
  print(x$roots)
  cat(
    paste("The process is", ifelse(x$is_stable, "stable ***", "not stable"))
  )
  cat("\n====================================================\n")
  for (i in 1:p) {
    cat(sprintf("OLS for B%i:\n", i))
    # print Bi----------------------
    print.default(
      x$coefficients[[i]],
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n\n")
  }
  cat("OLS for constant:\n")
  # print c-------------------------
  print.default(
    last(x$coefficients),
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\n====================================================\n")
  # cov and corr-----------------------------
  cat("Covariance matrix of the residuals:\n")
  print(x$covmat)
  cat("\nCorrelation matrix of the residuals:\n")
  print(x$corrmat)
  # information criteria--------------------
  cat("\n====================================================\n")
  cat("Information criteria:\n")
  print(x$ic)
}

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
