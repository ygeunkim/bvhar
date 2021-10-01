#' @rdname vhar_lm
#' @param x \code{vharlse} object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.vharlse <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # split the matrix for the print: Phi(d), Phi(w), Phi(m)
  # phihat_mat <- switch(
  #   x$type,
  #   "const" = {
  #     split.data.frame(x$coefficients[-(3 * x$m + 1),], gl(3, x$m)) %>% 
  #       lapply(t)
  #   },
  #   "none" = {
  #     split.data.frame(x$coefficients, gl(3, x$m)) %>% 
  #       lapply(t)
  #   }
  # )
  phihat_mat <- split_coef(x)
  names(phihat_mat) <- c("day", "week", "month")
  cat("VHAR Estimation")
  cat("====================================================\n\n")
  for (i in 1:x$p) {
    cat(paste0("LSE for ", names(phihat_mat)[i], ":\n"))
    # Phi(d), Phi(w), Phi(m)---------
    print.default(
      phihat_mat[[i]],
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
  cat("*_day, *_week, *_month of the Coefficient matrix: daily, weekly, and monthly term in the VHAR model\n\n")
  invisible(x)
}

#' @rdname vhar_lm
#' @param x \code{vharlse} object
#' @param ... not used
#' @order 3
#' @export
knit_print.vharlse <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "vharlse",
  knit_print.vharlse,
  envir = asNamespace("knitr")
)

#' @rdname summary.vharlse
#' @param x `summary.vharlse` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.summary.vharlse <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # for VAR(p)--------------------------------------------------------------------
  cat("VHAR Estimation\n")
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
  cat("\n====================================================\n")
  for (i in 1:3) {
    cat(paste0("LSE for ", names(x$coefficients)[i], ":\n"))
    # print Phi-----------------------
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

#' @rdname summary.vharlse
#' @param x `summary.vharlse` object
#' @param ... not used
#' @order 3
#' @export
knit_print.summary.vharlse <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "summary.vharlse",
  knit_print.summary.vharlse,
  envir = asNamespace("knitr")
)

