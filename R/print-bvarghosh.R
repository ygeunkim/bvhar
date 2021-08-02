#' @rdname bvar_ghosh
#' @param x \code{bvarghosh} object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvarghosh <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # split the matrix for the print: B1, ..., Bp
  bhat_mat <- 
    split.data.frame(x$mn_mean[-(x$m * x$p + 1),], gl(x$p, x$m)) %>% 
    lapply(t)
  # const term
  intercept <- x$mn_mean[x$m * x$p + 1,]
  cat(sprintf("BVAR(%i) with Simple Ghosh Prior\n", p))
  cat("====================================================\n\n")
  cat("B ~ Matrix Normal (Mean, U^{-1}, Scale 2 = Sigma)\n")
  cat("====================================================\n")
  for (i in 1:p) {
    cat(sprintf("Matrix Normal Mean for B%i part:\n", i))
    # B1, ..., Bp--------------------
    print.default(
      bhat_mat[[i]],
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n\n")
  }
  cat("Matrix Normal Mean for constant part:\n")
  # c-------------------------------
  print.default(
    intercept,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  # scale matrix-------------------
  cat("\n\ndim(Matrix Normal first scale matrix):\n")
  print.default(
    dim(x$mn_scale),
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\n\nSigma ~ Inverse-Wishart\n")
  cat("====================================================\n")
  cat("IW scale matrix:\n")
  print.default(
    x$iw_scale,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\n\n--------------------------------------------------\n")
  cat("*_j of the Coefficient matrix: j-th observation is the first observation corresponding to the coefficient\n\n")
  invisible(x)
}

#' @rdname bvar_ghosh
#' @param x \code{bvarghosh} object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvarghosh <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvarghosh",
  knit_print.bvarghosh,
  envir = asNamespace("knitr")
)