#' @rdname bvar_flat
#' @param x \code{bvarflat} object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvarflat <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # split the matrix for the print: B1, ..., Bp
  bhat_mat <- switch(
    x$type,
    "const" = {
      split.data.frame(x$coefficients[-(x$m * x$p + 1),], gl(x$p, x$m)) %>% 
        lapply(t)
    },
    "none" = {
      split.data.frame(x$coefficients, gl(x$p, x$m)) %>% 
        lapply(t)
    }
  )
  cat(sprintf("BVAR(%i) with Simple Ghosh Prior\n", x$p))
  cat("====================================================\n\n")
  cat("B ~ Matrix Normal (Mean, U^{-1}, Scale 2 = Sigma)\n")
  cat("====================================================\n")
  for (i in 1:(x$p)) {
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
  # const term-----------------------
  if (x$type == "const") {
    intercept <- x$coefficients[x$m * x$p + 1,]
    cat("Matrix Normal Mean for constant part:\n")
    print.default(
      intercept,
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n\n")
  }
  # scale matrix-------------------
  cat("dim(Matrix Normal precision matrix):\n")
  print.default(
    dim(x$mn_prec),
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

#' @rdname bvar_flat
#' @param x \code{bvarflat} object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvarflat <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvarflat",
  knit_print.bvarflat,
  envir = asNamespace("knitr")
)