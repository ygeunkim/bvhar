#' @rdname bvar_minnesota
#' @param x \code{bvarmn} object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvarmn <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
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
  cat(sprintf("BVAR(%i) with Minnesota Prior\n", x$p))
  cat("====================================================\n\n")
  cat("B ~ Matrix Normal (Mean, Precision, Scale = Sigma)\n")
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
  cat("Matrix Normal Mean for constant part:\n")
  # c-------------------------------
  print.default(
    intercept,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  # precision matrix---------------0
  cat("\n\ndim(Matrix Normal precision matrix):\n")
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
  cat("\nIW degrees of freedom:\n")
  print.default(x$iw_shape)
  cat("\n\n--------------------------------------------------\n")
  cat("*_j of the Coefficient matrix: j-th observation is the first observation corresponding to the coefficient\n\n")
  invisible(x)
}

#' @rdname bvar_minnesota
#' @param x \code{bvarmn} object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvarmn <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvarmn",
  knit_print.bvarmn,
  envir = asNamespace("knitr")
)

#' @rdname summary.bvarmn
#' @param x \code{summary.bvarmn} object
#' @param digits digit option to print
#' @param ... not used
#' @importFrom utils str
#' @order 2
#' @export
print.summary.bvarmn <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # Model description----------------
  cat(sprintf("BVAR(%i) with Minnesota Prior\n", x$p))
  cat("====================================================\n")
  cat("B ~ Matrix Normal (Mean, Precision, Scale = Sigma)\n")
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
  cat("\nCoefficients (B):\n")
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

#' @rdname summary.bvarmn
#' @param x \code{summary.bvarmn} object
#' @param ... not used
#' @order 3
#' @export
knit_print.summary.bvarmn <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "summary.bvarmn",
  knit_print.summary.bvarmn,
  envir = asNamespace("knitr")
)

