#' @rdname bvhar_minnesota
#' @param x \code{bvarmn} object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharmn <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # split the matrix for the print: B1, ..., Bp
  phihat_mat <- 
    split.data.frame(x$mn_mean[-(3 * x$m + 1),], gl(3, x$m)) %>% 
    lapply(t)
  names(phihat_mat) <- c("day", "week", "month")
  # const term
  intercept <- x$mn_mean[3 * x$m + 1,]
  cat("BVHAR with Minnesota Prior\n")
  cat("====================================================\n\n")
  cat("Phi ~ Matrix Normal (Mean, Scale 1, Scale 2 = Sigma)\n")
  cat("====================================================\n")
  for (i in 1:3) {
    cat(paste0("Matrix Normal Mean for ", names(phihat_mat)[i], ":\n"))
    # B1, ..., Bp--------------------
    print.default(
      phihat_mat[[i]],
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

#' @rdname bvhar_minnesota
#' @param x \code{bvarmn} object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharmn <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvharmn",
  knit_print.bvharmn,
  envir = asNamespace("knitr")
)
