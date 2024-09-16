#' @rdname bvar_minnesota
#' @param x `bvarmn` object
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
  bhat_mat <- split_coef(x)
  cat(sprintf("BVAR(%i) with Minnesota Prior\n", x$p))
  cat("====================================================\n\n")
  cat("A ~ Matrix Normal (Mean, Precision, Scale = Sigma)\n")
  cat("====================================================\n")
  for (i in 1:(x$p)) {
    cat(sprintf("Matrix Normal Mean for A%i part:\n", i))
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
    cat("Matrix Normal Mean for constant part:\n")
    print.default(
      intercept,
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n\n")
  }
  # precision matrix---------------0
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
    x$covmat,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\nIW degrees of freedom:\n")
  print.default(x$iw_shape)
  # cat("====================================================\n\n")
  # cat("Parameter record:\n")
  # print(
  #   x$param,
  #   digits = digits,
  #   print.gap = 2L,
  #   quote = FALSE
  # )
  cat("\n\n--------------------------------------------------\n")
  cat("*_j of the Coefficient matrix: corresponding to the j-th BVAR lag\n\n")
  invisible(x)
}

#' @rdname bvar_minnesota
#' @exportS3Method knitr::knit_print
knit_print.bvarmn <- function(x, ...) {
  print(x)
}

#' @rdname bvar_minnesota
#' @param x `bvarhm` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvarhm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep = "\n", collapse = "\n"), "\n\n",
    sep = ""
  )
  cat(sprintf("BVAR(%i) with Hierarchical Minnesota Prior\n", x$p))
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
    subset_draws(x$param, variable = "alpha"),
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\nSigma ~ Inverse-Wishart Record:\n")
  print(
    # x$sigma_record,
    subset_draws(x$param, variable = "sigma"),
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname bvar_minnesota
#' @exportS3Method knitr::knit_print
knit_print.bvarhm <- function(x, ...) {
  print(x)
}
