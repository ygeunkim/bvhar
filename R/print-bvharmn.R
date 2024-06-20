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
  # split the matrix for the print: Phi(d), Phi(w), Phi(m)
  phihat_mat <- split_coef(x)
  names(phihat_mat) <- c("day", "week", "month")
  cat("BVHAR with Minnesota Prior\n")
  cat("====================================================\n\n")
  cat("Phi ~ Matrix Normal (Mean, Scale 1, Scale 2 = Sigma)\n")
  cat("====================================================\n")
  for (i in 1:x$p) {
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
    x$covmat,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  # cat("====================================================\n\n")
  # cat("Parameter record:\n")
  # print(
  #   x$param,
  #   digits = digits,
  #   print.gap = 2L,
  #   quote = FALSE
  # )
  invisible(x)
}

#' @rdname bvhar_minnesota
#' @exportS3Method knitr::knit_print
knit_print.bvharmn <- function(x, ...) {
  print(x)
}

#' @rdname bvhar_minnesota
#' @param x `bvharhm` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharhm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep = "\n", collapse = "\n"), "\n\n",
    sep = ""
  )
  cat("BVHAR with Hierarchical Minnesota Prior\n")
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
    subset_draws(x$param, variable = "phi"),
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

#' @rdname bvhar_minnesota
#' @exportS3Method knitr::knit_print
knit_print.bvharhm <- function(x, ...) {
  print(x)
}
