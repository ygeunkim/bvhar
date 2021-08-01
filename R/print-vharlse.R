#' Print Method for \code{vhar} Object
#' 
#' @param x \code{vharlse} object
#' @param digits digit option
#' @param ... not used
#' 
#' @rdname vhar_lm
#' @export
print.vharlse <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # split the matrix for the print: B1, ..., Bp
  phihat_mat <- 
    split.data.frame(x$coefficients[-(3 * x$m + 1),], gl(3, x$m)) %>% 
    lapply(t)
  names(phihat_mat) <- c("day", "week", "month")
  # const term
  intercept <- x$coefficients[3 * x$m + 1,]
  cat("VHAR Estimation")
  cat("====================================================\n\n")
  for (i in 1:3) {
    cat(paste("LSE for", names(phihat_mat)[i]))
    # B1, ..., Bp--------------------
    print.default(
      phihat_mat[[i]],
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
  cat("*_day, *_week, *_month of the Coefficient matrix: daily, weekly, and monthly term in the VHAR model\n\n")
  invisible(x)
}

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

