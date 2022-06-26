#' @rdname analyze_ir
#' @param x `bvharirf` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharirf <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  irf_type <- ifelse(x$orthogonal, "Orthogonal Impulses", "Forecast Errors")
  cat(sprintf("Impulse Response Analysis of %s\n", x$process))
  cat(sprintf("Responses to %s:\n", irf_type))
  irf_list <- split_coef(x)
  cat("====================================================\n\n")
  for (i in 1:(x$lag_max + 1)) {
    cat(sprintf("Impulse -> Response (Period = %i):\n", i))
    print.default(
      irf_list[[i]],
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n")
  }
  invisible(x)
}

#' @rdname analyze_ir
#' @param x `bvharirf` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharirf <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvharirf",
  knit_print.bvharirf,
  envir = asNamespace("knitr")
)
