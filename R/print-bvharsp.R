#' @rdname bvar_ssvs
#' @param x `bvarssvs` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvarssvs <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  cat(sprintf("BVAR(%i) with SSVS Prior\n", x$p))
  cat("Fitted by Gibbs sampling\n")
  cat(paste0("Total number of iteration: ", x$iter, "\n"))
  cat(paste0("Number of burn-in: ", x$burn, "\n"))
  if (x$thin > 1) {
    cat(paste0("Thinning: ", x$thin, "\n"))
  }
  cat("====================================================\n\n")
  cat("Parameter Record:\n")
  print(
    x$param,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname bvar_ssvs
#' @exportS3Method knitr::knit_print
knit_print.bvarssvs <- function(x, ...) {
  print(x)
}

#' @rdname bvhar_ssvs
#' @param x `bvharssvs` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharssvs <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  cat("BVHAR with SSVS Prior\n")
  cat("Fitted by Gibbs sampling\n")
  cat(paste0("Total number of iteration: ", x$iter, "\n"))
  cat(paste0("Number of burn-in: ", x$burn, "\n"))
  if (x$thin > 1) {
    cat(paste0("Thinning: ", x$thin, "\n"))
  }
  cat("====================================================\n\n")
  cat("Parameter Record:\n")
  print(
    x$param,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname bvhar_ssvs
#' @exportS3Method knitr::knit_print
knit_print.bvharssvs <- function(x, ...) {
  print(x)
}

#' @rdname summary.ssvsmod
#' @param x `summary.ssvsmod` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.summary.ssvsmod <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  mod_type <- gsub(pattern = "\\_SSVS$", replacement = "", x$process)
  coef_list <- switch(
    x$type,
    "const" = {
      split.data.frame(x$coefficients[-(x$p * x$m + 1),], gl(x$p, x$m)) %>% 
        lapply(t)
    },
    "none" = {
      split.data.frame(x$coefficients, gl(x$p, x$m)) %>% 
        lapply(t)
    }
  )
  cat(paste0("Variable Selection for ", mod_type, "(", sprintf("%i", x$p), ") using SSVS\n"))
  cat("====================================================\n\n")
  coef_nm <- ifelse(mod_type == "VAR", "A", "Phi")
  for (i in seq_along(coef_list)) {
    cat(paste0("Variable selection for ", sprintf("%s", coef_nm), sprintf("%i:\n", i)))
    print.default(
      coef_list[[i]],
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
  }
  cat("\n")
  cat("Variable selection for Cholesky factors:\n")
  print.default(
    x$cholesky,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname summary.ssvsmod
#' @exportS3Method knitr::knit_print
knit_print.summary.ssvsmod <- function(x, ...) {
  print(x)
}
