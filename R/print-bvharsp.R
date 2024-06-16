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
  if (x$chain > 1) {
    cat(paste0("Number of chains: ", x$chain, "\n"))
  }
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
  if (x$chain > 1) {
    cat(paste0("Number of chains: ", x$chain, "\n"))
  }
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

#' @rdname bvar_horseshoe
#' @param x `bvarhs` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvarhs <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  cat(sprintf("BVAR(%i) with Horseshoe Prior\n", x$p))
  cat("Fitted by blocked Gibbs sampling\n")
  # cat(paste0("Fitted by ", x$algo, " sampling", "\n"))
  if (x$chain > 1) {
    cat(paste0("Number of chains: ", x$chain, "\n"))
  }
  cat(paste0("Total number of iteration: ", x$iter, "\n"))
  cat(paste0("Number of burn-in: ", x$burn, "\n"))
  if (x$thin > 1) {
    cat(paste0("Thinning: ", x$thin, "\n"))
  }
  cat("====================================================\n\n")
  print(
    x$param,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname bvar_horseshoe
#' @exportS3Method knitr::knit_print
knit_print.bvarhs <- function(x, ...) {
  print(x)
}

#' @rdname bvhar_horseshoe
#' @param x `bvharhs` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharhs <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  cat("BVHAR with Horseshoe Prior\n")
  # cat("Fitted by Gibbs sampling\n")
  cat(paste0("Fitted by ", x$algo, " sampling", "\n"))
  if (x$chain > 1) {
    cat(paste0("Number of chains: ", x$chain, "\n"))
  }
  cat(paste0("Total number of iteration: ", x$iter, "\n"))
  cat(paste0("Number of burn-in: ", x$burn, "\n"))
  if (x$thin > 1) {
    cat(paste0("Thinning: ", x$thin, "\n"))
  }
  cat("====================================================\n\n")
  print(
    x$param,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname bvhar_horseshoe
#' @exportS3Method knitr::knit_print
knit_print.bvharhs <- function(x, ...) {
  print(x)
}

#' @rdname bvar_sv
#' @param x `bvarsv` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvarsv <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep = "\n", collapse = "\n"), "\n\n",
    sep = ""
  )
  cat(sprintf("BVAR(%i) with Stochastic Volatility\n", x$p))
  cat("Fitted by Gibbs sampling\n")
  if (x$chain > 1) {
    cat(paste0("Number of chains: ", x$chain, "\n"))
  }
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

#' @rdname bvar_sv
#' @exportS3Method knitr::knit_print
knit_print.bvarsv <- function(x, ...) {
  print(x)
}

#' @rdname bvhar_sv
#' @param x `bvarsv` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharsv <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep = "\n", collapse = "\n"), "\n\n",
    sep = ""
  )
  cat("BVHAR with Stochastic Volatility\n")
  cat("Fitted by Gibbs sampling\n")
  if (x$chain > 1) {
    cat(paste0("Number of chains: ", x$chain, "\n"))
  }
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

#' @rdname bvhar_sv
#' @exportS3Method knitr::knit_print
knit_print.bvharsv <- function(x, ...) {
  print(x)
}

#' @rdname summary.bvharsp
#' @param x `summary.bvharsp` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.summary.bvharsp <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep = "\n", collapse = "\n"), "\n\n",
    sep = ""
  )
  # mod_type <- gsub(pattern = "\\_SSVS$", replacement = "", x$process)
  mod_type <- gsub(pattern = "\\_SSVS|\\_Horseshoe", replacement = "", x = x$process)
  vhar_type <- gsub(pattern = "_SV$", replacement = "", x = mod_type)
  selection_method <- x$method
  coef_list <- switch(x$type,
    "const" = {
      split.data.frame(x$coefficients[-(x$p * x$m + 1), ], gl(x$p, x$m))
    },
    "none" = {
      split.data.frame(x$coefficients, gl(x$p, x$m))
    }
  )
  # cat(paste0("Variable Selection for ", mod_type, "(", sprintf("%i", x$p), ") using SSVS\n"))
  cat(paste0(
    "Variable Selection for ",
    # mod_type,
    ifelse(
      vhar_type == "VHAR",
      mod_type,
      paste0(mod_type, "(", sprintf("%i", x$p), ")")
    ),
    # "(",
    # sprintf("%i", x$p),
    # ") using ",
    " using ",
    selection_method,
    "*\n"
  ))
  cat("====================================================\n\n")
  coef_nm <- ifelse(mod_type == "VAR", "A", "Phi")
  vhar_name <- c("Day", "Week", "Month")
  for (i in seq_along(coef_list)) {
    if (vhar_type == "VAR") {
      cat(paste0("A", sprintf("%i:\n", i)))
    } else {
      cat(paste0(vhar_name[i], ":\n"))
    }
    print.default(
      coef_list[[i]],
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n")
  }
  if (x$type == "const") {
    intercept <- x$coefficients[x$p * x$m + 1,]
    cat("Constant term:\n")
    print.default(
      intercept,
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n")
  }
  cat("--------------------------------------------------\n")
  if (selection_method == "ci") {
    cat(
      paste0("* 100(1-", sprintf("%g", x$level), ")% credible interval:\n")
    )
    print(
      x$interval,
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
  } else {
    cat(sprintf("* Threshold: %g", x$threshold))
  }
}

#' @rdname summary.bvharsp
#' @exportS3Method knitr::knit_print
knit_print.summary.bvharsp <- function(x, ...) {
  print(x)
}

#' @rdname var_bayes
#' @param x `bvarldlt` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvarldlt <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep = "\n", collapse = "\n"), "\n\n",
    sep = ""
  )
  cat(sprintf("BVAR(%i) with %s prior\n", x$p, x$spec$prior))
  cat("Fitted by Gibbs sampling\n")
  if (x$chain > 1) {
    cat(paste0("Number of chains: ", x$chain, "\n"))
  }
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

#' @rdname var_bayes
#' @exportS3Method knitr::knit_print
knit_print.bvarldlt <- function(x, ...) {
  print(x)
}

#' @rdname vhar_bayes
#' @param x `bvharldlt` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharldlt <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep = "\n", collapse = "\n"), "\n\n",
    sep = ""
  )
  cat(sprintf("BVHAR with %s prior\n", x$spec$prior))
  cat("Fitted by Gibbs sampling\n")
  if (x$chain > 1) {
    cat(paste0("Number of chains: ", x$chain, "\n"))
  }
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

#' @rdname vhar_bayes
#' @exportS3Method knitr::knit_print
knit_print.bvharldlt <- function(x, ...) {
  print(x)
}
