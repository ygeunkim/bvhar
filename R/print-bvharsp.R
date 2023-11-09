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
#' @param x `bvarssvs` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvarssvs <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvarssvs",
  knit_print.bvarssvs,
  envir = asNamespace("knitr")
)

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
#' @param x `bvharssvs` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharssvs <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvharssvs",
  knit_print.bvharssvs,
  envir = asNamespace("knitr")
)

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
#' @param x `summary.ssvsmod` object
#' @param ... not used
#' @order 3
#' @export
knit_print.summary.ssvsmod <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "summary.ssvsmod",
  knit_print.summary.ssvsmod,
  envir = asNamespace("knitr")
)

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
#' @param x `bvarhs` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvarhs <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvarhs",
  knit_print.bvarhs,
  envir = asNamespace("knitr")
)

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
#' @param x `bvharhs` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharhs <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvharhs",
  knit_print.bvharhs,
  envir = asNamespace("knitr")
)

#' @rdname summary.hsmod
#' @param x `summary.hsmod` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.summary.hsmod <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  mod_type <- gsub(pattern = "\\_Horseshoe$", replacement = "", x$process)
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
  cat(paste0("Variable Selection for ", mod_type, "(", sprintf("%i", x$p), ") using MVHS\n"))
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
  cat("--------------------------------------------------\n")
  cat(
    paste0("By 100(1-", sprintf("%g", x$level), ")% credible interval:\n")
  )
  print(
    x$interval,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname summary.hsmod
#' @param x `summary.hsmod` object
#' @param ... not used
#' @order 3
#' @export
knit_print.summary.hsmod <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "summary.hsmod",
  knit_print.summary.hsmod,
  envir = asNamespace("knitr")
)

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
#' @param x `bvarsv` object
#' @param ... not used
#' @order 3
#' @export
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
#' @param x `bvarsv` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharsv <- function(x, ...) {
  print(x)
}