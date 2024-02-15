#' @rdname set_bvar
#' @param x `bvharspec` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharspec <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(paste0("Model Specification for ", x$process, "\n\n"))
  cat("Parameters: Coefficent matrice and Covariance matrix\n")
  cat(paste0("Prior: ", x$prior, "\n"))
  fit_func <- switch(
    x$prior,
    "Minnesota" = "?bvar_minnesota",
    "Flat" = "?bvar_flat",
    "MN_VAR" = "?bvhar_minnesota",
    "MN_VHAR" = "?bvhar_minnesota",
    "MN_Hierarchical" = "?bvar_",
    stop("Invalid 'x$prior' element")
  )
  cat(paste0("# Type '", fit_func, "' in the console for some help.", "\n"))
  cat("========================================================\n\n")
  param <- x[!(names(x) %in% c("process", "prior"))]
  for (i in seq_along(param)) {
    cat(paste0("Setting for '", names(param)[i], "':\n"))
    if (is.matrix(param[[i]])) {
      type <- "a" # not large
      if (nrow(param[[i]]) > 10 & ncol(param[[i]]) > 5) {
        type <- "b" # both large
      } else if (nrow(param[[i]]) > 10 & ncol(param[[i]]) <= 5) {
        type <- "c" # large row
      } else if (nrow(param[[i]]) <= 10 & ncol(param[[i]]) > 5) {
        type <- "d" # large column
      }
      switch(
        type,
        "a" = {
          print.default(
            param[[i]],
            digits = digits,
            print.gap = 2L,
            quote = FALSE
          )
          cat("\n")
        },
        "b" = {
          cat(
            paste0("# A matrix: "), 
            paste(nrow(param[[i]]), "x", ncol(param[[i]])),
            "\n"
          )
          print.default(
            param[[i]][1:10, 1:5],
            digits = digits,
            print.gap = 2L,
            quote = FALSE
          )
          cat(paste0("# ... with ", nrow(param[[i]]) - 10, " more rows", "\n"))
        },
        "c" = {
          print.default(
            param[[i]][1:10,],
            digits = digits,
            print.gap = 2L,
            quote = FALSE
          )
          cat(paste0("# ... with ", nrow(param[[i]]) - 10, " more rows", "\n"))
        },
        "d" = {
          cat(
            paste0("# A matrix: "), 
            paste(nrow(param[[i]]), "x", ncol(param[[i]])), 
            "\n"
          )
          print.default(
            param[[i]][1:10, 1:5],
            digits = digits,
            print.gap = 2L,
            quote = FALSE
          )
          cat("\n")
        }
      )
    } else {
      if (is.bvharpriorspec(param[[i]])) {
        print(
          param[[i]],
          digits = max(3L, getOption("digits") - 3L),
          print.gap = 2L,
          quote = FALSE
        )
      } else {
        print.default(
          param[[i]],
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
      }
      cat("\n")
    }
  }
}

#' @rdname set_bvar
#' @exportS3Method knitr::knit_print
knit_print.bvharspec <- function(x, ...) {
  print(x)
}

#' @rdname set_intercept
#' @param x `interceptspec` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.interceptspec <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(paste0("Model Specification for ", x$process, ":\n"))
  cat("----------------------------------------------------\n")
  prior_var <- paste0(x$sd_non, "^2 I_dim")
  if (length(x$mean_non) > 1) {
    prior_mean <- paste0(
      "c(",
      paste0(x$mean, collapse = ","),
      ")"
    )
  } else {
    prior_mean <- paste0("rep(", x$mean_non, ", dim)")
  }
  cat(sprintf("%s(%s, %s)", x$prior, prior_mean, prior_var))
}

#' @rdname set_intercept
#' @exportS3Method knitr::knit_print
knit_print.interceptspec <- function(x, ...) {
  print(x)
}

#' @rdname set_ssvs
#' @param x `ssvsinput`
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.ssvsinput <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(paste0("Model Specification for ", x$process, " with ", x$prior, " Prior", "\n\n"))
  cat("Parameters: Coefficent matrix, Cholesky Factor, and Each Restriction Dummy\n")
  cat(paste0("Prior: ", x$prior, "\n"))
  fit_func <- switch(
    x$process,
    "VAR" = "?bvar_ssvs",
    "VHAR" = "?bvhar_ssvs",
    stop("Invalid 'x$prior' element")
  )
  cat(paste0("# Type '", fit_func, "' in the console for some help.", "\n"))
  cat("========================================================\n")
  param <- x[!(names(x) %in% c("process", "prior"))]
  for (i in seq_along(param)) {
    cat(paste0("Setting for '", names(param)[i], "':\n"))
    if (is.matrix(param[[i]])) {
      type <- "a"
    } else if (length(param[[i]]) == 1) {
      type <- "b"
    } else {
      type <- "c"
    }
    switch(
      type,
      "a" = {
        print.default(
          param[[i]],
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
      },
      "b" = {
        if (grepl(pattern = "s1$|s2$", names(param)[i])) {
          pseudo_param <- param[[i]] # s1, s2
        } else if (grepl(pattern = "^coef", names(param)[i]) && names(param)[i] != "sd_non") {
          pseudo_param <- paste0("rep(", param[[i]], ", dim^2 * p)") # coef_
        } else if (grepl(pattern = "^chol", names(param)[i])) {
          pseudo_param <- paste0("rep(", param[[i]], ", dim * (dim - 1) / 2)") # chol_
        } else if (names(param)[i] == "sd_non") {
          pseudo_param <- paste0("prior variance = ", param[[i]], "^2 I_dim")
        } else {
          pseudo_param <- paste0("rep(", param[[i]], ", dim)") # shape and rate
        }
        print.default(
          pseudo_param,
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
      },
      "c" = {
        print.default(
          param[[i]],
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
      }
    )
    cat("\n")
  }
  cat("--------------------------------------------------------------\n")
  cat("dim: time series dimension, p: VAR order")
}

#' @rdname set_ssvs
#' @exportS3Method knitr::knit_print
knit_print.ssvsinput <- function(x, ...) {
  print(x)
}

#' @rdname init_ssvs
#' @param x `ssvsinit`
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.ssvsinit <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(paste0("Gibbs Sampler Initialization for ", x$process, " with ", x$prior, " Prior", "\n\n"))
  cat("Parameters: Coefficent matrix, Cholesky Factor, and Each Restriction Dummy\n")
  # cat(paste0("Prior: ", x$prior, "\n"))
  fit_func <- switch(
    x$process,
    "VAR" = "?bvar_ssvs",
    "VHAR" = "?bvhar_ssvs",
    stop("Invalid 'x$prior' element")
  )
  cat(paste0("# Type '", fit_func, "' in the console for some help.", "\n"))
  cat("========================================================\n")
  param <- x[!(names(x) %in% c("process", "prior", "chain"))]
  # num_chain <- x$chain
  for (i in seq_along(param)) {
    cat(paste0("Initialization for '", names(param)[i], "':\n"))
    type <- "a"
    if (is.list(param[[i]])) {
      # type <- "a"
      if (nrow(param[[i]][[1]]) > 7 & ncol(param[[i]][[1]]) > 6) {
        type <- "a_large" # both large
      } else if (nrow(param[[i]][[1]]) > 7 & ncol(param[[i]][[1]]) <= 6) {
        type <- "a_row" # large row
      } else if (nrow(param[[i]][[1]]) <= 7 & ncol(param[[i]][[1]]) > 6) {
        type <- "a_column" # large column
      }
    } else if (is.matrix(param[[i]])) {
      type <- "b" # not large one matrix
      if (nrow(param[[i]]) > 7 & ncol(param[[i]]) > 6) {
        type <- "c" # both large
      } else if (nrow(param[[i]]) > 7 & ncol(param[[i]]) <= 6) {
        type <- "d" # large row
      } else if (nrow(param[[i]]) <= 7 & ncol(param[[i]]) > 6) {
        type <- "e" # large column
      }
    }
    switch(
      type,
      "a" = {
        for (j in seq_along(param[[i]])) {
          # cat(gettextf("# In chain %d:\n", j))
          print.default(
            param[[i]][[j]],
            digits = digits,
            print.gap = 2L,
            quote = FALSE
          )
        }
        cat("\n")
      },
      "a_large" = {
        for (j in seq_along(param[[i]])) {
          # cat(gettextf("# In chain %d:\n", j))
          cat(paste0(
            "# A matrix: ",
            nrow(param[[i]][[j]]),
            " x ",
            ncol(param[[i]][[j]]),
            "\n"
          ))
          print.default(
            param[[i]][[j]][1:7, 1:6],
            digits = digits,
            print.gap = 2L,
            quote = FALSE
          )
          cat(paste0("# ... with ", nrow(param[[i]][[j]]) - 7, " more rows", "\n"))
        }
        cat("\n")
      },
      "a_row" = {
        for (j in seq_along(param[[i]])) {
          # cat(gettextf("# In chain %d:\n", j))
          print.default(
            param[[i]][[j]][1:7,],
            digits = digits,
            print.gap = 2L,
            quote = FALSE
          )
          cat(paste0("# ... with ", nrow(param[[i]][[j]]) - 7, " more rows", "\n"))
        }
        cat("\n")
      },
      "a_column" = {
        for (j in seq_along(param[[i]])) {
          # cat(gettextf("# In chain %d:\n", j))
          cat(paste0(
            "# A matrix: ",
            nrow(param[[i]][[j]]),
            " x ",
            ncol(param[[i]][[j]]),
            "\n"
          ))
          print.default(
            param[[i]][[j]][1:7, 1:6],
            digits = digits,
            print.gap = 2L,
            quote = FALSE
          )
        }
        cat("\n")
      },
      "b" = {
        print.default(
          param[[i]],
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
        cat("\n")
      },
      "c" = {
        cat(
          paste0("# A matrix: "), 
          paste(nrow(param[[i]]), "x", ncol(param[[i]])),
          "\n"
        )
        print.default(
          param[[i]][1:7, 1:6],
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
        cat(paste0("# ... with ", nrow(param[[i]]) - 7, " more rows", "\n"))
      },
      "d" = {
        print.default(
          param[[i]][1:7,],
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
        cat(paste0("# ... with ", nrow(param[[i]]) - 7, " more rows", "\n"))
      },
      "e" = {
        cat(
          paste0("# A matrix: "), 
          paste(nrow(param[[i]]), "x", ncol(param[[i]])), 
          "\n"
        )
        print.default(
          param[[i]][1:7, 1:6],
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
        cat("\n")
      }
    )
  }
}

#' @rdname init_ssvs
#' @exportS3Method knitr::knit_print
knit_print.ssvsinit <- function(x, ...) {
  print(x)
}


#' @rdname set_horseshoe
#' @param x `horseshoespec`
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.horseshoespec <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(paste0("Model Specification for ", x$process, " with ", x$prior, " Prior", "\n\n"))
  cat("Parameters: Coefficent matrix, Covariance (precision) matrix\n")
  cat(paste0("Prior: ", x$prior, "\n"))
  fit_func <- switch(
    x$process,
    "VAR" = "?bvar_horseshoe",
    "VHAR" = "?bvhar_horseshoe",
    stop("Invalid 'x$process' element")
  )
  cat(paste0("# Type '", fit_func, "' in the console for some help.", "\n"))
  cat("========================================================\n")
  param <- x[!(names(x) %in% c("process", "prior", "chain"))]
  # num_chain <- x$chain
  for (i in seq_along(param)) {
    cat(paste0("Initialization for '", names(param)[i], "':\n"))
    print.default(
      param[[i]],
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
  }
  # if (num_chain > 1) {
  #   cat("--------------------------------------------------------------\n")
  #   cat("Initialized for multiple chain MCMC.")
  # }
  # cat("--------------------------------------------------------\n")
  # cat("'init_local': local shrinkage for each row of coefficients matrix")
}

#' @rdname set_horseshoe
#' @exportS3Method knitr::knit_print
knit_print.horseshoespec <- function(x, ...) {
  print(x)
}

#' @rdname set_sv
#' @param x `svspec`
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.svspec <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(paste0("Model Specification for ", x$process, " with ", x$prior, " Prior", "\n\n"))
  cat("Parameters: Contemporaneous coefficients, State variance, Initial state\n")
  cat(paste0("Prior: ", x$prior, "\n"))
  cat("========================================================\n")
  param <- x[!(names(x) %in% c("process", "prior"))]
  for (i in seq_along(param)) {
    cat(paste0("Setting for '", names(param)[i], "':\n"))
    if (is.matrix(param[[i]])) {
      type <- "a"
    } else if (length(param[[i]]) == 1) {
      type <- "b"
    } else {
      type <- "c"
    }
    switch(type,
      "a" = {
        print.default(
          param[[i]],
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
      },
      "b" = {
        if (names(param)[i] == "initial_prec") {
          pseudo_param <- paste0(param[[i]], " * diag(dim)")
        } else {
          pseudo_param <- paste0("rep(", param[[i]], ", dim)")
        }
        print.default(
          pseudo_param,
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
      },
      "c" = {
        print.default(
          param[[i]],
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
      }
    )
    cat("\n")
  }
}