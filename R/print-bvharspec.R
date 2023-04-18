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
#' @param x `bvharspec` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharspec <- function(x, ...) {
  print(x)
}

registerS3method(
  "knit_print", "bvharspec",
  knit_print.bvharspec,
  envir = asNamespace("knitr")
)
