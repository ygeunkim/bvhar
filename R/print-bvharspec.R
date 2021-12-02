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
      print.default(
        param[[i]],
        digits = digits,
        print.gap = 2L,
        quote = FALSE
      )
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

#' @rdname set_spikeslab_coef
#' @param x `bvharss_coef` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @importFrom utils head
#' @export
print.bvharss_coef <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Spike and slab specification for VAR coefficient matrix\n\n")
  cat(paste0(
    "alpha[j] | gamma[j] ~ (1 - gamma[j])N(0, kappa[0j]^2)",
    " + gamma[j]N(0, kappa[1j]^2)"
  ))
  # spike-------------------------------------
  cat("\nwhere head(kappa[0j]):\n")
  print.default(
    head(x$coef_spike),
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  # slab--------------------------------------
  cat("and head(kappa[1j]):\n")
  print.default(
    head(x$coef_slab),
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  # gamma-----------------------------------
  cat("\ngamma[j] ~ Bernoulli(q[j])\n")
  cat("where head(q[j]):\n")
  print.default(
    head(x$coef_mixture),
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname set_spikeslab_coef
#' @param x `bvharss_coef` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharss_coef <- function(x, ...) {
  print(x)
}

registerS3method(
  "knit_print", "bvharss_coef",
  knit_print.bvharss_coef,
  envir = asNamespace("knitr")
)

#' @rdname set_spikeslab_cov
#' @param x `bvharss_sig` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @importFrom utils head
#' @export
print.bvharss_sig <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Spike and slab specification for VAR covariance matrix\n\n")
  cat("Lower triangular Cholesky for precision matrix:\n")
  cat("Sigma^(-1) = Psi Psi^T where Psi is upper triangular\n")
  # Diagonal components--------------------------------------
  cat("Diagonal psi[jj]^2 ~ Gamma(shape = a[j], rate = b[j])\n")
  cat("where head(a[j]):\n")
  print.default(
    head(x$cov_shape),
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("and head(b[j]):\n")
  print.default(
    head(x$cov_rate),
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  # Off-diagonal components----------------------------------
  cat("Lower trianular w[ij] ~ Bernouli(q[ij])\n")
  cat("where head(q[ij]):\n")
  # later
}

#' @rdname set_spikeslab_cov
#' @param x `bvharss_sig` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharss_sig <- function(x, ...) {
  print(x)
}

registerS3method(
  "knit_print", "bvharss_sig",
  knit_print.bvharss_sig,
  envir = asNamespace("knitr")
)
