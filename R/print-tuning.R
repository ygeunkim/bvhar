#' @rdname choose_bvar
#' @param x `bvharemp` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharemp <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  print(x$spec)
  invisible(x)
}

#' @rdname choose_bvar
#' @param x `bvharemp` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharemp <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "bvharemp",
  knit_print.bvharemp,
  envir = asNamespace("knitr")
)

#' @rdname bound_bvhar
#' @param x `boundbvharemp` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.boundbvharemp <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  dim_data <- length(x$spec$sigma)
  lower_bound <- x$lower
  upper_bound <- x$upper
  mod_type <- ifelse(x$spec$prior == "Minnesota", "BVAR", "BVHAR")
  lower_spec <- switch(
    x$spec$prior,
    "Minnesota" = {
      set_bvar(
        sigma = lower_bound[1:dim_data],
        lambda = lower_bound[dim_data + 1],
        eps = 1e-04,
        delta = lower_bound[(dim_data + 2):(dim_data * 2 + 1)]
      )
    },
    "MN_VAR" = {
      set_bvhar(
        sigma = lower_bound[1:dim_data],
        lambda = lower_bound[dim_data + 1],
        eps = 1e-04,
        delta = lower_bound[(dim_data + 2):(dim_data * 2 + 1)]
      )
    },
    "MN_VHAR" = {
      set_weight_bvhar(
        sigma = lower_bound[1:dim_data],
        lambda = lower_bound[dim_data + 1],
        eps = 1e-04,
        daily = lower_bound[(dim_data + 2):(dim_data * 2 + 1)],
        weekly = lower_bound[(dim_data * 2 + 2):((dim_data * 3 + 1))],
        monthly = lower_bound[(dim_data * 3 + 2):((dim_data * 4 + 1))]
      )
    }
  )
  upper_spec <- switch(
    x$spec$prior,
    "Minnesota" = {
      set_bvar(
        sigma = upper_bound[1:dim_data],
        lambda = upper_bound[dim_data + 1],
        eps = 1e-04,
        delta = upper_bound[(dim_data + 2):(dim_data * 2 + 1)]
      )
    },
    "MN_VAR" = {
      set_bvhar(
        sigma = upper_bound[1:dim_data],
        lambda = upper_bound[dim_data + 1],
        eps = 1e-04,
        delta = upper_bound[(dim_data + 2):(dim_data * 2 + 1)]
      )
    },
    "MN_VHAR" = {
      set_weight_bvhar(
        sigma = upper_bound[1:dim_data],
        lambda = upper_bound[dim_data + 1],
        eps = 1e-04,
        daily = upper_bound[(dim_data + 2):(dim_data * 2 + 1)],
        weekly = upper_bound[(dim_data * 2 + 2):((dim_data * 3 + 1))],
        monthly = upper_bound[(dim_data * 3 + 2):((dim_data * 4 + 1))]
      )
    }
  )
  cat(sprintf("Lower bound specification for %s optimization (L-BFGS-B):\n", mod_type))
  cat("========================================================\n")
  cat("========================================================\n")
  print(lower_spec)
  cat("\n\n")
  cat(sprintf("Upper bound specification for %s optimization (L-BFGS-B):\n", mod_type))
  cat("========================================================\n")
  cat("========================================================\n")
  print(upper_spec)
  invisible(x)
}

#' @rdname bound_bvhar
#' @param x `boundbvharemp` object
#' @param ... not used
#' @order 3
#' @export
knit_print.boundbvharemp <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "boundbvharemp",
  knit_print.boundbvharemp,
  envir = asNamespace("knitr")
)
