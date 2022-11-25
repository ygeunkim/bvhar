#' Hyperpriors for Bayesian Models
#' 
#' Set hyperpriors of Bayesian VAR and VHAR models.
#' 
#' @param mode Mode of Gamma distribution
#' @param sd Standard deviation of Gamma distribution
#' @details 
#' In addition to Normal-IW priors [set_bvar()], [set_bvhar()], and [set_weight_bvhar()],
#' these functions give hierarchical structure to the model.
#' * `set_lambda()` specifies hyperprior for \eqn{\lambda} (`lambda`), which is Gamma distribution.
#' * `set_psi()` specifies hyperprior for \eqn{\psi / (\nu_0 - k - 1) = \sigma^2} (`sigma`), which is Inverse gamma distribution.
#' @references Giannone, D., Lenza, M., & Primiceri, G. E. (2015). *Prior Selection for Vector Autoregressions*. Review of Economics and Statistics, 97(2). doi:[10.1162/REST_a_00483](https://doi.org/10.1162/REST_a_00483)
#' @order 1
#' @export
set_lambda <- function(mode = .2, sd = .4) {
  params <- get_gammaparam(mode, sd)
  lam_prior <- list(
    hyperparam = "lambda",
    param = c(params$shape, params$rate),
    mode = mode
  )
  class(lam_prior) <- "bvharpriorspec"
  lam_prior
}

#' @rdname set_lambda
#' @param shape Shape of Inverse Gamma distribution
#' @param scale Scale of Inverse Gamma distribution
#' @order 1
#' @export
set_psi <- function(shape = 4e-4, scale = 4e-4) {
  psi_prior <- list(
    hyperparam = "psi",
    param = c(shape, scale),
    mode = scale / (shape + 1)
  )
  class(psi_prior) <- "bvharpriorspec"
  psi_prior
}

#' @rdname set_lambda
#' @param x `bvharpriorspec` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharpriorspec <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(paste0("Hyperprior specification for ", x$hyperparam, "\n\n"))
  hyper_prior <- ifelse(x$hyperparam == "lambda", "Gamma", "Inv-Gamma")
  switch (hyper_prior,
    "Gamma" = {
      print.default(
        paste0(
          x$hyperparam,
          " ~ ",
          hyper_prior,
          "(shape = ",
          x$param[1],
          ", rate =",
          x$param[2],
          ")"
        ),
        digits = digits,
        print.gap = 2L,
        quote = FALSE
      )
    },
    "Inv-Gamma" = {
      print.default(
        paste0(
          x$hyperparam,
          " ~ ",
          hyper_prior,
          "(shape = ",
          x$param[1],
          ", scale =",
          x$param[2],
          ")"
        ),
        digits = digits,
        print.gap = 2L,
        quote = FALSE
      )
    }
  )
  cat(sprintf("with mode: %.3f", x$mode))
  invisible(x)
}

#' @rdname set_lambda
#' @param x `bvharpriorspec` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharpriorspec <- function(x, ...) {
  print(x)
}

registerS3method(
  "knit_print", "bvharpriorspec",
  knit_print.bvharpriorspec,
  envir = asNamespace("knitr")
)
