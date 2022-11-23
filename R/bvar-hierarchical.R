#' Hyperpriors for Bayesian Models
#' 
#' Set hyperpriors of Bayesian VAR and VHAR models.
#' 
#' @param mode Mode of Gamma distribution
#' @param sd Standard deviation of Gamma distribution
#' @details 
#' In addition to Normal-IW priors [set_bvar()], [set_bvhar()], and [set_weight_bvhar()],
#' these functions give hierarchical structure to the model.
#' * `set_lambda()` specifies hyperprior for \eqn{\lambda}, which is Gamma distribution.
#' * `set_psi()` specifies hyperprior for \eqn{\psi / (\nu_0 - k - 1)}, which is Inverse gamma distribution.
#' @references Giannone, D., Lenza, M., & Primiceri, G. E. (2015). *Prior Selection for Vector Autoregressions*. Review of Economics and Statistics, 97(2). doi:[10.1162/REST_a_00483](https://doi.org/10.1162/REST_a_00483)
#' @order 1
#' @export
set_lambda <- function(mode = .2, sd = .4) {
  params <- get_gammaparam(mode, sd)
  lam_prior <- list(
    shape = params$shape,
    rate = params$rate,
    mode = mode,
    mean = params$shape / params$rate,
    sd = sd
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
    shape = shape,
    scale = scale,
    mode = scale / (shape + 1),
    mean = scale / (shape - 1),
    sd = scale / ((shape - 1) * sqrt(shape - 1))
  )
  class(psi_prior) <- "bvharpriorspec"
  psi_prior
}
