#' Generate Minnesota BVAR Parameters
#' 
#' This function generates parameters of BVAR with Minnesota prior.
#' 
#' @param p VAR lag
#' @param bayes_spec A BVAR model specification by [set_bvar()].
#' @param full Generate variance matrix from IW (default: `TRUE`) or not (`FALSE`)?
#' @details 
#' Implementing dummy observation constructions,
#' Bańbura et al. (2010) sets Normal-IW prior.
#' \deqn{A \mid \Sigma_e \sim MN(A_0, \Omega_0, \Sigma_e)}
#' \deqn{\Sigma_e \sim IW(S_0, \alpha_0)}
#' If `full = FALSE`, the result of \eqn{\Sigma_e} is the same as input (`diag(sigma)`).
#' @return List with the following component.
#' \describe{
#'   \item{coefficients}{BVAR coefficient (MN)}
#'   \item{covmat}{BVAR variance (IW or diagonal matrix of `sigma` of `bayes_spec`)}
#' }
#' @seealso 
#' * [set_bvar()] to specify the hyperparameters of Minnesota prior.
#' @references 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1).
#' 
#' Karlsson, S. (2013). *Chapter 15 Forecasting with Bayesian Vector Autoregression*. Handbook of Economic Forecasting, 2, 791-897.
#' 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25.
#' @examples 
#' # Generate (A, Sigma)
#' # BVAR(p = 2)
#' # sigma: 1, 1, 1
#' # lambda: .1
#' # delta: .1, .1, .1
#' # epsilon: 1e-04
#' set.seed(1)
#' sim_mncoef(
#'   p = 2,
#'   bayes_spec = set_bvar(
#'     sigma = rep(1, 3),
#'     lambda = .1,
#'     delta = rep(.1, 3),
#'     eps = 1e-04
#'   ),
#'   full = TRUE
#' )
#' @export
sim_mncoef <- function(p, bayes_spec = set_bvar(), full = TRUE) {
  # model specification---------------
  if (!is.bvharspec(bayes_spec)) {
    stop("Provide 'bvharspec' for 'bayes_spec'.")
  }
  if (bayes_spec$prior != "Minnesota") {
    stop("'bayes_spec' must be the result of 'set_bvar()'.")
  }
  if (is.null(bayes_spec$sigma)) {
    stop("'sigma' in 'set_bvar()' should be specified. (It is NULL.)")
  }
  sigma <- bayes_spec$sigma
  if (is.null(bayes_spec$delta)) {
    stop("'delta' in 'set_bvar()' should be specified. (It is NULL.)")
  }
  delta <- bayes_spec$delta
  lambda <- bayes_spec$lambda
  eps <- bayes_spec$eps
  dim_data <- length(sigma)
  # dummy-----------------------------
  Yp <- build_ydummy_export(p, sigma, lambda, delta, numeric(dim_data), numeric(dim_data), FALSE)
  Xp <- build_xdummy_export(1:p, lambda, sigma, eps, FALSE)
  # prior-----------------------------
  prior <- minnesota_prior(Xp, Yp)
  mn_mean <- prior$prior_mean
  mn_prec <- prior$prior_prec
  iw_scale <- prior$prior_scale
  iw_shape <- prior$prior_shape
  # random---------------------------
  if (full) {
    res <- sim_mniw_export(
      1,
      mn_mean, # mean of MN
      solve(mn_prec), # scale of MN = inverse of precision
      iw_scale, # scale of IW
      iw_shape, # shape of IW
      FALSE
    )[[1]]
    res <- list(
      coefficients = res[[1]],
      covmat = res[[2]]
    )
  } else {
    sig <- diag(sigma^2)
    res <- sim_matgaussian(
      mn_mean,
      mn_prec,
      sig,
      TRUE
    )
    res <- list(
      coefficients = res,
      covmat = sig
    )
  }
  res
}

#' Generate Minnesota BVAR Parameters
#' 
#' This function generates parameters of BVAR with Minnesota prior.
#' 
#' @param bayes_spec A BVHAR model specification by [set_bvhar()] (default) or [set_weight_bvhar()].
#' @param full Generate variance matrix from IW (default: `TRUE`) or not (`FALSE`)?
#' @details 
#' Normal-IW family for vector HAR model:
#' \deqn{\Phi \mid \Sigma_e \sim MN(M_0, \Omega_0, \Sigma_e)}
#' \deqn{\Sigma_e \sim IW(\Psi_0, \nu_0)}
#' @seealso 
#' * [set_bvhar()] to specify the hyperparameters of VAR-type Minnesota prior.
#' * [set_weight_bvhar()] to specify the hyperparameters of HAR-type Minnesota prior.
#' @return List with the following component.
#' \describe{
#'   \item{coefficients}{BVHAR coefficient (MN)}
#'   \item{covmat}{BVHAR variance (IW or diagonal matrix of `sigma` of `bayes_spec`)}
#' }
#' @references Kim, Y. G., and Baek, C. (2024). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation, 94(6), 1139-1157.
#' @examples 
#' # Generate (Phi, Sigma)
#' # BVHAR-S
#' # sigma: 1, 1, 1
#' # lambda: .1
#' # delta: .1, .1, .1
#' # epsilon: 1e-04
#' set.seed(1)
#' sim_mnvhar_coef(
#'   bayes_spec = set_bvhar(
#'     sigma = rep(1, 3),
#'     lambda = .1,
#'     delta = rep(.1, 3),
#'     eps = 1e-04
#'   ),
#'   full = TRUE
#' )
#' @export
sim_mnvhar_coef <- function(bayes_spec = set_bvhar(), full = TRUE) {
  # model specification---------------
  if (!is.bvharspec(bayes_spec)) {
    stop("Provide 'bvharspec' for 'bayes_spec'.")
  }
  if (bayes_spec$process != "BVHAR") {
    stop("'bayes_spec' must be the result of 'set_bvhar()' or 'set_weight_bvhar()'.")
  }
  if (is.null(bayes_spec$sigma)) {
    stop("'sigma' in 'set_bvhar()' or 'set_weight_bvhar()' should be specified. (It is NULL.)")
  }
  sigma <- bayes_spec$sigma
  lambda <- bayes_spec$lambda
  eps <- bayes_spec$eps
  minnesota_type <- bayes_spec$prior
  dim_data <- length(sigma)
  # dummy-----------------------------
  Yh <- switch(
    minnesota_type,
    "MN_VAR" = {
      if (is.null(bayes_spec$delta)) {
        stop("'delta' in 'set_bvhar()' should be specified. (It is NULL.)")
      }
      Yh <- build_ydummy_export(3, sigma, lambda, bayes_spec$delta, numeric(dim_data), numeric(dim_data), FALSE)
      Yh
    },
    "MN_VHAR" = {
      if (is.null(bayes_spec$daily)) {
        stop("'daily' in 'set_weight_bvhar()' should be specified. (It is NULL.)")
      }
      if (is.null(bayes_spec$weekly)) {
        stop("'weekly' in 'set_weight_bvhar()' should be specified. (It is NULL.)")
      }
      if (is.null(bayes_spec$monthly)) {
        stop("'monthly' in 'set_weight_bvhar()' should be specified. (It is NULL.)")
      }
      Yh <- build_ydummy_export(
        3,
        sigma, 
        lambda, 
        bayes_spec$daily, 
        bayes_spec$weekly, 
        bayes_spec$monthly,
        FALSE
      )
      Yh
    }
  )
  Xh <- build_xdummy_export(1:3, lambda, sigma, eps, FALSE)
  # prior-----------------------------
  prior <- minnesota_prior(Xh, Yh)
  mn_mean <- prior$prior_mean
  mn_prec <- prior$prior_prec
  iw_scale <- prior$prior_scale
  iw_shape <- prior$prior_shape
  # random---------------------------
  if (full) {
    res <- sim_mniw_export(
      1,
      mn_mean, # mean of MN
      solve(mn_prec), # scale of MN = inverse of precision
      iw_scale, # scale of IW
      iw_shape, # shape of IW
      FALSE
    )[[1]]
    res <- list(
      coefficients = res[[1]],
      covmat = res[[2]]
    )
  } else {
    sig <- diag(sigma^2)
    res <- sim_matgaussian(
      mn_mean,
      mn_prec,
      sig,
      TRUE
    )
    res <- list(
      coefficients = res,
      covmat = sig
    )
  }
  res
}
