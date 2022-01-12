#' Generate Minnesota BVAR Parameters
#' 
#' This function generates parameters of BVAR with Minnesota prior.
#' 
#' @param p VAR lag
#' @param bayes_spec A BVAR model specification by [set_bvar()].
#' @param full `r lifecycle::badge("experimental")` Generate variance matrix from IW or not?
#' @details 
#' Implementing dummy observation constructions,
#' Bańbura et al. (2010) sets Normal-IW prior.
#' \deqn{A \mid \Sigma_e \sim MN(A_0, \Omega_0, \Sigma_e)}
#' \deqn{\Sigma_e \sim IW(S_0, \alpha_0)}
#' @seealso 
#' * [set_bvar()] to specify the hyperparameters of Minnesota prior.
#' * [bvar_adding_dummy] for dummy observations definition.
#' 
#' @references 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
#' 
#' @export
sim_mncoef <- function(p, bayes_spec = set_bvar(), full = TRUE) {
  # model specification---------------
  if (!is.bvharspec(bayes_spec)) {
    stop("Provide 'bvharspec' for 'bayes_spec'.")
  }
  if (bayes_spec$process != "BVAR") {
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
  # dummy-----------------------------
  Yp <- build_ydummy(p, sigma, lambda, delta)
  Xp <- build_xdummy(p, lambda, sigma, eps)
  num_design <- nrow(Yp)
  dim_design <- ncol(Xp)
  Yp <- Yp[-num_design,]
  Xp <- Xp[-num_design, -dim_design]
  # prior-----------------------------
  prior <- minnesota_prior(Xp, Yp)
  mn_mean <- prior$prior_mean
  mn_prec <- prior$prior_prec
  iw_scale <- prior$prior_scale
  iw_shape <- prior$prior_shape
  # random---------------------------
  if (full) {
    res <- sim_mniw(
      1,
      mn_mean, # mean of MN
      solve(mn_prec), # scale of MN = inverse of precision
      iw_scale, # scale of IW
      iw_shape # shape of IW
    )
    res <- list(
      coefficients = res$mn,
      covmat = res$iw
    )
  } else {
    sig <- diag(sigma^2)
    res <- sim_matgaussian(
      mn_mean,
      solve(mn_prec),
      sig
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
#' @param full `r lifecycle::badge("experimental")` Generate variance matrix from IW or not?
#' @details 
#' Normal-IW family for vector HAR model:
#' \deqn{\Phi \mid \Sigma_e \sim MN(P_0, \Psi_0, \Sigma_e)}
#' \deqn{\Sigma_e \sim IW(U_0, d_0)}
#' @seealso 
#' * [set_bvhar()] to specify the hyperparameters of VAR-type Minnesota prior.
#' * [set_weight_bvhar()] to specify the hyperparameters of HAR-type Minnesota prior.
#' * [bvar_adding_dummy] for dummy observations definition.
#' 
#' @references 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
#' 
#' Baek, C. and Park, M. (2021). *Sparse vector heterogeneous autoregressive modeling for realized volatility*. J. Korean Stat. Soc. 50, 495–510. [https://doi.org/10.1007/s42952-020-00090-5](https://doi.org/10.1007/s42952-020-00090-5)
#' 
#' Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174–196. [https://doi:10.1093/jjfinec/nbp001](https://doi:10.1093/jjfinec/nbp001)
#' 
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
    stop("'sigma' in 'set_bvar()' should be specified. (It is NULL.)")
  }
  sigma <- bayes_spec$sigma
  lambda <- bayes_spec$lambda
  eps <- bayes_spec$eps
  minnesota_type <- bayes_spec$prior
  # dummy-----------------------------
  Yh <- switch(
    minnesota_type,
    "MN_VAR" = {
      if (is.null(bayes_spec$delta)) {
        stop("'delta' in 'set_bvar()' should be specified. (It is NULL.)")
      }
      Yh <- build_ydummy(3, sigma, lambda, bayes_spec$delta)
      Yh
    },
    "MN_VHAR" = {
      if (is.null(bayes_spec$daily)) {
        stop("'daily' in 'set_bvar()' should be specified. (It is NULL.)")
      }
      if (is.null(bayes_spec$weekly)) {
        stop("'weekly' in 'set_bvar()' should be specified. (It is NULL.)")
      }
      if (is.null(bayes_spec$monthly)) {
        stop("'monthly' in 'set_bvar()' should be specified. (It is NULL.)")
      }
      Yh <- build_ydummy_bvhar(
        sigma, 
        lambda, 
        bayes_spec$daily, 
        bayes_spec$weekly, 
        bayes_spec$monthly
      )
      Yh
    }
  )
  Xh <- build_xdummy(3, lambda, sigma, eps)
  num_design <- nrow(Yh)
  dim_design <- ncol(Xh)
  Yh <- Yh[-num_design,]
  Xh <- Xh[-num_design, -dim_design]
  # prior-----------------------------
  prior <- minnesota_prior(Xh, Yh)
  mn_mean <- prior$prior_mean
  mn_prec <- prior$prior_prec
  iw_scale <- prior$prior_scale
  iw_shape <- prior$prior_shape
  # random---------------------------
  if (full) {
    res <- sim_mniw(
      1,
      mn_mean, # mean of MN
      solve(mn_prec), # scale of MN = inverse of precision
      iw_scale, # scale of IW
      iw_shape # shape of IW
    )
    res <- list(
      coefficients = res$mn,
      covmat = res$iw
    )
  } else {
    sig <- diag(sigma^2)
    res <- sim_matgaussian(
      mn_mean,
      solve(mn_prec),
      sig
    )
    res <- list(
      coefficients = res,
      covmat = sig
    )
  }
  res
}
