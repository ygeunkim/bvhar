#' Summarizing Bayesian Multivariate Time Series Model
#' 
#' `summary` method for `normaliw` class.
#' 
#' @param object `normaliw` object
#' @param num_iter Number to sample MNIW distribution
#' @param num_burn Number of burn-in
#' @param thinning Thinning every thinning-th iteration
#' @param ... not used
#' @details 
#' From Minnesota prior, set of coefficient matrices and residual covariance matrix have matrix Normal Inverse-Wishart distribution.
#' 
#' BVAR:
#' 
#' \deqn{(A, \Sigma_e) \sim MNIW(\hat{A}, \hat{V}^{-1}, \hat\Sigma_e, \alpha_0 + n)}
#' where \eqn{\hat{V} = X_\ast^T X_\ast} is the posterior precision of MN.
#' 
#' BVHAR:
#' 
#' \deqn{(\Phi, \Sigma_e) \sim MNIW(\hat\Phi, \hat{V}_H^{-1}, \hat\Sigma_e, \nu + n)}
#' where \eqn{\hat{V}_H = X_{+}^T X_{+}} is the posterior precision of MN.
#' 
#' @return `summary.normaliw` [class] has the following components:
#' \describe{
#'  \item{names}{Variable names}
#'  \item{totobs}{Total number of the observation}
#'  \item{obs}{Sample size used when training = `totobs` - `p`}
#'  \item{p}{Lag of VAR}
#'  \item{m}{Dimension of the data}
#'  \item{call}{Matched call}
#'  \item{spec}{Model specification (`bvharspec`)}
#'  \item{mn_mean}{MN Mean of posterior distribution (MN-IW)}
#'  \item{mn_prec}{MN Precision of posterior distribution (MN-IW)}
#'  \item{iw_scale}{IW scale of posterior distribution (MN-IW)}
#'  \item{iw_shape}{IW df of posterior distribution (MN-IW)}
#'  \item{iter}{Number of MCMC iterations}
#'  \item{burn}{Number of MCMC burn-in}
#'  \item{thin}{MCMC thinning}
#'  \item{alpha_record (BVAR) and phi_record (BVHAR)}{MCMC record of coefficients vector}
#'  \item{psi_record}{MCMC record of upper cholesky factor}
#'  \item{omega_record}{MCMC record of diagonal of cholesky factor}
#'  \item{eta_record}{MCMC record of upper part of cholesky factor}
#'  \item{param}{MCMC record of every parameter}
#'  \item{coefficients}{Posterior mean of coefficients}
#'  \item{covmat}{Posterior mean of covariance}
#' }
#' @references 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25.
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1).
#' @importFrom posterior as_draws_df bind_draws
#' @order 1
#' @export
summary.normaliw <- function(object, num_iter = 10000L, num_burn = floor(num_iter / 2), thinning = 1L, ...) {
  mn_mean <- object$coefficients
  mn_prec <- object$mn_prec
  iw_scale <- object$iw_scale
  nu <- object$iw_shape
  # list of mn and iw-------------------------
  # each simulation is column-stacked
  coef_and_sig <- sim_mniw_export(
    num_iter,
    mn_mean, # mean of MN
    chol2inv(chol(mn_prec)), # precision of MN = inverse of precision
    iw_scale, # scale of IW
    nu # shape of IW
  ) %>%
    simplify2array()
  # preprocess--------------------------------
  dim_design <- object$df # k or h = 3m + 1 or 3m
  dim_data <- ncol(object$y0)
  res <- list(
    names = colnames(object$y0),
    totobs = object$totobs,
    obs = object$obs,
    p = object$p,
    m = object$m,
    call = object$call,
    spec = object$spec,
    # posterior------------
    mn_mean = mn_mean,
    mn_prec = mn_prec,
    iw_scale = iw_scale,
    iw_shape = nu,
    # MCMC-----------------
    iter = num_iter,
    burn = num_burn,
    thin = thinning
  )
  thin_id <- seq(from = num_burn + 1, to = num_iter, by = thinning)
  len_res <- length(thin_id)
  mn_name <- ifelse(grepl(pattern = "^BVAR_", object$process), "alpha", "phi")
  # coef_record <- 
  #   coef_and_sig$mn %>% 
  #   t() %>% 
  #   split.data.frame(gl(num_iter, object$m)) %>% 
  #   lapply(function(x) c(t(x)))
  coef_record <- lapply(coef_and_sig[1,], c)
  coef_record <- coef_record[thin_id]
  coef_record <- do.call(rbind, coef_record)
  colnames(coef_record) <- paste0(mn_name, "[", seq_len(ncol(coef_record)), "]")
  res$coefficients <- 
    colMeans(coef_record) %>% 
    matrix(ncol = object$m)
  # coef_and_sig$iw <- split_psirecord(t(coef_and_sig$iw), chain = 1, varname = "psi")
  coef_and_sig$iw <- coef_and_sig[2,]
  coef_and_sig$iw <- coef_and_sig$iw[thin_id]
  res$psi_record <- lapply(coef_and_sig$iw, function(x) chol2inv(chol(x)))
  res$covmat <- Reduce("+", coef_and_sig$iw) / length(coef_and_sig$iw)
  res$omega_record <- 
    lapply(res$psi_record, diag) %>% 
    do.call(rbind, .)
  colnames(res$omega_record) <- paste0("omega[", seq_len(ncol(res$omega_record)), "]")
  res$omega_record <- as_draws_df(res$omega_record)
  res$eta_record <-
    lapply(res$psi_record, function(x) x[upper.tri(x, diag = FALSE)])
  res$eta_record <- do.call(rbind, res$eta_record)
  colnames(res$eta_record) <- paste0("eta[", seq_len(ncol(res$eta_record)), "]")
  res$eta_record <- as_draws_df(res$eta_record)
  rownames(res$coefficients) <- rownames(object$coefficients)
  colnames(res$coefficients) <- colnames(object$coefficients)
  rownames(res$covmat) <- rownames(object$iw_scale)
  colnames(res$covmat) <- colnames(object$iw_scale)
  if (mn_name == "alpha") {
    res$alpha_record <- coef_record
    res$alpha_record <- as_draws_df(res$alpha_record)
    res$param <- bind_draws(
      res$alpha_record,
      res$omega_record,
      res$eta_record
    )
  } else if (mn_name == "phi") {
    res$phi_record <- coef_record
    res$phi_record <- as_draws_df(res$phi_record)
    res$param <- bind_draws(
      res$phi_record,
      res$omega_record,
      res$eta_record
    )
  }
  class(res) <- "summary.normaliw"
  res
}

