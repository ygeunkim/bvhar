#' Summarizing Bayesian Multivariate Time Series Model
#' 
#' `summary` method for `normaliw` class.
#' 
#' @param object A `normaliw` object
#' @param num_chains Number of MCMC chains
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @param num_thread Number of threads
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
summary.normaliw <- function(object, num_chains = 1, num_iter = 1000,  num_burn = floor(num_iter / 2), thinning = 1, verbose = FALSE, num_thread = 1, ...) {
  # mn_mean <- object$coefficients
  # mn_prec <- object$mn_prec
  # iw_scale <- object$covmat
  # nu <- object$iw_shape
  # cred_int <- compute_ci(subset_draws(object$param, variable = "alpha|phi", regex = TRUE), level = level)
  # selection <- matrix(cred_int$conf.low * cred_int$conf.high >= 0, ncol = object$m)
  # if (object$type == "const") {
  #   cred_int_const <- compute_ci(object$c_record, level = level)
  #   selection <- rbind(
  #     selection,
  #     cred_int_const$conf.low * cred_int_const$conf.high >= 0
  #   )
  # }
  # rownames(selection) <- rownames(object$coefficients)
  # colnames(selection) <- colnames(object$coefficients)
  # coef_res <- selection * object$coefficients
  # rownames(coef_res) <- rownames(object$coefficients)
  # colnames(coef_res) <- colnames(object$coefficients)

  # int num_chains, int num_iter, int num_burn, int thin,
  # 											 const Eigen::MatrixXd& mn_mean, const Eigen::MatrixXd& mn_prec,
  # 											 const Eigen::MatrixXd& iw_scale, double iw_shape,
  # 											 Eigen::VectorXi seed_chain, bool display_progress, int nthreads
  res <- estimate_mniw(
    num_chains = num_chains, num_iter = num_iter, thin = thinning,
    mn_mean = object$coefficients, mn_prec = object$mn_prec,
    iw_scale = object$covmat, iw_shape = object$iw_shape,
    seed_chain = sample.int(.Machine$integer.max, size = num_chains),
    nthreads = num_thread
  )
  res <- do.call(rbind, res)
  # # list of mn and iw-------------------------
  # # each simulation is column-stacked
  # coef_and_sig <- sim_mniw_export(
  #   num_iter,
  #   mn_mean, # mean of MN
  #   mn_prec, # precision of MN = inverse of precision
  #   iw_scale, # scale of IW
  #   nu, # shape of IW
  #   TRUE
  # ) |>
  #   simplify2array()
  # # preprocess--------------------------------
  # dim_design <- object$df # k or h = 3m + 1 or 3m
  # dim_data <- ncol(object$y0)
  # res <- list(
  #   names = colnames(object$y0),
  #   totobs = object$totobs,
  #   obs = object$obs,
  #   p = object$p,
  #   m = object$m,
  #   call = object$call,
  #   spec = object$spec,
  #   # posterior------------
  #   mn_mean = mn_mean,
  #   mn_prec = mn_prec,
  #   iw_scale = iw_scale,
  #   iw_shape = nu,
  #   # MCMC-----------------
  #   iter = num_iter,
  #   burn = num_burn,
  #   thin = thinning
  # )
  # thin_id <- seq(from = num_burn + 1, to = num_iter, by = thinning)
  # len_res <- length(thin_id)
  mn_name <- ifelse(grepl(pattern = "^BVAR_", object$process), "alpha", "phi")
  # # coef_record <-
  # #   coef_and_sig$mn |>
  # #   t() |>
  # #   split.data.frame(gl(num_iter, object$m)) |>
  # #   lapply(function(x) c(t(x)))
  # coef_record <- lapply(coef_and_sig[1,], c)
  # coef_record <- coef_record[thin_id]
  # coef_record <- do.call(rbind, coef_record)
  # colnames(coef_record) <- paste0(mn_name, "[", seq_len(ncol(coef_record)), "]")
  # res$coefficients <-
  #   colMeans(coef_record) |>
  #   matrix(ncol = object$m)
  # # coef_and_sig$iw <- split_psirecord(t(coef_and_sig$iw), chain = 1, varname = "psi")
  # coef_and_sig$iw <- coef_and_sig[2,]
  # coef_and_sig$iw <- coef_and_sig$iw[thin_id]
  # res$cov_record <- coef_and_sig$iw
  # res$cov_record <- lapply(
  #   coef_and_sig$iw,
  #   function(x) {
  #     rownames(x) <- rownames(object$iw_scale)
  #     colnames(x) <- colnames(object$iw_scale)
  #     x
  #   }
  # )
  # prec_record <- lapply(coef_and_sig$iw, function(x) chol2inv(chol(x)))
  # res$covmat <- Reduce("+", coef_and_sig$iw) / length(coef_and_sig$iw)
  # res$omega_record <-
  #   lapply(prec_record, diag) |>
  #   do.call(rbind, .)
  # colnames(res$omega_record) <- paste0("omega[", seq_len(ncol(res$omega_record)), "]")
  # res$omega_record <- as_draws_df(res$omega_record)
  # res$eta_record <-
  #   lapply(prec_record, function(x) x[upper.tri(x, diag = FALSE)])
  # res$eta_record <- do.call(rbind, res$eta_record)
  # colnames(res$eta_record) <- paste0("eta[", seq_len(ncol(res$eta_record)), "]")
  # res$eta_record <- as_draws_df(res$eta_record)
  # rownames(res$coefficients) <- rownames(object$coefficients)
  # colnames(res$coefficients) <- colnames(object$coefficients)
  # rownames(res$covmat) <- rownames(object$iw_scale)
  # colnames(res$covmat) <- colnames(object$iw_scale)
  # if (mn_name == "alpha") {
    # res$alpha_record <- coef_record
    # res$alpha_record <- as_draws_df(res$alpha_record)
    # res$param <- bind_draws(
    #   res$alpha_record,
    #   res$omega_record,
    #   res$eta_record
    # )
  # } else if (mn_name == "phi") {
    # res$phi_record <- coef_record
    # res$phi_record <- as_draws_df(res$phi_record)
    # res$param <- bind_draws(
    #   res$phi_record,
    #   res$omega_record,
    #   res$eta_record
    # )
  # }
  dim_data <- object$m
  res$coefficients <- matrix(colMeans(res$alpha_record), ncol = dim_data)
  res$covmat <- matrix(colMeans(res$sigma_record), ncol = dim_data)
  rownames(res$coefficients) <- rownames(object$coefficients)
  colnames(res$coefficients) <- colnames(object$coefficients)
  rownames(res$covmat) <- rownames(object$covmat)
  colnames(res$covmat) <- colnames(object$covmat)
  if (mn_name == "phi") {
    colnames(res) <- gsub(pattern = "^alpha", replacement = "phi", x = colnames(res)) # alpha to phi
  }
  rec_names <- colnames(res)
  param_names <- gsub(pattern = "_record$", replacement = "", rec_names)
  # res <- apply(res, 2, function(x) do.call(rbind, x))
  res <- apply(
    res,
    2,
    function(x) {
      if (is.vector(x[[1]])) {
        return(as.matrix(unlist(x)))
      }
      do.call(rbind, x)
    }
  )
  names(res) <- rec_names
  res$totobs <- object$totobs
  res$obs <- object$obs
  res$p <- object$p
  res$m <- object$m
  res$call <- object$call
  res$spec <- object$spec
  res$mn_mean <- object$coefficients
  res$mn_prec <- object$mn_prec
  res$iw_scale <- object$covmat
  res$iw_shape <- object$iw_shape
  res$iter <- num_iter
  res$burn <- num_burn
  res$thin <- thinning
  # res <- list(
  #   call = object$call,
  #   process = object$process,
  #   p = object$p,
  #   m = object$m,
  #   type = object$type,
  #   coefficients = coef_res,
  #   posterior_mean = object$coefficients,
  #   choose_coef = selection,
  #   method = "ci"
  # )
  class(res) <- c("summary.normaliw", "summary.bvharsp")
  res
}

