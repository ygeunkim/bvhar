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
#' * [bvar_adding_dummy] for dummy observations definition.
#' @references 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1).
#' 
#' Karlsson, S. (2013). *Chapter 15 Forecasting with Bayesian Vector Autoregression*. Handbook of Economic Forecasting, 2, 791–897.
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
      iw_shape # shape of IW
    )[[1]]
    res <- list(
      coefficients = res[[1]],
      covmat = res[[2]]
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
#' @param full Generate variance matrix from IW (default: `TRUE`) or not (`FALSE`)?
#' @details 
#' Normal-IW family for vector HAR model:
#' \deqn{\Phi \mid \Sigma_e \sim MN(M_0, \Omega_0, \Sigma_e)}
#' \deqn{\Sigma_e \sim IW(\Psi_0, \nu_0)}
#' @seealso 
#' * [set_bvhar()] to specify the hyperparameters of VAR-type Minnesota prior.
#' * [set_weight_bvhar()] to specify the hyperparameters of HAR-type Minnesota prior.
#' * [bvar_adding_dummy] for dummy observations definition.
#' @return List with the following component.
#' \describe{
#'   \item{coefficients}{BVHAR coefficient (MN)}
#'   \item{covmat}{BVHAR variance (IW or diagonal matrix of `sigma` of `bayes_spec`)}
#' }
#' @references Kim, Y. G., and Baek, C. (n.d.). *Bayesian vector heterogeneous autoregressive modeling*. submitted.
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
      iw_shape # shape of IW
    )[[1]]
    res <- list(
      coefficients = res[[1]],
      covmat = res[[2]]
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

#' Generate SSVS Parameters
#' 
#' This function generates parameters of VAR with SSVS prior.
#' 
#' @param bayes_spec A SSVS model specification by [set_ssvs()].
#' @param p VAR lag
#' @param dim_data Specify the dimension of the data if hyperparameters of `bayes_spec` have constant values.
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param minnesota Only use off-diagonal terms of each coefficient matrices for restriction.
#' In `sim_ssvs_var()` function, use `TRUE` or `FALSE` (default).
#' In `sim_ssvs_vhar()` function, `"no"` (default), `"short"` type, or `"longrun"` type.
#' @param mn_prob Probability for own-lags.
#' @param method Method to compute \eqn{\Sigma^{1/2}}.
#' @section VAR(p) with SSVS prior:
#' Let \eqn{\alpha} be the vectorized coefficient of VAR(p).
#' \deqn{(\alpha \mid \gamma)}
#' \deqn{(\gamma_i)}
#' \deqn{(\eta_j \mid \omega_j)}
#' \deqn{(\omega_{ij})}
#' \deqn{(\psi_{ii}^2)}
#' @return List including coefficients.
#' @references 
#' George, E. I., & McCulloch, R. E. (1993). *Variable Selection via Gibbs Sampling*. Journal of the American Statistical Association, 88(423), 881–889.
#' 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions*. Journal of Econometrics, 142(1), 553–580.
#' 
#' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526).
#' 
#' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358.
#' @importFrom stats rbinom rnorm rgamma
#' @export
sim_ssvs_var <- function(bayes_spec,
                         p,
                         dim_data = NULL,
                         include_mean = TRUE,
                         minnesota = FALSE,
                         mn_prob = 1,
                         method = c("eigen", "chol")) {
  if (!is.ssvsinput(bayes_spec)) {
    stop("Provide 'ssvsinput' for 'bayes_spec'.")
  }
  dim_design <- ifelse(include_mean, dim_data * p + 1, dim_data * p)
  num_coef <- dim_data * dim_design
  num_restrict <- dim_data^2 * p
  num_eta <- dim_data * (dim_data - 1) / 2
  if (length(bayes_spec$coef_spike) == 1) {
    bayes_spec$coef_spike <- rep(bayes_spec$coef_spike, num_restrict)
  }
  if (length(bayes_spec$coef_slab) == 1) {
    bayes_spec$coef_slab <- rep(bayes_spec$coef_slab, num_restrict)
  }
  if (length(bayes_spec$coef_mixture) == 1) {
    bayes_spec$coef_mixture <- rep(bayes_spec$coef_mixture, num_restrict)
  }
  if (length(bayes_spec$shape) == 1) {
    bayes_spec$shape <- rep(bayes_spec$shape, dim_data)
  }
  if (length(bayes_spec$rate) == 1) {
    bayes_spec$rate <- rep(bayes_spec$rate, dim_data)
  }
  if (length(bayes_spec$chol_spike) == 1) {
    bayes_spec$chol_spike <- rep(bayes_spec$chol_spike, num_eta)
  }
  if (length(bayes_spec$chol_slab) == 1) {
    bayes_spec$chol_slab <- rep(bayes_spec$chol_slab, num_eta)
  }
  if (length(bayes_spec$chol_mixture) == 1) {
    bayes_spec$chol_mixture <- rep(bayes_spec$chol_mixture, num_eta)
  }
  if (minnesota) {
    coef_prob <- split.data.frame(matrix(bayes_spec$coef_mixture, ncol = dim_data), gl(p, dim_data))
    diag(coef_prob[[1]]) <- mn_prob
    bayes_spec$coef_mixture <- c(do.call(rbind, coef_prob))
  }
  # dummy for coefficients-------------------------
  coef_dummy <- rbinom(num_restrict, 1, bayes_spec$coef_mixture)
  coef_diag <- diag((1 - coef_dummy) * bayes_spec$coef_spike^2 + coef_dummy * bayes_spec$coef_slab^2)
  coef_gamma <- matrix(coef_dummy, ncol = dim_data)
  # coefficients----------------------------------
  coef_mat <- sim_mnormal(1, rep(0, num_restrict), coef_diag, method = method) * coef_dummy
  coef_mat <- matrix(coef_mat, ncol = dim_data)
  eigen_root <- 
    compute_stablemat(coef_mat) %>% 
    eigen() %>% 
    .$values %>% 
    Mod()
  is_stable <- all(eigen_root < 1)
  if (!is_stable) {
    message("The process is unstable")
  }
  # when including constant term------------------
  if (include_mean) {
    coef_non <- sim_mnormal(1, rep(0, dim_data), bayes_spec$coef_non * diag(dim_data))
    coef_mat <- rbind(coef_mat, coef_non)
    coef_gamma <- rbind(coef_gamma, 1)
  }
  # dummy for cholesky factors--------------------
  chol_dummy <- numeric(dim_data)
  chol_omega <- diag(dim_data)
  chol_diag <- matrix(nrow = dim_data, ncol = dim_data)
  chol_mat <- diag(dim_data)
  diag(chol_mat) <- rgamma(dim_data, shape = bayes_spec$shape, rate = bayes_spec$rate)
  eta_id <- 1
  for (i in 2:dim_data) {
    chol_dummy <- rbinom(i - 1, 1, bayes_spec$chol_mixture)
    chol_omega[1:(i - 1), i] <- chol_dummy
    chol_diag <- (1 - chol_dummy) * bayes_spec$chol_spike[(eta_id):(eta_id + i - 2)]^2 + chol_dummy * bayes_spec$chol_slab[(eta_id):(eta_id + i - 2)]^2
    if (i == 2) {
      chol_mat[1, 2] <- rnorm(1, 0, sqrt(chol_diag)) * chol_dummy
    } else {
      chol_mat[1:(i - 1), i] <- sim_mnormal(1, rep(0, i - 1), diag(chol_diag), method = method) * chol_dummy
    }
    eta_id <- eta_id + (i - 1)
  }
  prec_mat <- chol_mat %*% t(chol_mat)
  # sig_mat <- solve(prec_mat)
  chol_inv <- solve(chol_mat)
  sig_mat <- t(chol_inv) %*% chol_inv
  snr <- sapply(
    1:p,
    function(var_lag) {
      norm(coef_mat[1:(dim_data * p),][((var_lag - 1) * dim_data + 1):(dim_data * var_lag),], type = "F") /
        norm(sig_mat, type = "F")
    }
  )
  res <- list(
    coef = coef_mat,
    root = eigen_root,
    stability = is_stable,
    snr = snr,
    gamma = coef_gamma,
    chol = chol_mat,
    omega = chol_omega,
    prec = prec_mat,
    covmat = sig_mat
  )
  res
}

#' @rdname sim_ssvs_var
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @section VHAR with SSVS prior:
#' Let \eqn{\phi} be the vectorized coefficient of VHAR.
#' \deqn{(\phi \mid \gamma)}
#' \deqn{(\gamma_i)}
#' \deqn{(\eta_j \mid \omega_j)}
#' \deqn{(\omega_{ij})}
#' \deqn{(\psi_{ii}^2)}
#' 
#' @export
sim_ssvs_vhar <- function(bayes_spec,
                          har = c(5, 22),
                          dim_data = NULL,
                          include_mean = TRUE,
                          minnesota = c("no", "short", "longrun"),
                          mn_prob = 1,
                          method = c("eigen", "chol")) {
  if (!is.ssvsinput(bayes_spec)) {
    stop("Provide 'ssvsinput' for 'bayes_spec'.")
  }
  minnesota <- match.arg(minnesota)
  num_har <- ifelse(include_mean, 3 * dim_data + 1, 3 * dim_data)
  num_coef <- dim_data * num_har
  num_restrict <- 3 * dim_data^2
  num_eta <- dim_data * (dim_data - 1) / 2
  if (length(bayes_spec$coef_spike) == 1) {
    bayes_spec$coef_spike <- rep(bayes_spec$coef_spike, num_restrict)
  }
  if (length(bayes_spec$coef_slab) == 1) {
    bayes_spec$coef_slab <- rep(bayes_spec$coef_slab, num_restrict)
  }
  if (length(bayes_spec$coef_mixture) == 1) {
    bayes_spec$coef_mixture <- rep(bayes_spec$coef_mixture, num_restrict)
  }
  if (length(bayes_spec$shape) == 1) {
    bayes_spec$shape <- rep(bayes_spec$shape, dim_data)
  }
  if (length(bayes_spec$rate) == 1) {
    bayes_spec$rate <- rep(bayes_spec$rate, dim_data)
  }
  if (length(bayes_spec$chol_spike) == 1) {
    bayes_spec$chol_spike <- rep(bayes_spec$chol_spike, num_eta)
  }
  if (length(bayes_spec$chol_slab) == 1) {
    bayes_spec$chol_slab <- rep(bayes_spec$chol_slab, num_eta)
  }
  if (length(bayes_spec$chol_mixture) == 1) {
    bayes_spec$chol_mixture <- rep(bayes_spec$chol_mixture, num_eta)
  }
  bayes_spec$coef_mixture <- 
    switch(
      minnesota,
      "no" = bayes_spec$coef_mixture,
      "short" = {
        coef_prob <- split.data.frame(matrix(bayes_spec$coef_mixture, ncol = dim_data), gl(3, dim_data))
        diag(coef_prob[[1]]) <- mn_prob
        c(do.call(rbind, coef_prob))
        
      },
      "longrun" = {
        split.data.frame(matrix(bayes_spec$coef_mixture, ncol = dim_data), gl(3, dim_data)) %>% 
          lapply(
            function(pij) {
              diag(pij) <- mn_prob
              pij
            }
          ) %>% 
          do.call(rbind, .) %>% 
          c()
      }
    )
  # dummy for coefficients-------------------------
  coef_dummy <- rbinom(num_restrict, 1, bayes_spec$coef_mixture)
  coef_diag <- diag((1 - coef_dummy) * bayes_spec$coef_spike^2 + coef_dummy * bayes_spec$coef_slab^2)
  coef_gamma <- matrix(coef_dummy, ncol = dim_data)
  # coefficients----------------------------------
  coef_mat <- sim_mnormal(1, rep(0, num_restrict), coef_diag, method = method) * coef_dummy
  coef_mat <- matrix(coef_mat, ncol = dim_data)
  har_trans <- scale_har(dim_data, har[1], har[2], FALSE)
  eigen_root <-
    compute_stablemat(t(har_trans) %*% coef_mat) %>%
    eigen() %>%
    .$values %>%
    Mod()
  is_stable <- all(eigen_root < 1)
  if (!is_stable) {
    message("Innovations generated by this coefficients matrix might be unstable")
  }
  # when including constant term------------------
  if (include_mean) {
    coef_non <- sim_mnormal(1, rep(0, dim_data), bayes_spec$coef_non * diag(dim_data))
    coef_mat <- rbind(coef_mat, coef_non)
    coef_gamma <- rbind(coef_gamma, 1)
  }
  # dummy for cholesky factors--------------------
  chol_dummy <- numeric(dim_data)
  chol_omega <- diag(dim_data)
  chol_diag <- matrix(nrow = dim_data, ncol = dim_data)
  chol_mat <- diag(dim_data)
  diag(chol_mat) <- rgamma(dim_data, shape = bayes_spec$shape, rate = bayes_spec$rate)
  eta_id <- 1
  for (i in 2:dim_data) {
    chol_dummy <- rbinom(i - 1, 1, bayes_spec$chol_mixture)
    chol_omega[1:(i - 1), i] <- chol_dummy
    chol_diag <- (1 - chol_dummy) * bayes_spec$chol_spike[(eta_id):(eta_id + i - 2)]^2 + chol_dummy * bayes_spec$chol_slab[(eta_id):(eta_id + i - 2)]^2
    if (i == 2) {
      chol_mat[1, 2] <- rnorm(1, 0, sqrt(chol_diag)) * chol_dummy
    } else {
      chol_mat[1:(i - 1), i] <- sim_mnormal(1, rep(0, i - 1), diag(chol_diag), method = method) * chol_dummy
    }
    eta_id <- eta_id + (i - 1)
  }
  prec_mat <- chol_mat %*% t(chol_mat)
  # sig_mat <- solve(prec_mat)
  chol_inv <- solve(chol_mat)
  sig_mat <- t(chol_inv) %*% chol_inv
  res <- list(
    coef = coef_mat,
    root = eigen_root,
    stability = is_stable,
    gamma = coef_gamma,
    chol = chol_mat,
    omega = chol_omega,
    prec = prec_mat,
    covmat = sig_mat
  )
  res
}

#' Generate Horseshoe Parameters
#'
#' This function generates parameters of VAR with Horseshoe prior.
#' 
#' @param p VAR lag
#' @param dim_data Specify the dimension of the data if hyperparameters of `bayes_spec` have constant values.
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param minnesota Only use off-diagonal terms of each coefficient matrices for restriction.
#' In `sim_horseshoe_var()` function, use `TRUE` or `FALSE` (default).
#' In `sim_horseshoe_vhar()` function, `"no"` (default), `"short"` type, or `"longrun"` type.
#' @param method Method to compute \eqn{\Sigma^{1/2}}.
#' @importFrom stats rbinom rnorm rgamma
#' @export
sim_horseshoe_var <- function(p,
                              dim_data = NULL,
                              include_mean = TRUE,
                              minnesota = FALSE,
                              method = c("eigen", "chol")) {
  dim_design <- ifelse(include_mean, dim_data * p + 1, dim_data * p)
  num_coef <- dim_data * dim_design
  num_alpha <- dim_data^2 * p
  num_restrict <- ifelse(
    include_mean,
    num_alpha + dim_data,
    num_alpha
  )
  # Minnesota specification-------------
  glob_idmat <- matrix(1L, nrow = dim_design, ncol = dim_data)
  if (minnesota) {
    if (include_mean) {
      idx <- c(gl(p, dim_data), p + 1)
    } else {
      idx <- gl(p, dim_data)
    }
    glob_idmat <- split.data.frame(
      matrix(rep(0, num_restrict), ncol = dim_data),
      idx
    )
    glob_idmat[[1]] <- diag(dim_data) + 1
    id <- 1
    for (i in 2:p) {
      glob_idmat[[i]] <- matrix(i + 1, nrow = dim_data, ncol = dim_data)
      id <- id + 2
    }
    glob_idmat <- do.call(rbind, glob_idmat)
  }
  grp_id <- unique(c(glob_idmat[1:(dim_data * p), ]))
  latent_local <- numeric(num_restrict)
  latent_global <- numeric(length(grp_id))
  latent_inv_local <- rgamma(num_restrict, shape = 1 / 2, rate = 1)
  latent_inv_global <- rgamma(length(grp_id), shape = 1 / 2, rate = 1)
  local_sparsity <- 1 / rgamma(num_restrict, shape = 1 / 2, scale = latent_inv_local)
  global_sparsity <- 1 / rgamma(length(grp_id), shape = 1 / 2, scale = latent_inv_global)
  shrinkage_vec <- global_sparsity[c(glob_idmat)] * local_sparsity
  coef_mat <-
    sim_mnormal(1, rep(0, num_restrict), diag(shrinkage_vec)) %>%
    matrix(nrow = dim_design, ncol = dim_data)
  eigen_root <-
    compute_stablemat(coef_mat) %>%
    eigen() %>%
    .$values %>%
    Mod()
  is_stable <- all(eigen_root < 1)
  if (!is_stable) {
    message("The process is unstable")
  }
  res <- list(
    coef = coef_mat,
    root = eigen_root,
    stability = is_stable
  )
  res
}

#' @rdname sim_horseshoe_var
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#'
#' @export
sim_horseshoe_vhar <- function(har = c(5, 22),
                               dim_data = NULL,
                               include_mean = TRUE,
                               minnesota = c("no", "short", "longrun"),
                               method = c("eigen", "chol")) {
  dim_har <- ifelse(include_mean, 3 * dim_data + 1, 3 * dim_data)
  num_coef <- dim_data * dim_har
  num_phi <- 3 * dim_data^2
  num_restrict <- ifelse(
    include_mean,
    num_phi + dim_data,
    num_phi
  )
  # num_contem <- dim_data * (dim_data - 1) / 2
  # Minnesota specification------------
  if (include_mean) {
    idx <- c(gl(3, dim_data), 4)
  } else {
    idx <- gl(3, dim_data)
  }
  glob_idmat <- switch(minnesota,
    "no" = matrix(1L, nrow = dim_har, ncol = dim_data),
    "short" = {
      glob_idmat <- split.data.frame(
        matrix(rep(0, num_restrict), ncol = dim_data),
        idx
      )
      glob_idmat[[1]] <- diag(dim_data) + 1
      id <- 1
      for (i in 2:3) {
        glob_idmat[[i]] <- matrix(i + 1, nrow = dim_data, ncol = dim_data)
        id <- id + 2
      }
      do.call(rbind, glob_idmat)
    },
    "longrun" = {
      glob_idmat <- split.data.frame(
        matrix(rep(0, num_restrict), ncol = dim_data),
        idx
      )
      id <- 1
      for (i in 1:3) {
        glob_idmat[[i]] <- diag(dim_data) + id
        id <- id + 2
      }
      do.call(rbind, glob_idmat)
    }
  )
  grp_id <- unique(c(glob_idmat[1:(dim_data * 3), ]))
  latent_local <- numeric(num_restrict)
  latent_global <- numeric(length(grp_id))
  latent_inv_local <- rgamma(num_restrict, shape = 1 / 2, rate = 1)
  latent_inv_global <- rgamma(length(grp_id), shape = 1 / 2, rate = 1)
  local_sparsity <- 1 / rgamma(num_restrict, shape = 1 / 2, scale = latent_inv_local)
  global_sparsity <- 1 / rgamma(length(grp_id), shape = 1 / 2, scale = latent_inv_global)
  shrinkage_vec <- global_sparsity[c(glob_idmat)] * local_sparsity
  coef_mat <-
    sim_mnormal(1, rep(0, num_restrict), diag(shrinkage_vec)) %>%
    matrix(nrow = dim_har, ncol = dim_data)
  har_trans <- scale_har(dim_data, har[1], har[2], FALSE)
  eigen_root <-
    compute_stablemat(t(har_trans) %*% coef_mat) %>%
    eigen() %>%
    .$values %>%
    Mod()
  is_stable <- all(eigen_root < 1)
  if (!is_stable) {
    message("Innovations generated by this coefficients matrix might be unstable")
  }
  res <- list(
    coef = coef_mat,
    root = eigen_root,
    stability = is_stable
  )
  res
  # contem_local_sparsity <- 1 / rgamma(
  #   num_contem,
  #   shape = 1 / 2,
  #   scale = rgamma(num_contem, shape = 1 / 2, rate = 1)
  # )
  # contem_global_sparsity <- 1 / rgamma(
  #   1,
  #   shape = 1 / 2,
  #   scale = rgamma(1, shape = 1 / 2, rate = 1)
  # )
  # contem_shrink_vec <- contem_global_sparsity * contem_local_sparsity
  # contem_coef <-
  #   sim_mnormal(1, rep(0, num_contem), diag(contem_shrink_vec))
  # lower_mat <- diag(dim_data)
  # lower_mat[lower.tri(lower_mat, diag = FALSE)] <- contem_coef
}
