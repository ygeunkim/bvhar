#' Fitting Bayesian VHAR of Minnesota Prior
#' 
#' This function fits BVHAR with Minnesota prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec A BVHAR model specification by [set_bvhar()] (default) or [set_weight_bvhar()].
#' @param scale_variance Proposal distribution scaling constant to adjust an acceptance rate
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param parallel List the same argument of [optimParallel::optimParallel()]. By default, this is empty, and the function does not execute parallel computation.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @details 
#' Apply Minnesota prior to Vector HAR: \eqn{\Phi} (VHAR matrices) and \eqn{\Sigma_e} (residual covariance).
#' 
#' \deqn{\Phi \mid \Sigma_e \sim MN(M_0, \Omega_0, \Sigma_e)}
#' \deqn{\Sigma_e \sim IW(\Psi_0, \nu_0)}
#' (MN: [matrix normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution), IW: [inverse-wishart](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution))
#' 
#' There are two types of Minnesota priors for BVHAR:
#' 
#' * VAR-type Minnesota prior specified by [set_bvhar()], so-called BVHAR-S model.
#' * VHAR-type Minnesota prior specified by [set_weight_bvhar()], so-called BVHAR-L model.
#' 
#' Two types of Minnesota priors builds different dummy variables for Y0.
#' See [var_design_formulation].
#' @return `bvhar_minnesota()` returns an object `bvharmn` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Posterior Mean matrix of Matrix Normal distribution}
#'   \item{fitted.values}{Fitted values}
#'   \item{residuals}{Residuals}
#'   \item{mn_prec}{Posterior precision matrix of Matrix Normal distribution}
#'   \item{iw_scale}{Posterior scale matrix of posterior inverse-wishart distribution}
#'   \item{iw_shape}{Posterior shape of inverse-Wishart distribution (\eqn{\nu_0} - obs + 2). \eqn{\nu_0}: nrow(Dummy observation) - k}
#'   \item{df}{Numer of Coefficients: 3m + 1 or 3m}
#'   \item{p}{3, this element exists to run the other functions}
#'   \item{week}{Order for weekly term}
#'   \item{month}{Order for monthly term}
#'   \item{m}{Dimension of the time series}
#'   \item{obs}{Sample size used when training = `totobs` - 22}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Process string in the `bayes_spec`: `"BVHAR_MN_VAR"` (BVHAR-S) or `"BVHAR_MN_VHAR"` (BVHAR-L)}
#'   \item{spec}{Model specification (`bvharspec`)}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{prior_mean}{Prior mean matrix of Matrix Normal distribution: \eqn{M_0}}
#'   \item{prior_precision}{Prior precision matrix of Matrix Normal distribution: \eqn{\Omega_0^{-1}}}
#'   \item{prior_scale}{Prior scale matrix of inverse-Wishart distribution: \eqn{\Psi_0}}
#'   \item{prior_shape}{Prior shape of inverse-Wishart distribution: \eqn{\nu_0}}
#'   \item{HARtrans}{VHAR linear transformation matrix: \eqn{C_{HAR}}}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input (`matrix`)}
#' }
#' It is also `normaliw` and `bvharmod` class.
#' @references Kim, Y. G., and Baek, C. (2023+). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation.
#' @seealso 
#' * [set_bvhar()] to specify the hyperparameters of BVHAR-S
#' * [set_weight_bvhar()] to specify the hyperparameters of BVHAR-L
#' * [coef.bvharmn()], [residuals.bvharmn()], and [fitted.bvharmn()]
#' * [summary.normaliw()] to summarize BVHAR model
#' * [predict.bvharmn()] to forecast the BVHAR process
#' @examples
#' # Perform the function using etf_vix dataset
#' fit <- bvhar_minnesota(y = etf_vix[,1:3])
#' class(fit)
#' 
#' # Extract coef, fitted values, and residuals
#' coef(fit)
#' head(residuals(fit))
#' head(fitted(fit))
#' @importFrom stats sd optim
#' @importFrom optimParallel optimParallel
#' @importFrom posterior as_draws_df bind_draws
#' @order 1
#' @export
bvhar_minnesota <- function(y,
                            har = c(5, 22),
                            num_iter = 1000,
                            num_burn = floor(num_iter / 2),
                            thinning = 1,
                            bayes_spec = set_bvhar(),
                            scale_variance = .05,
                            include_mean = TRUE,
                            parallel = list(),
                            verbose = FALSE) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  if (!is.bvharspec(bayes_spec)) {
    stop("Provide 'bvharspec' for 'bayes_spec'.")
  }
  if (bayes_spec$process != "BVHAR") {
    stop("'bayes_spec' must be the result of 'set_bvhar()' or 'set_weight_bvhar()'.")
  }
  minnesota_type <- bayes_spec$prior
  if (!(
    minnesota_type == "MN_VAR"
    || minnesota_type == "HMN_VAR"
    || minnesota_type == "MN_VHAR"
  )) {
    stop("Provide Minnesota prior specification.")
  }
  if (length(har) != 2 || !is.numeric(har)) {
    stop("'har' should be numeric vector of length 2.")
  }
  if (har[1] > har[2]) {
    stop("'har[1]' should be smaller than 'har[2]'.")
  }
  week <- har[1] # 5
  month <- har[2] # 22
  dim_data <- ncol(y)
  # N <- nrow(y)
  # num_coef <- 3 * dim_data + 1
  # Y0 = X0 A + Z---------------------
  Y0 <- build_y0(y, month, month + 1)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
  }
  colnames(Y0) <- name_var
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  X0 <- build_design(y, month, include_mean)
  HARtrans <- scale_har(dim_data, week, month, include_mean)
  name_har <- concatenate_colnames(name_var, c("day", "week", "month"), include_mean) # in misc-r.R file
  X1 <- X0 %*% t(HARtrans)
  colnames(X1) <- name_har
  # model specification---------------
  if (bayes_spec$prior == "HMN_VAR") {
    psi <- bayes_spec$sigma$mode
    psi <- rep(psi, dim_data)
    lambda <- bayes_spec$lambda$mode
  } else {
    if (is.null(bayes_spec$sigma)) {
      bayes_spec$sigma <- apply(y, 2, sd)
    }
    sigma <- bayes_spec$sigma
    lambda <- bayes_spec$lambda
  }
  # if (is.null(bayes_spec$sigma)) {
  #   bayes_spec$sigma <- apply(y, 2, sd)
  # }
  # sigma <- bayes_spec$sigma
  # lambda <- bayes_spec$lambda
  eps <- bayes_spec$eps
  # dummy-----------------------------
  # Yh <- switch(
  #   minnesota_type,
  #   "MN_VAR" = {
  #     if (is.null(bayes_spec$delta)) {
  #       bayes_spec$delta <- rep(1, dim_data)
  #     }
  #     Yh <- build_ydummy(3, sigma, lambda, bayes_spec$delta, numeric(dim_data), numeric(dim_data), include_mean)
  #     colnames(Yh) <- name_var
  #     Yh
  #   },
  #   "MN_VHAR" = {
  #     if (is.null(bayes_spec$daily)) {
  #       bayes_spec$daily <- rep(1, dim_data)
  #     }
  #     if (is.null(bayes_spec$weekly)) {
  #       bayes_spec$weekly <- rep(1, dim_data)
  #     }
  #     if (is.null(bayes_spec$monthly)) {
  #       bayes_spec$monthly <- rep(1, dim_data)
  #     }
  #     Yh <- build_ydummy(
  #       3,
  #       sigma, 
  #       lambda, 
  #       bayes_spec$daily, 
  #       bayes_spec$weekly, 
  #       bayes_spec$monthly,
  #       include_mean
  #     )
  #     colnames(Yh) <- name_var
  #     Yh
  #   }
  # )
  # Xh <- build_xdummy(1:3, lambda, sigma, eps, include_mean)
  # colnames(Xh) <- name_har
  # estimate-bvar.cpp-----------------
  # posterior <- estimate_bvar_mn(X1, Y0, Xh, Yh)
  posterior <- switch(
    minnesota_type,
    "MN_VAR" = {
      if (is.null(bayes_spec$delta)) {
        bayes_spec$delta <- rep(1, dim_data)
      }
      Yh <- build_ydummy(3, sigma, lambda, bayes_spec$delta, numeric(dim_data), numeric(dim_data), include_mean)
      colnames(Yh) <- name_var
      Xh <- build_xdummy(1:3, lambda, sigma, eps, include_mean)
      colnames(Xh) <- name_har
      estimate_bvar_mn(X1, Y0, Xh, Yh)
    },
    "MN_VHAR" = {
      if (is.null(bayes_spec$daily)) {
        bayes_spec$daily <- rep(1, dim_data)
      }
      if (is.null(bayes_spec$weekly)) {
        bayes_spec$weekly <- rep(1, dim_data)
      }
      if (is.null(bayes_spec$monthly)) {
        bayes_spec$monthly <- rep(1, dim_data)
      }
      Yh <- build_ydummy(
        3,
        sigma,
        lambda,
        bayes_spec$daily,
        bayes_spec$weekly,
        bayes_spec$monthly,
        include_mean
      )
      colnames(Yh) <- name_var
      estimate_bvar_mn(X1, Y0, Xh, Yh)
    },
    "HMN_VAR" = { 
      if (is.null(bayes_spec$delta)) {
        bayes_spec$delta <- rep(1, dim_data)
      }
      delta <- bayes_spec$delta
      # prior selection-------------------
      lower_vec <- unlist(bayes_spec)
      lower_vec <- as.numeric(lower_vec[grepl(pattern = "lower$", x = names(unlist(bayes_spec)))])
      upper_vec <- unlist(bayes_spec)
      upper_vec <- as.numeric(upper_vec[grepl(pattern = "upper$", x = names(unlist(bayes_spec)))])
      if (length(parallel) > 0) {
        init_par <-
          optimParallel(
            par = c(lambda, psi),
            fn = logml_bvharhm,
            lower = lower_vec,
            upper = upper_vec,
            hessian = TRUE,
            delta = delta,
            eps = bayes_spec$eps,
            y = y,
            har = har,
            include_mean = include_mean,
            parallel = parallel
          )
      } else {
        init_par <-
          optim(
            par = c(lambda, psi),
            fn = logml_bvharhm,
            method = "L-BFGS-B",
            lower = lower_vec,
            upper = upper_vec,
            hessian = TRUE,
            delta = delta,
            eps = bayes_spec$eps,
            y = y,
            har = har,
            include_mean = include_mean
          )
      }
      lambda <- init_par$par[1]
      psi <- init_par$par[2:(1 + dim_data)]
      hess <- init_par$hessian
      Yh <- build_ydummy(3, psi, lambda, delta, numeric(dim_data), numeric(dim_data), include_mean)
      colnames(Yh) <- name_var
      Xh <- build_xdummy(1:3, lambda, psi, eps, include_mean)
      colnames(Xh) <- name_har
      estimate_bvar_mn(X1, Y0, Xh, Yh)
    }
  )
  # Prior-----------------------------
  prior_mean <- posterior$prior_mean
  prior_prec <- posterior$prior_prec
  prior_scale <- posterior$prior_scale
  prior_shape <- posterior$prior_shape
  # Matrix normal---------------------
  mn_mean <- posterior$mnmean # posterior mean
  colnames(mn_mean) <- name_var
  rownames(mn_mean) <- name_har
  mn_prec <- posterior$mnprec
  colnames(mn_prec) <- name_har
  rownames(mn_prec) <- name_har
  yhat <- posterior$fitted
  colnames(yhat) <- name_var
  # Inverse-wishart-------------------
  iw_scale <- posterior$iwscale
  colnames(iw_scale) <- name_var
  rownames(iw_scale) <- name_var
  iw_shape <- prior_shape + nrow(Y0)
  # S3--------------------------------
  if (minnesota_type == "HMN_VAR") {
    res <- estimate_hierachical_niw(
      num_iter = num_iter,
      num_burn = num_burn,
      x = X1,
      y = Y0,
      prior_prec = prior_prec,
      prior_scale = prior_scale,
      prior_shape = prior_shape,
      mn_mean = mn_mean,
      mn_prec = mn_prec,
      iw_scale = iw_scale,
      posterior_shape = iw_shape,
      gamma_shp = bayes_spec$lambda$param[1],
      gamma_rate = bayes_spec$lambda$param[2],
      invgam_shp = bayes_spec$sigma$param[1],
      invgam_scl = bayes_spec$sigma$param[2],
      acc_scale = scale_variance,
      obs_information = hess,
      init_lambda = lambda,
      init_psi = psi,
      display_progress = verbose
    )
    # preprocess the results------------
    names(res) <- gsub(pattern = "^alpha", replacement = "phi", x = names(res)) # alpha to phi
    thin_id <- seq(from = 1, to = num_iter - num_burn, by = thinning)
    # thinparam_id <- seq(from = 1, to = num_iter - 1 - num_burn, by = thinning)
    thinparam_id <- seq(from = 1, to = num_iter - num_burn, by = thinning)
    res$psi_record <- res$psi_record[thin_id, ]
    res$phi_record <- res$phi_record[thinparam_id, ]
    # hyperparameters------------------
    colnames(res$psi_record) <- paste0("psi[", seq_len(ncol(res$psi_record)), "]")
    res$lambda_record <- as.matrix(res$lambda_record[thin_id])
    colnames(res$lambda_record) <- "lambda"
    res$psi_record <- as_draws_df(res$psi_record)
    res$lambda_record <- as_draws_df(res$lambda_record)
    # parameters-----------------------
    colnames(res$phi_record) <- paste0("alpha[", seq_len(ncol(res$phi_record)), "]")
    res$coefficients <-
      colMeans(res$phi_record) %>%
      matrix(ncol = dim_data)
    colnames(res$coefficients) <- name_var
    rownames(res$coefficients) <- name_har
    res$phi_record <- as_draws_df(res$phi_record)
    # posterior mean-------------------
    res$covmat <- Reduce("+", split_psirecord(res$sigma_record, 1, "sigma")) / length(thinparam_id)
    colnames(res$covmat) <- name_var
    rownames(res$covmat) <- name_var
    res$sigma_record <-
      t(res$sigma_record) %>%
      matrix(ncol = 1) %>%
      split.data.frame(gl(dim_data^2, 1, nrow(res$sigma_record) * dim_data)) %>%
      lapply(function(x) x[thinparam_id, ])
    names(res$sigma_record) <- paste0("sigma[", 1:(dim_data^2), "]")
    res$sigma_record <- as_draws_df(res$sigma_record)
    # acceptance rate------------------
    res$acc_rate <- mean(res$acceptance) # change to matrix and define in chain > 1 later
    res$hyperparam <- bind_draws(
      res$lambda_record,
      res$psi_record
    )
    res$param <- bind_draws(
      res$phi_record,
      res$sigma_record
    )
    res$iter <- num_iter
    res$burn <- num_burn
    res$thin <- thinning
  } else {
    res <- list(
      # posterior-----------
      coefficients = mn_mean,
      fitted.values = yhat,
      residuals = posterior$residuals,
      mn_prec = mn_prec,
      iw_scale = iw_scale,
      iw_shape = iw_shape, # if adding improper prior, d0 + n + 2
      # prior----------------
      prior_mean = prior_mean,
      prior_precision = prior_prec,
      prior_scale = prior_scale,
      prior_shape = prior_shape
    )
  }
  # res <- list(
  #   # posterior-----------
  #   coefficients = mn_mean,
  #   fitted.values = yhat,
  #   residuals = posterior$residuals,
  #   mn_prec = mn_prec,
  #   iw_scale = iw_scale,
  #   iw_shape = prior_shape + nrow(Y0), # if adding improper prior, d0 + s + 2
  #   # variables-----------
  #   df = nrow(mn_mean), # 3 * m + 1 or 3 * m
  #   p = 3, # add for other function (df = 3m + 1 = mp + 1)
  #   week = week, # default: 5
  #   month = month, # default: 22
  #   m = dim_data, # m
  #   obs = nrow(Y0), # n = T - 22
  #   totobs = nrow(y), # T
  #   # about model---------
  #   call = match.call(),
  #   process = paste(bayes_spec$process, minnesota_type, sep = "_"),
  #   spec = bayes_spec,
  #   type = ifelse(include_mean, "const", "none"),
  #   # prior----------------
  #   prior_mean = prior_mean,
  #   prior_precision = prior_prec,
  #   prior_scale = prior_scale,
  #   prior_shape = prior_shape,
  #   # data----------------
  #   HARtrans = HARtrans,
  #   y0 = Y0,
  #   design = X0,
  #   y = y
  # )
  # variables-------------------------
  res$df <- nrow(mn_mean)
  res$p <- 3
  res$m <- dim_data
  res$obs <- nrow(Y0)
  res$totobs <- nrow(y)
  # model-----------------------------
  res$call <- match.call()
  res$process <- paste(bayes_spec$process, minnesota_type, sep = "_")
  res$type <- ifelse(include_mean, "const", "none")
  res$spec <- bayes_spec
  # data------------------------------
  res$HARtrans <- HARtrans
  res$y0 <- Y0
  res$design <- X0
  res$y <- y
  if (minnesota_type == "HMN_VAR") {
    class(res) <- c("bvharhm", "hmnmod", "bvharsp")
  } else {
    class(res) <- c("bvharmn", "normaliw", "bvharmod")
  }
  res
}