#' Fitting Bayesian VAR(p) of Minnesota Prior
#' 
#' This function fits BVAR(p) with Minnesota prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag (Default: 1)
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec A BVAR model specification by [set_bvar()].
#' @param scale_variance Proposal distribution scaling constant to adjust an acceptance rate
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param parallel List the same argument of [optimParallel::optimParallel()]. By default, this is empty, and the function does not execute parallel computation.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @details 
#' Minnesota prior gives prior to parameters \eqn{A} (VAR matrices) and \eqn{\Sigma_e} (residual covariance).
#' 
#' \deqn{A \mid \Sigma_e \sim MN(A_0, \Omega_0, \Sigma_e)}
#' \deqn{\Sigma_e \sim IW(S_0, \alpha_0)}
#' (MN: [matrix normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution), IW: [inverse-wishart](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution))
#' @return `bvar_minnesota()` returns an object `bvarmn` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Posterior Mean matrix of Matrix Normal distribution}
#'   \item{fitted.values}{Fitted values}
#'   \item{residuals}{Residuals}
#'   \item{mn_prec}{Posterior precision matrix of Matrix Normal distribution}
#'   \item{iw_scale}{Posterior scale matrix of posterior inverse-Wishart distribution}
#'   \item{iw_shape}{Posterior shape of inverse-Wishart distribution (\eqn{alpha_0} - obs + 2). \eqn{\alpha_0}: nrow(Dummy observation) - k}
#'   \item{df}{Numer of Coefficients: mp + 1 or mp}
#'   \item{p}{Lag of VAR}
#'   \item{m}{Dimension of the time series}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Process string in the `bayes_spec`: `"BVAR_Minnesota"`}
#'   \item{spec}{Model specification (`bvharspec`)}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{prior_mean}{Prior mean matrix of Matrix Normal distribution: \eqn{A_0}}
#'   \item{prior_precision}{Prior precision matrix of Matrix Normal distribution: \eqn{\Omega_0^{-1}}}
#'   \item{prior_scale}{Prior scale matrix of inverse-Wishart distribution: \eqn{S_0}}
#'   \item{prior_shape}{Prior shape of inverse-Wishart distribution: \eqn{\alpha_0}}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input (`matrix`)}
#' }
#' It is also `normaliw` and `bvharmod` class.
#' @references 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1).
#' 
#' Giannone, D., Lenza, M., & Primiceri, G. E. (2015). *Prior Selection for Vector Autoregressions*. Review of Economics and Statistics, 97(2).
#' 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25.
#' 
#' KADIYALA, K.R. and KARLSSON, S. (1997), *NUMERICAL METHODS FOR ESTIMATION AND INFERENCE IN BAYESIAN VAR-MODELS*. J. Appl. Econ., 12: 99-132.
#' 
#' Karlsson, S. (2013). *Chapter 15 Forecasting with Bayesian Vector Autoregression*. Handbook of Economic Forecasting, 2, 791–897.
#' 
#' Sims, C. A., & Zha, T. (1998). *Bayesian Methods for Dynamic Multivariate Models*. International Economic Review, 39(4), 949–968.
#' @seealso 
#' * [set_bvar()] to specify the hyperparameters of Minnesota prior.
#' * [coef.bvarmn()], [residuals.bvarmn()], and [fitted.bvarmn()]
#' * [summary.normaliw()] to summarize BVAR model
#' * [predict.bvarmn()] to forecast the BVAR process
#' @examples
#' # Perform the function using etf_vix dataset
#' fit <- bvar_minnesota(y = etf_vix[,1:3], p = 2)
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
bvar_minnesota <- function(y,
                           p = 1,
                           num_iter = 1000, 
                           num_burn = floor(num_iter / 2),
                           thinning = 1,
                           bayes_spec = set_bvar(),
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
  dim_data <- ncol(y)
  # model specification---------------
  if (!is.bvharspec(bayes_spec)) {
    stop("Provide 'bvharspec' for 'bayes_spec'.")
  }
  if (bayes_spec$process != "BVAR") {
    stop("'bayes_spec' must be the result of 'set_bvar()'.")
  }
  # if (bayes_spec$prior != "Minnesota") {
  #   stop("In 'set_bvar()', just input numeric values.")
  # }
  if (!(bayes_spec$prior == "Minnesota" || bayes_spec$prior == "MN_Hierarchical")) {
    stop("Provide Minnesota prior specification.")
  }
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, p, p + 1)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
  }
  colnames(Y0) <- name_var
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  X0 <- build_design(y, p, include_mean)
  name_lag <- concatenate_colnames(name_var, 1:p, include_mean) # in misc-r.R file
  colnames(X0) <- name_lag
  # s <- nrow(Y0)
  # k <- m * p + 1
  # m <- ncol(y)
  bvar_est <- switch(
    bayes_spec$prior,
    "Minnesota" = {
      # 
      if (is.null(bayes_spec$sigma)) {
        bayes_spec$sigma <- apply(y, 2, sd)
      }
      sigma <- bayes_spec$sigma
      # m <- ncol(y)
      if (is.null(bayes_spec$delta)) {
        bayes_spec$delta <- rep(1, dim_data)
      }
      delta <- bayes_spec$delta
      lambda <- bayes_spec$lambda
      eps <- bayes_spec$eps
      # dummy-----------------------------
      Yp <- build_ydummy(p, sigma, lambda, delta, numeric(dim_data), numeric(dim_data), include_mean)
      colnames(Yp) <- name_var
      Xp <- build_xdummy(1:p, lambda, sigma, eps, include_mean)
      colnames(Xp) <- name_lag
      estimate_bvar_mn(X0, Y0, Xp, Yp)
      
    },
    "MN_Hierarchical" = {
      psi <- bayes_spec$sigma$mode
      psi <- rep(psi, dim_data)
      if (is.null(bayes_spec$delta)) {
        bayes_spec$delta <- rep(1, dim_data)
      }
      delta <- bayes_spec$delta
      lambda <- bayes_spec$lambda$mode
      eps <- bayes_spec$eps
      # prior selection-------------------
      lower_vec <- unlist(bayes_spec)
      lower_vec <- as.numeric(lower_vec[grepl(pattern = "lower$", x = names(unlist(bayes_spec)))])
      upper_vec <- unlist(bayes_spec)
      upper_vec <- as.numeric(upper_vec[grepl(pattern = "upper$", x = names(unlist(bayes_spec)))])
      if (length(parallel) > 0) {
        init_par <-
          optimParallel(
            par = c(lambda, psi),
            fn = logml_bvarhm,
            lower = lower_vec,
            upper = upper_vec,
            hessian = TRUE,
            delta = delta,
            eps = bayes_spec$eps,
            y = y,
            p = p,
            include_mean = include_mean,
            parallel = parallel
          )
      } else {
        init_par <-
          optim(
            par = c(lambda, psi),
            fn = logml_bvarhm,
            method = "L-BFGS-B",
            lower = lower_vec,
            upper = upper_vec,
            hessian = TRUE,
            delta = delta,
            eps = bayes_spec$eps,
            y = y,
            p = p,
            include_mean = include_mean
          )
      }
      lambda <- init_par$par[1]
      psi <- init_par$par[2:(1 + dim_data)]
      hess <- init_par$hessian
      # dummy-----------------------------
      Yp <- build_ydummy(p, psi, lambda, delta, numeric(dim_data), numeric(dim_data), include_mean)
      colnames(Yp) <- name_var
      Xp <- build_xdummy(1:p, lambda, psi, eps, include_mean)
      colnames(Xp) <- name_lag
      # NIW-------------------------------
      estimate_bvar_mn(X0, Y0, Xp, Yp)
    }
  )
  # estimate-bvar.cpp-----------------
  # bvar_est <- estimate_bvar_mn(X0, Y0, Xp, Yp)
  # Prior-----------------------------
  prior_mean <- bvar_est$prior_mean # A0
  prior_prec <- bvar_est$prior_prec # U0
  prior_scale <- bvar_est$prior_scale # S0
  prior_shape <- bvar_est$prior_shape # a0
  # Matrix normal---------------------
  mn_mean <- bvar_est$mnmean # matrix normal mean
  colnames(mn_mean) <- name_var
  rownames(mn_mean) <- name_lag
  mn_prec <- bvar_est$mnprec # matrix normal precision
  colnames(mn_prec) <- name_lag
  rownames(mn_prec) <- name_lag
  yhat <- bvar_est$fitted
  colnames(yhat) <- name_var
  # Inverse-wishart-------------------
  iw_scale <- bvar_est$iwscale # IW scale
  colnames(iw_scale) <- name_var
  rownames(iw_scale) <- name_var
  iw_shape <- prior_shape + nrow(Y0)
  res <- switch(
    bayes_spec$prior,
    "Minnesota" = {
      res <- list(
        # posterior------------
        coefficients = mn_mean, # posterior mean of MN
        fitted.values = yhat,
        residuals = bvar_est$residuals,
        mn_prec = mn_prec, # posterior precision of MN
        iw_scale = iw_scale, # posterior scale of IW
        iw_shape = iw_shape, # posterior shape of IW
        # prior----------------
        prior_mean = prior_mean, # A0
        prior_precision = prior_prec, # U0 = (Omega)^{-1}
        prior_scale = prior_scale, # S0
        prior_shape = prior_shape # a0
      )
    },
    "MN_Hierarchical" = {
      res <- estimate_hierachical_niw(
        num_iter = num_iter,
        num_burn = num_burn,
        x = X0,
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
      thin_id <- seq(from = 1, to = num_iter - num_burn, by = thinning)
      # thinparam_id <- seq(from = 1, to = num_iter - 1 - num_burn, by = thinning)
      thinparam_id <- seq(from = 1, to = num_iter - num_burn, by = thinning)
      res$psi_record <- res$psi_record[thin_id, ]
      res$alpha_record <- res$alpha_record[thinparam_id, ]
      # hyperparameters------------------
      colnames(res$psi_record) <- paste0("psi[", seq_len(ncol(res$psi_record)), "]")
      res$lambda_record <- as.matrix(res$lambda_record[thin_id])
      colnames(res$lambda_record) <- "lambda"
      res$psi_record <- as_draws_df(res$psi_record)
      res$lambda_record <- as_draws_df(res$lambda_record)
      # parameters-----------------------
      colnames(res$alpha_record) <- paste0("alpha[", seq_len(ncol(res$alpha_record)), "]")
      res$coefficients <-
        colMeans(res$alpha_record) %>%
        matrix(ncol = dim_data)
      colnames(res$coefficients) <- name_var
      rownames(res$coefficients) <- name_lag
      res$alpha_record <- as_draws_df(res$alpha_record)
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
        res$alpha_record,
        res$sigma_record
      )
      res$iter <- num_iter
      res$burn <- num_burn
      res$thin <- thinning
      res
    }
  )
  # variables-------------------------
  res$df <- nrow(mn_mean)
  res$p <- p
  res$m <- dim_data
  res$obs <- nrow(Y0)
  res$totobs <- nrow(y)
  res$call <- match.call()
  # model-----------------------------
  res$process <- paste(bayes_spec$process, bayes_spec$prior, sep = "_")
  res$type <- ifelse(include_mean, "const", "none")
  res$spec <- bayes_spec
  # data------------------------------
  res$y0 <- Y0
  res$design <- X0
  res$y <- y
  if (bayes_spec$prior == "Minnesota") {
    class(res) <- c("bvarmn", "normaliw", "bvharmod")
  } else {
    class(res) <- c("bvarhm", "bvharsp")
  }
  res
}