#' Log ML Function of Hierarchical BVAR to be in `optim`
#'
#' This function is for `fn` of [stats::optim()].
#'
#' @param param Vector of hyperparameter settings for [set_lambda()] and [set_psi()] in order.
#' @param delta delta in BVAR specification
#' @param y Time series data
#' @param p BVAR lag
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param ... not used
#' @details
#' `par` is the collection of hyperparameters.
#' * `lambda`
#' * `psi`
#' in order.
#' @noRd
logml_bvarhm <- function(param, delta, eps = 1e-04, y, p, include_mean = TRUE, ...) {
  dim_data <- ncol(y)
  if (length(param) != dim_data + 1) {
    stop("The number of 'param' is wrong.")
  }
  bvar_spec <- set_bvar(
    sigma = param[2:(dim_data + 1)],
    lambda = param[1],
    delta = delta,
    eps = eps
  )
  fit <- bvar_minnesota(y = y, p = p, bayes_spec = bvar_spec, include_mean = include_mean)
  -logml_stable(fit)
}

#' Fitting Bayesian VAR(p) of Minnesota Prior
#' 
#' This function fits BVAR(p) with Minnesota prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag (Default: 1)
#' @param num_chains Number of MCMC chains
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec A BVAR model specification by [set_bvar()].
#' @param scale_variance Proposal distribution scaling constant to adjust an acceptance rate
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param parallel List the same argument of [optimParallel::optimParallel()]. By default, this is empty, and the function does not execute parallel computation.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @param num_thread Number of threads
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
#'   \item{coefficients}{Posterior Mean}
#'   \item{fitted.values}{Fitted values}
#'   \item{residuals}{Residuals}
#'   \item{mn_mean}{Posterior mean matrix of Matrix Normal distribution}
#'   \item{mn_prec}{Posterior precision matrix of Matrix Normal distribution}
#'   \item{iw_scale}{Posterior scale matrix of posterior inverse-Wishart distribution}
#'   \item{iw_shape}{Posterior shape of inverse-Wishart distribution (\eqn{alpha_0} - obs + 2). \eqn{\alpha_0}: nrow(Dummy observation) - k}
#'   \item{df}{Numer of Coefficients: mp + 1 or mp}
#'   \item{m}{Dimension of the time series}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{prior_mean}{Prior mean matrix of Matrix Normal distribution: \eqn{A_0}}
#'   \item{prior_precision}{Prior precision matrix of Matrix Normal distribution: \eqn{\Omega_0^{-1}}}
#'   \item{prior_scale}{Prior scale matrix of inverse-Wishart distribution: \eqn{S_0}}
#'   \item{prior_shape}{Prior shape of inverse-Wishart distribution: \eqn{\alpha_0}}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{p}{Lag of VAR}
#'   \item{totobs}{Total number of the observation}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{y}{Raw input (`matrix`)}
#'   \item{call}{Matched call}
#'   \item{process}{Process string in the `bayes_spec`: `"BVAR_Minnesota"`}
#'   \item{spec}{Model specification (`bvharspec`)}
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
#' * [summary.normaliw()] to summarize BVAR model
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
                           num_chains = 1,
                           num_iter = 1000, 
                           num_burn = floor(num_iter / 2),
                           thinning = 1,
                           bayes_spec = set_bvar(),
                           scale_variance = .05,
                           include_mean = TRUE,
                           parallel = list(),
                           verbose = FALSE,
                           num_thread = 1) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  # model specification---------------
  if (!is.bvharspec(bayes_spec)) {
    # stop("Provide 'bvharspec' for 'bayes_spec'.")
    if (!(is.list(bayes_spec) && all(sapply(bayes_spec, is.bvharspec)))) {
      stop("Provide 'bvharspec' for 'bayes_spec'. Or, it should be the list of 'bvharspec'.")
    }
  }
  if (bayes_spec$process != "BVAR") {
    stop("'bayes_spec' must be the result of 'set_bvar()'.")
  }
  # if (bayes_spec$prior != "Minnesota") {
  #   stop("In 'set_bvar()', just input numeric values.")
  # }
  # if (bayes_spec$prior != "MN_Hierarchical") {
  #   stop("'bayes_spec' must be the result of 'set_lambda()' and 'set_psi()'.")
  # }
  if (bayes_spec$prior != "Minnesota" && bayes_spec$prior != "MN_Hierarchical") {
    stop("Wrong 'prior' inf 'set_bvar()'.")
  }
  dim_data <- ncol(y)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
  }
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  name_lag <- concatenate_colnames(name_var, 1:p, include_mean) # in misc-r.R file
  if (is.null(bayes_spec$delta)) {
    bayes_spec$delta <- rep(1, dim_data)
  }
  if (bayes_spec$prior == "Minnesota") {
    if (is.null(bayes_spec$sigma)) {
      bayes_spec$sigma <- apply(y, 2, sd)
    }
    res <- estimate_bvar_mn(
      y = y, lag = p,
      num_chains = num_chains, num_iter = num_iter, num_burn = num_burn, thin = thinning,
      bayes_spec = bayes_spec,
      include_mean = include_mean,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      display_progress = verbose, nthreads = num_thread
    )
    # res <- do.call(rbind, res)
    # return(res)
    # res <- append(res, do.call(rbind, res$record))
    # res$record <- NULL
    res$record <- do.call(rbind, res$record)
    # res <- append(res, do.call(rbind, res$record))
    rec_names <- colnames(res$record)
    param_names <- gsub(pattern = "_record$", replacement = "", rec_names)
    res$record <- apply(
      res$record,
      2,
      function(x) {
        if (is.vector(x[[1]])) {
          return(as.matrix(unlist(x)))
        }
        do.call(rbind, x)
      }
    )
    names(res$record) <- rec_names
    res <- append(res, res$record)
    res$record <- NULL
    # summary across chains-------------
    res$coefficients <- matrix(colMeans(res$alpha_record), ncol = dim_data)
    res$covmat <- matrix(colMeans(res$sigma_record), ncol = dim_data)
    # preprocess the results------------
    if (num_chains > 1) {
      res[rec_names] <- lapply(
        seq_along(res[rec_names]),
        function(id) {
          split_chain(res[rec_names][[id]], chain = num_chains, varname = param_names[id])
        }
      )
    } else {
      res[rec_names] <- lapply(
        seq_along(res[rec_names]),
        function(id) {
          colnames(res[rec_names][[id]]) <- paste0(param_names[id], "[", seq_len(ncol(res[rec_names][[id]])), "]")
          res[rec_names][[id]]
        }
      )
    }
    res[rec_names] <- lapply(res[rec_names], as_draws_df)
    res$param <- bind_draws(
      res$alpha_record,
      res$sigma_record
    )
    res[rec_names] <- NULL
    res$param_names <- param_names
    # coef_and_sig <- sim_mniw_export(
    #   num_iter,
    #   res$mn_mean,
    #   res$mn_prec,
    #   res$iw_scale,
    #   res$iw_shape,
    #   TRUE
    # ) %>%
    #   simplify2array()
    # thin_id <- seq(from = num_burn + 1, to = num_iter, by = thinning)
    # len_res <- length(thin_id)
    # coef_record <- lapply(coef_and_sig[1, ], c)
    # coef_record <- coef_record[thin_id]
    # coef_record <- do.call(rbind, coef_record)
    # colnames(coef_record) <- paste0("alpha", "[", seq_len(ncol(coef_record)), "]")
    # res$coefficients <-
    #   colMeans(coef_record) %>%
    #   matrix(ncol = dim_data)
    # coef_and_sig$iw <- coef_and_sig[2, ]
    # coef_and_sig$iw <- coef_and_sig$iw[thin_id]
    # res$covmat <- Reduce("+", coef_and_sig$iw) / length(coef_and_sig$iw)
    # sig_record <- do.call(
    #   rbind,
    #   lapply(coef_and_sig$iw, c)
    # )
    # colnames(sig_record) <- paste0("sigma[", seq_len(ncol(sig_record)), "]")
    # res$param <- bind_rows(
    #   as_draws_df(coef_record),
    #   as_draws_df(sig_record)
    # )
    colnames(res$y) <- name_var
    colnames(res$y0) <- name_var
    colnames(res$design) <- name_lag
    # Prior-----------------------------
    colnames(res$prior_mean) <- name_var
    rownames(res$prior_mean) <- name_lag
    colnames(res$prior_precision) <- name_lag
    rownames(res$prior_precision) <- name_lag
    colnames(res$prior_scale) <- name_var
    rownames(res$prior_scale) <- name_var
    # Matrix normal---------------------
    colnames(res$mn_mean) <- name_var
    rownames(res$mn_mean) <- name_lag
    colnames(res$mn_prec) <- name_lag
    rownames(res$mn_prec) <- name_lag
    colnames(res$fitted.values) <- name_var
    # Inverse-wishart-------------------
    colnames(res$iw_scale) <- name_var
    rownames(res$iw_scale) <- name_var
  } else {
    psi <- bayes_spec$sigma$mode
    psi <- rep(psi, dim_data)
    delta <- bayes_spec$delta
    lambda <- bayes_spec$lambda$mode
    eps <- bayes_spec$eps
    # Y0 = X0 A + Z---------------------
    Y0 <- build_response(y, p, p + 1)
    if (!is.null(colnames(y))) {
      name_var <- colnames(y)
    } else {
      name_var <- paste0("y", seq_len(dim_data))
    }
    colnames(Y0) <- name_var
    X0 <- build_design(y, p, include_mean)
    name_lag <- concatenate_colnames(name_var, 1:p, include_mean)
    colnames(X0) <- name_lag
    # prior selection-------------------
    lower_vec <- unlist(bayes_spec)
    lower_vec <- as.numeric(lower_vec[grepl(pattern = "lower$", x = names(unlist(bayes_spec)))])
    lower_vec <- c(lower_vec[2], rep(lower_vec[1], dim_data))
    upper_vec <- unlist(bayes_spec)
    upper_vec <- as.numeric(upper_vec[grepl(pattern = "upper$", x = names(unlist(bayes_spec)))])
    upper_vec <- c(upper_vec[2], rep(upper_vec[1], dim_data))
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
    Yp <- build_ydummy_export(p, psi, lambda, delta, numeric(dim_data), numeric(dim_data), include_mean)
    colnames(Yp) <- name_var
    Xp <- build_xdummy_export(1:p, lambda, psi, eps, include_mean)
    colnames(Xp) <- name_lag
    param_init <- lapply(1:num_chains, function(x) append(init_par, list(scale_variance = scale_variance)))
    res <- estimate_bvar_mh(
      num_chains = num_chains,
      num_iter = num_iter,
      num_burn = num_burn,
      thin = thinning,
      x = X0,
      y = Y0,
      x_dummy = Xp,
      y_dummy = Yp,
      param_prior = bayes_spec,
      param_init = param_init,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      display_progress = verbose,
      nthreads = num_thread
    )
    res <- do.call(rbind, res)
    rec_names <- colnames(res)
    param_names <- gsub(pattern = "_record$", replacement = "", rec_names)
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
    # summary across chains-------------
    res$coefficients <- matrix(colMeans(res$alpha_record), ncol = dim_data)
    res$covmat <- matrix(colMeans(res$sigma_record), ncol = dim_data)
    res$acc_rate <- mean(res$accept_record)
    # colnames(res$coefficients) <- name_var
    # rownames(res$coefficients) <- name_lag
    # colnames(res$covmat) <- name_var
    # rownames(res$covmat) <- name_var
    # preprocess the results------------
    if (num_chains > 1) {
      res[rec_names] <- lapply(
        seq_along(res[rec_names]),
        function(id) {
          split_chain(res[rec_names][[id]], chain = num_chains, varname = param_names[id])
        }
      )
    } else {
      res[rec_names] <- lapply(
        seq_along(res[rec_names]),
        function(id) {
          colnames(res[rec_names][[id]]) <- paste0(param_names[id], "[", seq_len(ncol(res[rec_names][[id]])), "]")
          res[rec_names][[id]]
        }
      )
    }
    res[rec_names] <- lapply(res[rec_names], as_draws_df)
    res$hyperparam <- bind_draws(
      res$lambda_record,
      res$psi_record,
      res$accept_record
    )
    res$param <- bind_draws(
      res$alpha_record,
      res$sigma_record
    )
    res[rec_names] <- NULL
    res$param_names <- param_names
    # data------------------------------
    res$y0 <- Y0
    res$design <- X0
    # res$y <- y
    # variables-------------------------
    res$df <- nrow(res$coefficients)
    # res$p <- p
    res$m <- dim_data
    res$obs <- nrow(Y0)
    res$totobs <- nrow(y)
    # model-----------------------------
    # res$type <- ifelse(include_mean, "const", "none")
    # res$iter <- num_iter
    # res$burn <- num_burn
    # res$thin <- thinning
  }
  res$y <- y
  res$p <- p
  res$type <- ifelse(include_mean, "const", "none")
  # res <- estimate_bvar_mn(y, p, bayes_spec, include_mean)
  # colnames(res$y) <- name_var
  # colnames(res$y0) <- name_var
  # colnames(res$design) <- name_lag
  # Prior-----------------------------
  # colnames(res$prior_mean) <- name_var
  # rownames(res$prior_mean) <- name_lag
  # colnames(res$prior_precision) <- name_lag
  # rownames(res$prior_precision) <- name_lag
  # colnames(res$prior_scale) <- name_var
  # rownames(res$prior_scale) <- name_var
  # # Matrix normal---------------------
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_lag
  colnames(res$covmat) <- name_var
  rownames(res$covmat) <- name_var
  # colnames(res$mn_prec) <- name_lag
  # rownames(res$mn_prec) <- name_lag
  # colnames(res$fitted.values) <- name_var
  # # Inverse-wishart-------------------
  # colnames(res$iw_scale) <- name_var
  # rownames(res$iw_scale) <- name_var
  res$chain <- num_chains
  res$iter <- num_iter
  res$burn <- num_burn
  res$thin <- thinning
  # model-----------------------------
  res$call <- match.call()
  res$process <- paste(bayes_spec$process, bayes_spec$prior, sep = "_")
  res$spec <- bayes_spec
  # class(res) <- c("bvarmn", "normaliw", "bvharmod")
  if (bayes_spec$prior == "Minnesota") {
    class(res) <- c("bvarmn", "bvharmod")
  } else {
    class(res) <- c("bvarhm", "bvharsp")
  }
  class(res) <- c(class(res), "normaliw")
  res
}