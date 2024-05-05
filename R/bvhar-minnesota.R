#' Log ML Function of Hierarchical BVHAR to be in `optim`
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
logml_bvharhm <- function(param, delta, eps = 1e-04, y, har, include_mean = TRUE, ...) {
  dim_data <- ncol(y)
  if (length(param) != dim_data + 1) {
    stop("The number of 'param' is wrong.")
  }
  bvhar_spec <- set_bvhar(
    sigma = param[2:(dim_data + 1)],
    lambda = param[1],
    delta = delta,
    eps = eps
  )
  fit <- bvhar_minnesota(y = y, har = har, bayes_spec = bvhar_spec, include_mean = include_mean)
  -logml_stable(fit)
}

#' Log ML Function of Hierarchical BVHAR to be in `optim`
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
logml_weight_bvharhm <- function(param, daily, weekly, monthly, eps = 1e-04, y, har, include_mean = TRUE, ...) {
  dim_data <- ncol(y)
  if (length(param) != dim_data + 1) {
    stop("The number of 'param' is wrong.")
  }
  bvhar_spec <- set_weight_bvhar(
    sigma = param[2:(dim_data + 1)],
    lambda = param[1],
    daily = daily,
    weekly = weekly,
    monthly = monthly,
    eps = eps
  )
  fit <- bvhar_minnesota(y = y, har = har, bayes_spec = bvhar_spec, include_mean = include_mean)
  -logml_stable(fit)
}

#' Fitting Bayesian VHAR of Minnesota Prior
#' 
#' This function fits BVHAR with Minnesota prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param num_chains Number of MCMC chains
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec A BVHAR model specification by [set_bvhar()] (default) or [set_weight_bvhar()].
#' @param scale_variance Proposal distribution scaling constant to adjust an acceptance rate
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param parallel List the same argument of [optimParallel::optimParallel()]. By default, this is empty, and the function does not execute parallel computation.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @param num_thread Number of threads
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
#' @return `bvhar_minnesota()` returns an object `bvharmn` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Posterior Mean}
#'   \item{fitted.values}{Fitted values}
#'   \item{residuals}{Residuals}
#'   \item{mn_mean}{Posterior mean matrix of Matrix Normal distribution}
#'   \item{mn_prec}{Posterior precision matrix of Matrix Normal distribution}
#'   \item{iw_scale}{Posterior scale matrix of posterior inverse-wishart distribution}
#'   \item{iw_shape}{Posterior shape of inverse-Wishart distribution (\eqn{\nu_0} - obs + 2). \eqn{\nu_0}: nrow(Dummy observation) - k}
#'   \item{df}{Numer of Coefficients: 3m + 1 or 3m}
#'   \item{m}{Dimension of the time series}
#'   \item{obs}{Sample size used when training = `totobs` - 22}
#'   \item{prior_mean}{Prior mean matrix of Matrix Normal distribution: \eqn{M_0}}
#'   \item{prior_precision}{Prior precision matrix of Matrix Normal distribution: \eqn{\Omega_0^{-1}}}
#'   \item{prior_scale}{Prior scale matrix of inverse-Wishart distribution: \eqn{\Psi_0}}
#'   \item{prior_shape}{Prior shape of inverse-Wishart distribution: \eqn{\nu_0}}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{p}{3, this element exists to run the other functions}
#'   \item{week}{Order for weekly term}
#'   \item{month}{Order for monthly term}
#'   \item{totobs}{Total number of the observation}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{HARtrans}{VHAR linear transformation matrix: \eqn{C_{HAR}}}
#'   \item{y}{Raw input (`matrix`)}
#'   \item{call}{Matched call}
#'   \item{process}{Process string in the `bayes_spec`: `"BVHAR_MN_VAR"` (BVHAR-S) or `"BVHAR_MN_VHAR"` (BVHAR-L)}
#'   \item{spec}{Model specification (`bvharspec`)}
#' }
#' It is also `normaliw` and `bvharmod` class.
#' @references Kim, Y. G., and Baek, C. (2023). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation.
#' @seealso 
#' * [set_bvhar()] to specify the hyperparameters of BVHAR-S
#' * [set_weight_bvhar()] to specify the hyperparameters of BVHAR-L
#' * [summary.normaliw()] to summarize BVHAR model
#' @examples
#' # Perform the function using etf_vix dataset
#' fit <- bvhar_minnesota(y = etf_vix[,1:3])
#' class(fit)
#' 
#' # Extract coef, fitted values, and residuals
#' coef(fit)
#' head(residuals(fit))
#' head(fitted(fit))
#' @order 1
#' @export
bvhar_minnesota <- function(y,
                            har = c(5, 22),
                            num_chains = 1,
                            num_iter = 1000, 
                            num_burn = floor(num_iter / 2),
                            thinning = 1,
                            bayes_spec = set_bvhar(),
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
  if (!is.bvharspec(bayes_spec)) {
    stop("Provide 'bvharspec' for 'bayes_spec'.")
  }
  if (bayes_spec$process != "BVHAR") {
    stop("'bayes_spec' must be the result of 'set_bvhar()' or 'set_weight_bvhar()'.")
  }
  if (bayes_spec$prior != "MN_VAR" && bayes_spec$prior != "MN_VHAR" && bayes_spec$prior != "MN_Hierarchical") {
    stop("Wrong 'prior' inf 'set_bvhar()' or 'set_weight_bvhar()'.")
  }
  if (length(har) != 2 || !is.numeric(har)) {
    stop("'har' should be numeric vector of length 2.")
  }
  if (har[1] > har[2]) {
    stop("'har[1]' should be smaller than 'har[2]'.")
  }
  week <- har[1] # 5
  month <- har[2] # 22
  minnesota_type <- bayes_spec$prior
  dim_data <- ncol(y)
  # model specification---------------
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
  }
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  name_har <- concatenate_colnames(name_var, c("day", "week", "month"), include_mean) # in misc-r.R file
  if (minnesota_type == "MN_VAR" || minnesota_type == "MN_Hierarchical") {
    if (is.null(bayes_spec$delta)) {
      bayes_spec$delta <- rep(1, dim_data)
    }
  } else {
    if (is.null(bayes_spec$daily)) {
      bayes_spec$daily <- rep(1, dim_data)
    }
    if (is.null(bayes_spec$weekly)) {
      bayes_spec$weekly <- rep(1, dim_data)
    }
    if (is.null(bayes_spec$monthly)) {
      bayes_spec$monthly <- rep(1, dim_data)
    }
  }
  if (minnesota_type != "MN_Hierarchical") {
    if (is.null(bayes_spec$sigma)) {
      bayes_spec$sigma <- apply(y, 2, sd)
    }
    # is_short <- minnesota_type == "MN_VAR"
    # res <- estimate_bvhar_mn(y, week, month, bayes_spec, include_mean, is_short)
    res <- estimate_bvhar_mn(
      y = y, week = week, month = month,
      num_chains = num_chains, num_iter = num_iter, num_burn = num_burn, thin = thinning,
      bayes_spec = bayes_spec,
      include_mean = include_mean,
      seed_chain = sample.int(.Machine$integer.max, size = num_chains),
      display_progress = verbose, nthreads = num_thread
    )
    # res <- do.call(rbind, res)
    # res <- append(res, do.call(rbind, res$record))
    # res$record <- NULL
    res$record <- do.call(rbind, res$record)
    colnames(res$record) <- gsub(pattern = "^alpha", replacement = "phi", x = colnames(res$record)) # alpha to phi
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
    res$coefficients <- matrix(colMeans(res$phi_record), ncol = dim_data)
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
      res$phi_record,
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
    # colnames(coef_record) <- paste0("phi", "[", seq_len(ncol(coef_record)), "]")
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
    # Prior-----------------------------
    colnames(res$prior_mean) <- name_var
    rownames(res$prior_mean) <- name_har
    colnames(res$prior_precision) <- name_har
    rownames(res$prior_precision) <- name_har
    colnames(res$prior_scale) <- name_var
    rownames(res$prior_scale) <- name_var
    # Matrix normal---------------------
    colnames(res$mn_mean) <- name_var
    rownames(res$mn_mean) <- name_har
    colnames(res$mn_prec) <- name_har
    rownames(res$mn_prec) <- name_har
    colnames(res$fitted.values) <- name_var
    # Inverse-wishart-------------------
    colnames(res$iw_scale) <- name_var
    rownames(res$iw_scale) <- name_var
  } else {
    psi <- bayes_spec$sigma$mode
    psi <- rep(psi, dim_data)
    # delta <- bayes_spec$delta
    lambda <- bayes_spec$lambda$mode
    eps <- bayes_spec$eps
    # Y0 = X0 A + Z---------------------
    Y0 <- build_response(y, month, month + 1)
    if (!is.null(colnames(y))) {
      name_var <- colnames(y)
    } else {
      name_var <- paste0("y", seq_len(dim_data))
    }
    colnames(Y0) <- name_var
    X0 <- build_design(y, month, include_mean)
    HARtrans <- scale_har(dim_data, week, month, include_mean)
    name_har <- concatenate_colnames(name_var, c("day", "week", "month"), include_mean) # in misc-r.R file
    X1 <- X0 %*% t(HARtrans)
    colnames(X1) <- name_har
    # num_design <- nrow(Y0)
    # dim_har <- ncol(X1) # 3 * dim_data + 1
    # 
    # prior selection-------------------
    lower_vec <- unlist(bayes_spec)
    lower_vec <- as.numeric(lower_vec[grepl(pattern = "lower$", x = names(unlist(bayes_spec)))])
    lower_vec <- c(lower_vec[2], rep(lower_vec[1], dim_data))
    upper_vec <- unlist(bayes_spec)
    upper_vec <- as.numeric(upper_vec[grepl(pattern = "upper$", x = names(unlist(bayes_spec)))])
    upper_vec <- c(upper_vec[2], rep(upper_vec[1], dim_data))
    is_short <- is.null(bayes_spec$daily)
    if (is_short) {
      # delta <- bayes_spec$delta
      if (length(parallel) > 0) {
        init_par <-
          optimParallel(
            par = c(lambda, psi),
            fn = logml_bvharhm,
            lower = lower_vec,
            upper = upper_vec,
            hessian = TRUE,
            delta = bayes_spec$delta,
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
            delta = bayes_spec$delta,
            eps = bayes_spec$eps,
            y = y,
            har = har,
            include_mean = include_mean
          )
      }
    } else {
      if (length(parallel) > 0) {
        init_par <-
          optimParallel(
            par = c(lambda, psi),
            fn = logml_weight_bvharhm,
            lower = lower_vec,
            upper = upper_vec,
            hessian = TRUE,
            daily = bayes_spec$daily,
            weekly = bayes_spec$weekly,
            monthly = bayes_spec$monthly,
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
            fn = logml_weight_bvharhm,
            method = "L-BFGS-B",
            lower = lower_vec,
            upper = upper_vec,
            hessian = TRUE,
            daily = bayes_spec$daily,
            weekly = bayes_spec$weekly,
            monthly = bayes_spec$monthly,
            eps = bayes_spec$eps,
            y = y,
            har = har,
            include_mean = include_mean
          )
      }
    }
    lambda <- init_par$par[1]
    psi <- init_par$par[2:(1 + dim_data)]
    hess <- init_par$hessian
    # dummy-----------------------------
    # Yp <- build_ydummy_export(3, psi, lambda, delta, numeric(dim_data), numeric(dim_data), include_mean)
    if (is_short) {
      Yp <- build_ydummy_export(3, psi, lambda, bayes_spec$delta, numeric(dim_data), numeric(dim_data), include_mean)
    } else {
      Yp <- build_ydummy_export(3, psi, lambda, bayes_spec$daily, bayes_spec$weekly, bayes_spec$monthly, include_mean)
    }
    colnames(Yp) <- name_var
    Xp <- build_xdummy_export(1:3, lambda, psi, eps, include_mean)
    colnames(Xp) <- name_har
    param_init <- lapply(1:num_chains, function(x) append(init_par, list(scale_variance = scale_variance)))
    res <- estimate_bvar_mh(
      num_chains = num_chains,
      num_iter = num_iter,
      num_burn = num_burn,
      thin = thinning,
      x = X1,
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
    colnames(res) <- gsub(pattern = "^alpha", replacement = "phi", x = colnames(res)) # alpha to p
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
    res$coefficients <- matrix(colMeans(res$phi_record), ncol = dim_data)
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
      res$phi_record,
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
    # res$p <- 3
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
  res$p <- 3
  res$week <- week
  res$month <- month
  res$type <- ifelse(include_mean, "const", "none")
  # is_short <- minnesota_type == "MN_VAR"
  # res <- estimate_bvhar_mn(y, week, month, bayes_spec, include_mean, is_short)
  # colnames(res$y) <- name_var
  # colnames(res$y0) <- name_var
  # # Prior-----------------------------
  # colnames(res$prior_mean) <- name_var
  # rownames(res$prior_mean) <- name_har
  # colnames(res$prior_precision) <- name_har
  # rownames(res$prior_precision) <- name_har
  # colnames(res$prior_scale) <- name_var
  # rownames(res$prior_scale) <- name_var
  # Matrix normal---------------------
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_har
  # colnames(res$mn_prec) <- name_har
  # rownames(res$mn_prec) <- name_har
  # colnames(res$fitted.values) <- name_var
  colnames(res$covmat) <- name_var
  rownames(res$covmat) <- name_var
  # Inverse-wishart-------------------
  # colnames(res$iw_scale) <- name_var
  # rownames(res$iw_scale) <- name_var
  res$chain <- num_chains
  res$iter <- num_iter
  res$burn <- num_burn
  res$thin <- thinning
  # S3--------------------------------
  res$call <- match.call()
  res$process <- paste(bayes_spec$process, minnesota_type, sep = "_")
  res$spec <- bayes_spec
  # class(res) <- c("bvharmn", "normaliw", "bvharmod")
  if (minnesota_type == "MN_Hierarchical") {
    class(res) <- c("bvharhm", "bvharsp")
  } else {
    class(res) <- c("bvharmn", "bvharmod")
  }
  class(res) <- c(class(res), "normaliw")
  res
}