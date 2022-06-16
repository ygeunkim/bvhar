#' Fitting Bayesian VHAR of Minnesota Prior
#' 
#' This function fits BVHAR with Minnesota prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param har `r lifecycle::badge("experimental")` Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param bayes_spec A BVHAR model specification by [set_bvhar()] (default) or [set_weight_bvhar()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' 
#' @details 
#' Apply Minnesota prior to Vector HAR: \eqn{\Phi} (VHAR matrices) and \eqn{\Sigma_e} (residual covariance).
#' 
#' \deqn{\Phi \mid \Sigma_e \sim MN(P_0, \Psi_0, \Sigma_e)}
#' \deqn{\Sigma_e \sim IW(U_0, d_0)}
#' (MN: [matrix normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution), IW: [inverse-wishart](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution))
#' 
#' There are two types of Minnesota priors for BVHAR:
#' 
#' * VAR-type Minnesota prior specified by [set_bvhar()]
#' * VHAR-type Minnesota prior specified by [set_weight_bvhar()]
#' 
#' Two types of Minnesota priors builds different dummy variables for Y0.
#' See [var_design_formulation].
#' 
#' @return `bvhar_minnesota` returns an object `bvharmn` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Posterior Mean matrix of Matrix Normal distribution}
#'   \item{fitted.values}{Fitted values}
#'   \item{residuals}{Residuals}
#'   \item{mn_prec}{Posterior precision matrix of Matrix Normal distribution}
#'   \item{iw_scale}{Posterior scale matrix of posterior inverse-wishart distribution}
#'   \item{iw_shape}{Posterior shape of inverse-wishart distribution (\eqn{d_0} - obs + 2). \eqn{d_0}: nrow(Dummy observation) - k}
#'   \item{df}{Numer of Coefficients: 3m + 1 or 3m}
#'   \item{p}{3, this element exists to run the other functions}
#'   \item{m}{Dimension of the time series}
#'   \item{obs}{Sample size used when training = `totobs` - 22}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Process string in the `bayes_spec`: `"BVHAR_MN_VAR"` or `"BVHAR_MN_VHAR"`}
#'   \item{spec}{Model specification (`bvharspec`)}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{prior_mean}{Prior mean matrix of Matrix Normal distribution: \eqn{P_0}}
#'   \item{prior_precision}{Prior precision matrix of Matrix Normal distribution: \eqn{\Psi_0}}
#'   \item{prior_scale}{Prior scale matrix of inverse-wishart distribution: \eqn{U_0}}
#'   \item{prior_shape}{Prior shape of inverse-wishart distribution: \eqn{d_0}}
#'   \item{HARtrans}{VHAR linear transformation matrix: \eqn{T_{HAR}}}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' 
#' @references 
#' Kim, Y. G., and Baek, C. (n.d.). *Bayesian vector heterogeneous autoregressive modeling*. Preprint.
#' 
#' @seealso 
#' * [set_bvhar()] to specify the hyperparameters of VAR-type Minnesota prior.
#' * [set_weight_bvhar()] to specify the hyperparameters of HAR-type Minnesota prior.
#' * [coef.bvharmn()], [residuals.bvharmn()], and [fitted.bvharmn()]
#' * [summary.normaliw()] to summarize BVHAR model
#' * [predict.bvharmn()] to forecast the BVHAR process
#' 
#' @examples
#' # Perform the function using etf_vix dataset
#' fit <- bvhar_minnesota(y = etf_vix)
#' class(fit)
#' str(fit)
#' 
#' # Extract coef, fitted values, and residuals
#' coef(fit)
#' residuals(fit)
#' fitted(fit)
#' 
#' @order 1
#' @export
bvhar_minnesota <- function(y, har = c(5, 22), bayes_spec = set_bvhar(), include_mean = TRUE) {
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
  if (length(har) != 2 || !is.numeric(har)) {
    stop("'har' should be numeric vector of length 2.")
  }
  if (har[1] > har[2]) {
    stop("'har[1]' should be smaller than 'har[2]'.")
  }
  week <- har[1] # 5
  month <- har[2] # 22
  minnesota_type <- bayes_spec$prior
  m <- ncol(y)
  N <- nrow(y)
  num_coef <- 3 * m + 1
  # model specification---------------
  if (is.null(bayes_spec$sigma)) {
    bayes_spec$sigma <- apply(y, 2, sd)
  }
  sigma <- bayes_spec$sigma
  lambda <- bayes_spec$lambda
  eps <- bayes_spec$eps
  # Y0 = X0 A + Z---------------------
  Y0 <- build_y0(y, month, month + 1)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(m))
  }
  colnames(Y0) <- name_var
  s <- nrow(Y0)
  X0 <- build_design(y, month)
  HARtrans <- scale_har(m, week, month)
  name_har <- concatenate_colnames(name_var, c("day", "week", "month")) # in misc-r.R file
  # dummy-----------------------------
  Yh <- switch(
    minnesota_type,
    "MN_VAR" = {
      if (is.null(bayes_spec$delta)) {
        bayes_spec$delta <- rep(1, m)
      }
      Yh <- build_ydummy(3, sigma, lambda, bayes_spec$delta, numeric(m), numeric(m))
      colnames(Yh) <- name_var
      Yh
    },
    "MN_VHAR" = {
      if (is.null(bayes_spec$daily)) {
        bayes_spec$daily <- rep(1, m)
      }
      if (is.null(bayes_spec$weekly)) {
        bayes_spec$weekly <- rep(1, m)
      }
      if (is.null(bayes_spec$monthly)) {
        bayes_spec$monthly <- rep(1, m)
      }
      Yh <- build_ydummy(
        3,
        sigma, 
        lambda, 
        bayes_spec$daily, 
        bayes_spec$weekly, 
        bayes_spec$monthly
      )
      colnames(Yh) <- name_var
      Yh
    }
  )
  Xh <- build_xdummy(1:3, lambda, sigma, eps)
  colnames(Xh) <- name_har
  # const or none---------------------
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  if (!include_mean) {
    X0 <- X0[, -(month * m + 1)] # exclude 1 column
    HARtrans <- HARtrans[-num_coef, -(month * m + 1)] # HARtrans: 3m x 22m matrix
    Th <- nrow(Yh)
    Yh <- Yh[-Th,] # exclude intercept block from Yh (last row)
    Xh <- Xh[-Th, -num_coef] # exclude intercept block from Xh (last row and last column)
    name_har <- name_har[-num_coef] # remove const (row)name
    num_coef <- num_coef - 1 # df = 3 * m
  }
  X1 <- X0 %*% t(HARtrans)
  colnames(X1) <- name_har
  # estimate-bvar.cpp-----------------
  posterior <- estimate_bvar_mn(X1, Y0, Xh, Yh)
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
  # S3--------------------------------
  res <- list(
    # posterior-----------
    coefficients = mn_mean,
    fitted.values = yhat,
    residuals = posterior$residuals,
    mn_prec = mn_prec,
    iw_scale = iw_scale,
    iw_shape = prior_shape + s, # if adding improper prior, d0 + s + 2
    # variables-----------
    df = num_coef, # nrow(Phihat) = 3 * m + 1 or 3 * m
    p = 3, # add for other function (df = 3m + 1 = mp + 1)
    week = week, # default: 5
    month = month, # default: 22
    m = m, # m
    obs = s, # s = n - 22
    totobs = N, # n
    # about model---------
    call = match.call(),
    process = paste(bayes_spec$process, minnesota_type, sep = "_"),
    spec = bayes_spec,
    type = ifelse(include_mean, "const", "none"),
    # prior----------------
    prior_mean = prior_mean,
    prior_precision = prior_prec,
    prior_scale = prior_scale,
    prior_shape = prior_shape, 
    # data----------------
    HARtrans = HARtrans,
    y0 = Y0,
    design = X0,
    y = y
  )
  class(res) <- c("bvharmn", "normaliw", "bvharmod")
  res
}
