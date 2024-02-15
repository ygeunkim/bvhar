#' Fitting Bayesian VHAR of Minnesota Prior
#' 
#' This function fits BVHAR with Minnesota prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param bayes_spec A BVHAR model specification by [set_bvhar()] (default) or [set_weight_bvhar()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
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
#' @references Kim, Y. G., and Baek, C. (2023). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation.
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
  dim_data <- ncol(y)
  # model specification---------------
  if (is.null(bayes_spec$sigma)) {
    bayes_spec$sigma <- apply(y, 2, sd)
  }
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
  }
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  name_har <- concatenate_colnames(name_var, c("day", "week", "month"), include_mean) # in misc-r.R file
  if (minnesota_type == "MN_VAR") {
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
  is_short <- minnesota_type == "MN_VAR"
  res <- estimate_bvhar_mn(y, week, month, bayes_spec, include_mean, is_short)
  colnames(res$y) <- name_var
  colnames(res$y0) <- name_var
  # Prior-----------------------------
  colnames(res$prior_mean) <- name_var
  rownames(res$prior_mean) <- name_har
  colnames(res$prior_prec) <- name_har
  rownames(res$prior_prec) <- name_har
  colnames(res$prior_scale) <- name_var
  rownames(res$prior_scale) <- name_var
  # Matrix normal---------------------
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_har
  colnames(res$mn_prec) <- name_har
  rownames(res$mn_prec) <- name_har
  colnames(res$fitted.values) <- name_var
  # Inverse-wishart-------------------
  colnames(res$iw_scale) <- name_var
  rownames(res$iw_scale) <- name_var
  # S3--------------------------------
  res$call <- match.call()
  res$process <- paste(bayes_spec$process, minnesota_type, sep = "_")
  res$spec <- bayes_spec
  class(res) <- c("bvharmn", "normaliw", "bvharmod")
  res
}