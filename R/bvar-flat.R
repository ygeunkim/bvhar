#' Fitting Bayesian VAR(p) of Flat Prior
#' 
#' This function fits BVAR(p) with flat prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param bayes_spec A BVAR model specification by [set_bvar_flat()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @details 
#' Ghosh et al. (2018) gives flat prior for residual matrix in BVAR.
#' 
#' Under this setting, there are many models such as hierarchical or non-hierarchical.
#' This function chooses the most simple non-hierarchical matrix normal prior in Section 3.1.
#' 
#' \deqn{A \mid \Sigma_e \sim MN(0, U^{-1}, \Sigma_e)}
#' where U: precision matrix (MN: [matrix normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution)).
#' \deqn{p (\Sigma_e) \propto 1}
#' @return `bvar_flat()` returns an object `bvarflat` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Posterior Mean matrix of Matrix Normal distribution}
#'   \item{fitted.values}{Fitted values}
#'   \item{residuals}{Residuals}
#'   \item{mn_prec}{Posterior precision matrix of Matrix Normal distribution}
#'   \item{iw_scale}{Posterior scale matrix of posterior inverse-wishart distribution}
#'   \item{iw_shape}{Posterior shape of inverse-wishart distribution}
#'   \item{df}{Numer of Coefficients: mp + 1 or mp}
#'   \item{p}{Lag of VAR}
#'   \item{m}{Dimension of the time series}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{process}{Process string in the `bayes_spec`: `"BVAR_Flat"`}
#'   \item{spec}{Model specification (`bvharspec`)}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{call}{Matched call}
#'   \item{prior_mean}{Prior mean matrix of Matrix Normal distribution: zero matrix}
#'   \item{prior_precision}{Prior precision matrix of Matrix Normal distribution: \eqn{U^{-1}}}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input (`matrix`)}
#' }
#' @references 
#' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526).
#' 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25.
#' @seealso 
#' * [set_bvar_flat()] to specify the hyperparameters of BVAR flat prior.
#' * [coef.bvarflat()], [residuals.bvarflat()], and [fitted.bvarflat()]
#' * [predict.bvarflat()] to forecast the BVHAR process
#' @order 1
#' @export
bvar_flat <- function(y, p, bayes_spec = set_bvar_flat(), include_mean = TRUE) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  if (!is.bvharspec(bayes_spec)) {
    stop("Provide 'bvharspec' for 'bayes_spec'.")
  }
  if (bayes_spec$process != "BVAR") {
    stop("'bayes_spec' must be the result of 'set_bvar_flat()'.")
  }
  # Y0 = X0 B + Z---------------------
  Y0 <- build_response(y, p, p + 1)
  m <- ncol(y)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(m))
  }
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  colnames(Y0) <- name_var
  X0 <- build_design(y, p, include_mean)
  name_lag <- concatenate_colnames(name_var, 1:p, include_mean) # in misc-r.R file
  colnames(X0) <- name_lag
  # spec------------------------------
  if (is.null(bayes_spec$U)) {
    bayes_spec$U <- diag(ncol(X0)) # identity matrix
  }
  prior_prec <- bayes_spec$U
  # Matrix normal---------------------
  posterior <- estimate_mn_flat(X0, Y0, prior_prec)
  mn_mean <- posterior$mnmean # posterior mean
  colnames(mn_mean) <- name_var
  rownames(mn_mean) <- name_lag
  mn_prec <- posterior$mnprec
  colnames(mn_prec) <- name_lag
  rownames(mn_prec) <- name_lag
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
    residuals = Y0 - yhat,
    mn_prec = mn_prec,
    iw_scale = iw_scale,
    iw_shape = posterior$iwshape,
    # variables-----------
    df = nrow(mn_mean), # k = mp + 1 or mp
    p = p, # p
    m = m, # m
    obs = nrow(Y0), # s = n - p
    totobs = nrow(y), # n
    # about model---------
    process = paste(bayes_spec$process, bayes_spec$prior, sep = "_"),
    spec = bayes_spec,
    type = ifelse(include_mean, "const", "none"),
    call = match.call(),
    # prior----------------
    prior_mean = array(0L, dim = dim(mn_mean)), # zero matrix
    prior_precision = prior_prec, # given as input
    # data----------------
    y0 = Y0,
    design = X0,
    y = y
  )
  class(res) <- c("bvarflat", "normaliw", "bvharmod")
  res
}
