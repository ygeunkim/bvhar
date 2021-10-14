#' Fitting Bayesian VAR(p) of Flat Prior
#' 
#' This function fits BVAR(p) with flat prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param bayes_spec `r lifecycle::badge("experimental")` A BVAR model specification by [set_bvar_flat()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' 
#' @details 
#' Ghosh et al. (2018) gives flat prior for residual matrix in BVAR.
#' 
#' Under this setting, there are many models such as hierarchical or non-hierarchical.
#' This function chooses the most simple non-hierarchical matrix normal prior in Section 3.1.
#' 
#' \deqn{A \mid \Sigma_e \sim MN(0, U^{-1}, \Sigma_e)}
#' \eqn{\Sigma_e \sim} flat
#' where U: precision matrix.
#' 
#' Then in VAR design equation (Y0 = X0 B + Z),
#' MN mean can be derived by
#' \deqn{\hat{A} = (X_0^T X_0 + U)^{-1} X_0^T Y_0}
#' and the MN scale matrix can be derived by
#' \deqn{\hat\Sigma_e = Y_0^T (I_s - X_0(X_0^T X_0 + U)^{-1} X_0^T) Y_0}
#' and IW shape by \eqn{s - m - 1}.
#' 
#' (MN: [matrix normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution), IW: [inverse-wishart](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution)).
#' 
#' @return `bvar_flat` returns an object `bvarflat` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{design}{\eqn{X_0}}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{y}{Raw input}
#'   \item{p}{Lag of VAR}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = \code{totobs} - \code{p}}
#'   \item{totobs}{Total number of the observation}
#'   \item{process}{Process: BVAR_Flat}
#'   \item{spec}{Model specification (\code{bvharspec})}
#'   \item{type}{include constant term (\code{const}) or not (\code{none})}
#'   \item{call}{Matched call}
#'   \item{mn_mean}{Location of posterior matrix normal distribution}
#'   \item{fitted.values}{Fitted values}
#'   \item{residuals}{Residuals}
#'   \item{mn_prec}{Precision matrix of posterior matrix normal distribution}
#'   \item{iw_scale}{Scale matrix of posterior inverse-wishart distribution}
#'   \item{iw_shape}{Shape of posterior inverse-wishart distribution}
#' }
#' 
#' @references 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
#' 
#' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). [https://doi:10.1080/01621459.2018.1437043](https://doi:10.1080/01621459.2018.1437043)
#' 
#' @seealso 
#' * [set_bvar_flat()] to specify the hyperparameters of BVAR flat prior.
#' * [estimate_mn_flat()] to compute BVAR flat prior and posterior.
#' 
#' @examples
#' # Perform the function using etf_vix dataset
#' \dontrun{
#'   fit <- bvar_flat(y = etf_vix, p = 2)
#'   class(fit)
#'   str(fit)
#' }
#' 
#' # Extract coef, fitted values, and residuals
#' \dontrun{
#'   coef(fit)
#'   residuals(fit)
#'   fitted(fit)
#' }
#' 
#' @order 1
#' @export
bvar_flat <- function(y, p, bayes_spec = set_bvar_flat(), include_mean = TRUE) {
  if (!all(apply(y, 2, is.numeric))) stop("Every column must be numeric class.")
  if (!is.matrix(y)) y <- as.matrix(y)
  if (!is.bvharspec(bayes_spec)) stop("Provide 'bvharspec' for 'bayes_spec'.")
  if (bayes_spec$process != "BVAR") stop("'bayes_spec' must be the result of 'set_bvar_flat()'.")
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, p, p + 1)
  name_var <- colnames(y)
  colnames(Y0) <- name_var
  X0 <- build_design(y, p)
  name_lag <- concatenate_colnames(name_var, 1:p) # in misc-r.R file
  colnames(X0) <- name_lag
  # const or none---------------------
  if (!is.logical(include_mean)) stop("'include_mean' is logical.")
  m <- ncol(y)
  k <- m * p + 1 # df
  if (!include_mean) {
    X0 <- X0[, -k] # exclude 1 column
    k <- k - 1 # df = no intercept
    name_lag <- name_lag[1:k] # colnames(X0)
  }
  # spec------------------------------
  if (is.null(bayes_spec$U)) bayes_spec$U <- diag(ncol(X0)) # identity matrix
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
    df = k, # k = mp + 1 or mp
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
  class(res) <- c("bvarflat", "bvharmod")
  res
}
