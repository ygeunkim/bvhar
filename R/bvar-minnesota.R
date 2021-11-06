#' Fitting Bayesian VAR(p) of Minnesota Prior
#' 
#' This function fits BVAR(p) with Minnesota prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param bayes_spec `r lifecycle::badge("experimental")` A BVAR model specification by [set_bvar()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' 
#' @details 
#' Minnesota prior give prior to parameters \eqn{B} (VAR matrices) and \eqn{\Sigma_e} (residual covariance).
#' 
#' \deqn{A \mid \Sigma_e \sim MN(A_0, \Omega_0, \Sigma_e)}
#' \deqn{\Sigma_e \sim IW(S_0, \alpha_0)}
#' (MN: [matrix normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution), IW: [inverse-wishart](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution))
#' 
#' \eqn{\delta_i} are related to the belief to random walk.
#' 
#' * If \eqn{\delta_i = 1} for all i, random walk prior
#' * If \eqn{\delta_i = 0} for all i, white noise prior
#' 
#' \eqn{\lambda} controls the overall tightness of the prior around these two prior beliefs.
#' 
#' * If \eqn{\lambda = 0}, the posterior is equivalent to prior and the data do not influence the estimates.
#' * If \eqn{\lambda = \infty}, the posterior mean becomes OLS estimates (VAR).
#' 
#' \eqn{\sigma_i^2 / \sigma_j^2} in Minnesota moments explain the data scales.
#' 
#' @return `bvar_minnesota` returns an object `bvarmn` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Posterior Mean matrix of Matrix Normal distribution}
#'   \item{fitted.values}{Fitted values}
#'   \item{residuals}{Residuals}
#'   \item{mn_prec}{Posterior precision matrix of Matrix Normal distribution}
#'   \item{iw_scale}{Posterior scale matrix of posterior inverse-wishart distribution}
#'   \item{iw_shape}{Posterior shape of inverse-wishart distribution (\eqn{alpha_0} - obs + 2). \eqn{\alpha_0}: nrow(Dummy observation) - k}
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
#'   \item{prior_precision}{Prior precision matrix of Matrix Normal distribution: \eqn{\Omega_0}}
#'   \item{prior_scale}{Prior scale matrix of inverse-wishart distribution: \eqn{S_0}}
#'   \item{prior_shape}{Prior shape of inverse-wishart distribution: \eqn{\alpha_0}}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' 
#' @references 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
#' 
#' KADIYALA, K.R. and KARLSSON, S. (1997), *NUMERICAL METHODS FOR ESTIMATION AND INFERENCE IN BAYESIAN VAR-MODELS*. J. Appl. Econ., 12: 99-132. [https://doi.org/10.1002/(SICI)1099-1255(199703)12:2<99::AID-JAE429>3.0.CO;2-A](https://doi.org/10.1002/(SICI)1099-1255(199703)12:2<99::AID-JAE429>3.0.CO;2-A)
#' 
#' Sims, C. A., & Zha, T. (1998). *Bayesian Methods for Dynamic Multivariate Models*. International Economic Review, 39(4), 949–968. [https://doi.org/10.2307/2527347](https://doi.org/10.2307/2527347)
#' 
#' Domenico Giannone, Michele Lenza, Giorgio E. Primiceri; *Prior Selection for Vector Autoregressions*. The Review of Economics and Statistics 2015; 97 (2): 436–451. doi: [https://doi.org/10.1162/REST_a_00483](https://doi.org/10.1162/REST_a_00483)
#' 
#' @seealso 
#' * [set_bvar()] to specify the hyperparameters of Minnesota prior.
#' * [build_ydummy()] and [build_xdummy()] to construct dummy observations.
#' * [estimate_bvar_mn()] to compute BVAR prior and posterior.
#' 
#' @examples
#' # Perform the function using etf_vix dataset
#' \dontrun{
#'   fit <- bvar_minnesota(y = etf_vix, p = 5)
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
#' @importFrom stats sd
#' @order 1
#' @export
bvar_minnesota <- function(y, p, bayes_spec = set_bvar(), include_mean = TRUE) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  # model specification---------------
  if (!is.bvharspec(bayes_spec)) {
    stop("Provide 'bvharspec' for 'bayes_spec'.")
  }
  if (bayes_spec$process != "BVAR") {
    stop("'bayes_spec' must be the result of 'set_bvar()'.")
  }
  if (is.null(bayes_spec$sigma)) {
    bayes_spec$sigma <- apply(y, 2, sd)
  }
  sigma <- bayes_spec$sigma
  m <- ncol(y)
  if (is.null(bayes_spec$delta)) {
    bayes_spec$delta <- rep(1, m)
  }
  delta <- bayes_spec$delta
  lambda <- bayes_spec$lambda
  eps <- bayes_spec$eps
  # Y0 = X0 B + Z---------------------
  Y0 <- build_y0(y, p, p + 1)
  name_var <- colnames(y)
  colnames(Y0) <- name_var
  X0 <- build_design(y, p)
  name_lag <- concatenate_colnames(name_var, 1:p) # in misc-r.R file
  colnames(X0) <- name_lag
  s <- nrow(Y0)
  k <- m * p + 1
  # dummy-----------------------------
  Yp <- build_ydummy(p, sigma, lambda, delta)
  colnames(Yp) <- name_var
  Xp <- build_xdummy(p, lambda, sigma, eps)
  colnames(Xp) <- name_lag
  # const or none---------------------
  if (!is.logical(include_mean)) {
    stop("'include_mean' is logical.")
  }
  if (!include_mean) {
    X0 <- X0[, -k] # exclude 1 column
    Tp <- nrow(Yp)
    Yp <- Yp[-Tp,] # exclude intercept block from Yp (last row)
    Xp <- Xp[-Tp, -k] # exclude intercept block from Xp (last row and last column)
    name_lag <- name_lag[-k] # colnames(X0)
    k <- k - 1 # df = no intercept
  }
  # estimate-bvar.cpp-----------------
  posterior <- estimate_bvar_mn(X0, Y0, Xp, Yp)
  # Prior-----------------------------
  prior_mean <- posterior$prior_mean # A0
  prior_prec <- posterior$prior_prec # U0
  prior_scale <- posterior$prior_scale # S0
  prior_shape <- posterior$prior_shape # a0
  # Matrix normal---------------------
  mn_mean <- posterior$mnmean # matrix normal mean
  colnames(mn_mean) <- name_var
  rownames(mn_mean) <- name_lag
  mn_prec <- posterior$mnprec # matrix normal precision
  colnames(mn_prec) <- name_lag
  rownames(mn_prec) <- name_lag
  yhat <- posterior$fitted
  colnames(yhat) <- name_var
  # Inverse-wishart-------------------
  iw_scale <- posterior$iwscale # IW scale
  colnames(iw_scale) <- name_var
  rownames(iw_scale) <- name_var
  # S3--------------------------------
  res <- list(
    # posterior------------
    coefficients = mn_mean, # posterior mean of MN
    fitted.values = yhat,
    residuals = posterior$residuals,
    mn_prec = mn_prec, # posterior precision of MN
    iw_scale = iw_scale, # posterior scale of IW
    iw_shape = prior_shape + s, # posterior shape of IW
    # variables------------
    df = k, # k = m * p + 1 or m * p
    p = p, # p
    m = m, # m = dimension of Y_t
    obs = s, # s = n - p
    totobs = nrow(y), # n = total number of sample size
    # about model----------
    call = match.call(),
    process = paste(bayes_spec$process, bayes_spec$prior, sep = "_"),
    spec = bayes_spec,
    type = ifelse(include_mean, "const", "none"),
    # prior----------------
    prior_mean = prior_mean, # A0
    prior_precision = prior_prec, # U0 = (Omega)^{-1}
    prior_scale = prior_scale, # S0
    prior_shape = prior_shape, # a0
    # data-----------------
    y0 = Y0,
    design = X0,
    y = y
  )
  class(res) <- c("bvarmn", "normaliw", "bvharmod")
  res
}
