#' @keywords internal
#' @useDynLib bvhar, .registration = TRUE
#' @details 
#' `r lifecycle::badge("experimental")`
#' 
#' The bvhar package provides function to analyze and forecast multivariate time series data via vector autoregressive modelling.
#' Here, vector autoregressive modelling includes:
#' * Vector autoregressive (VAR) model: [var_lm()]
#' * Vector heterogeneous autoregressive (VHAR) model: [vhar_lm()]
#' * Bayesian VAR (BVAR) model: [bvar_minnesota()], [bvar_flat()]
#' * Bayesian VHAR (BVHAR) model: [bvhar_minnesota()]
#' 
#' Each function returns S3 class `varlse`, `vharlse`, `bvarmn`, `bvarflat`, and `bvharmn`.
#' As dealing with other statistical functions (such as [stats::lm()] and [stats::glm()]),
#' users can analyze time series with S3 methods:
#' * Extract: [coef.varlse()], [residuals.varlse()], [fitted.varlse()]
#' * log-likelihood: [logLik.varlse()]
#' * Forecasting: [predict.varlse()]
#' @references 
#' Litterman, R. B. (1986). *Forecasting with Bayesian Vector Autoregressions: Five Years of Experience*. Journal of Business & Economic Statistics, 4(1), 25. [https://doi:10.2307/1391384](https://doi:10.2307/1391384)
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1). [https://doi:10.1002/jae.1137](https://doi:10.1002/jae.1137)
#' 
#' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526). [https://doi:10.1080/01621459.2018.1437043](https://doi:10.1080/01621459.2018.1437043)
#' 
#' Baek, C. and Park, M. (2021). *Sparse vector heterogeneous autoregressive modeling for realized volatility*. J. Korean Stat. Soc. 50, 495–510. [https://doi.org/10.1007/s42952-020-00090-5](https://doi.org/10.1007/s42952-020-00090-5)
"_PACKAGE"

# The following block is used by usethis to automatically manage
# roxygen namespace tags. Modify with care!
## usethis namespace: start
#' @importFrom lifecycle deprecated is_present deprecate_warn deprecate_soft
#' @importFrom Rcpp sourceCpp
## usethis namespace: end
NULL
