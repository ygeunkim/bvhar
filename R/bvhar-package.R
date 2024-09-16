#' @keywords internal
#' @useDynLib bvhar, .registration = TRUE
#' @details 
#' The bvhar package provides function to analyze and forecast multivariate time series data via vector autoregressive modeling.
#' Here, vector autoregressive modelling includes:
#' * Vector autoregressive (VAR) model: [var_lm()]
#' * Vector heterogeneous autoregressive (VHAR) model: [vhar_lm()]
#' * Bayesian VAR (BVAR) model: [var_bayes()]
#' * Bayesian VHAR (BVHAR) model: [vhar_bayes()]
#' @references 
#' Kim, Y. G., and Baek, C. (2024). *Bayesian vector heterogeneous autoregressive modeling*. Journal of Statistical Computation and Simulation, 94(6), 1139-1157.
#' 
#' Kim, Y. G., and Baek, C. (n.d.). Working paper.
"_PACKAGE"

# The following block is used by usethis to automatically manage
# roxygen namespace tags. Modify with care!
## usethis namespace: start
#' @importFrom lifecycle deprecated is_present deprecate_warn deprecate_soft
#' @importFrom Rcpp evalCpp
## usethis namespace: end
NULL
