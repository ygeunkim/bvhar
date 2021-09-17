#' Hyperpriors for Minnesota BVAR
#' 
#' Sets hyperprior for [bvar_minnesota()].
#' 
#' @param sigma standard error for each variable (Default: sd)
#' @param lambda tightness of the prior around a random walk or white noise (Default: .1)
#' @param delta Persistence (Litterman sets 1 = random walk prior, Default: White noise prior = 0)
#' @param eps very small number
#' 
#' @export
set_bvar <- function(sigma, lambda = .1, delta, eps = 1e-04) {
  list(
    sigma = sigma,
    lambda = lambda,
    delta = delta,
    eps = eps
  )
}

#' Hyperpriors for VAR-type Minnesota BHVAR
#' 
#' Sets hyperprior for [bvhar_minnesota()] with `mn_type = "VAR"`.
#' 
#' @param sigma Standard error vector for each variable
#' @param lambda Tightness of the prior around a random walk or white noise
#' @param eps very small number
#' @param daily Same as delta in VHAR type
#' @param weekly Fill the second part in the first block
#' @param monthly Fill the third part in the first block
#' 
#' @export
set_bvhar_mn <- function(sigma, lambda = .1, delta, eps = 1e-04) {
  list(
    prior = "VAR",
    sigma = sigma,
    lambda = lambda,
    delta = delta,
    eps = eps
  )
}

#' Hyperpriors for HAR-type Minnesota BHVAR
#' 
#' Sets hyperprior for [bvhar_minnesota()] with `mn_type = "VHAR"`.
#' 
#' @param sigma Standard error vector for each variable
#' @param lambda Tightness of the prior around a random walk or white noise
#' @param delta Prior belief about white noise (Litterman sets 1: default)
#' @param eps very small number
#' 
#' @export
set_bvhar_har <- function(sigma,
                          lambda = .1,
                          eps = 1e-04,
                          daily,
                          weekly,
                          monthly) {
  list(
    prior = "VHAR",
    sigma = sigma,
    lambda = lambda,
    eps = eps,
    daily = daily,
    weekly = weekly,
    monthly = monthly
  )
}
