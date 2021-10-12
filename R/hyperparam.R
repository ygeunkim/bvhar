#' Hyperparameters for Bayesian Models
#' 
#' Set hyperparameters of Bayesian VAR and VHAR models.
#' 
#' @param sigma Standard error vector for each variable (Default: sd of each variable)
#' @param lambda Tightness of the prior around a random walk or white noise (Default: .1)
#' @param delta Persistence (Litterman sets 1 = random walk prior (default: rep(1, number of variables)), White noise prior = 0)
#' @param eps Very small number (Default: 1e-04)
#' @details 
#' * Missing arguments will be set to be default values in each model function mentioned above.
#' * `set_bvar` sets hyperparameters for [bvar_minnesota()].
#' @return Every function returns `bvharspec` [class].
#' It is the list of which the components are the same as the arguments provided.
#' @examples 
#' # Minnesota BVAR specification
#' # m = 3
#' \dontrun{
#'   bvar_spec <- set_bvar(
#'     sigma = c(.03, .02, .01), # Sigma = diag(.03^2, .02^2, .01^2)
#'     lambda = .2, # lambda = .2
#'     delta = rep(.1, 3), # delta1 = .1, delta2 = .1, delta3 = .1
#'     eps = 1e-04 # eps = 1e-04
#'   )
#'   class(bvar_spec)
#'   str(bvar_spec)
#' }
#' @order 1
#' @export
set_bvar <- function(sigma, lambda = .1, delta, eps = 1e-04) {
  if (missing(sigma)) sigma <- NULL
  if (missing(delta)) delta <- NULL
  bvar_param <- list(
    process = "BVAR",
    prior = "Minnesota",
    sigma = sigma,
    lambda = lambda,
    delta = delta,
    eps = eps
  )
  class(bvar_param) <- "bvharspec"
  bvar_param
}

#' @rdname set_bvar
#' @param U Positive definite matrix. By default, identity matrix of dimension ncol(X0)
#' @details 
#' * `set_bvar_flat` sets hyperparameters for [bvar_flat()].
#' @examples 
#' # Flat BVAR specification
#' # m = 3
#' # p = 5 with constant term
#' # U = 500 * I(mp + 1)
#' \dontrun{
#'   bvar_spec <- set_bvar_flat(U = 500 * diag(16))
#'   class(bvar_spec)
#'   str(bvar_spec)
#' }
#' @order 1
#' @export
set_bvar_flat <- function(U) {
  if (missing(U)) U <- NULL
  bvar_param <- list(
    process = "BVAR",
    prior = "Flat",
    U = U
  )
  class(bvar_param) <- "bvharspec"
  bvar_param
}

#' @rdname set_bvar
#' @param sigma Standard error vector for each variable (Default: sd)
#' @param lambda Tightness of the prior around a random walk or white noise (Default: .1)
#' @param delta Persistence (Default: Litterman sets 1 = random walk prior, White noise prior = 0)
#' @param eps Very small number (Default: 1e-04)
#' @details 
#' * `set_bvhar` sets hyperparameters for [bvhar_minnesota()] with VAR-type Minnesota prior.
#' @examples 
#' # VAR-type Minnesota BVHAR specification
#' # m = 3
#' \dontrun{
#'   bvhar_spec <- set_bvhar(
#'     sigma = c(.03, .02, .01), # Sigma = diag(.03^2, .02^2, .01^2)
#'     lambda = .2, # lambda = .2
#'     delta = rep(.1, 3), # delta1 = .1, delta2 = .1, delta3 = .1
#'     eps = 1e-04 # eps = 1e-04
#'   )
#'   class(bvhar_spec)
#'   str(bvhar_spec)
#' }
#' @order 1
#' @export
set_bvhar <- function(sigma, lambda = .1, delta, eps = 1e-04) {
  if (missing(sigma)) sigma <- NULL
  if (missing(delta)) delta <- NULL
  bvhar_param <- list(
    process = "BVHAR",
    prior = "MN_VAR",
    sigma = sigma,
    lambda = lambda,
    delta = delta,
    eps = eps
  )
  class(bvhar_param) <- "bvharspec"
  bvhar_param
}

#' @rdname set_bvar
#' @param sigma Standard error vector for each variable (Default: sd)
#' @param lambda Tightness of the prior around a random walk or white noise (Default: .1)
#' @param eps Very small number (Default: 1e-04)
#' @param daily Same as delta in VHAR type (Default: 1 as Litterman)
#' @param weekly Fill the second part in the first block (Default: 1)
#' @param monthly Fill the third part in the first block (Default: 1)
#' @details 
#' * `set_weight_bvhar` sets hyperparameters for [bvhar_minnesota()] with VHAR-type Minnesota prior.
#' @examples 
#' # HAR-type Minnesota BVHAR specification
#' # m = 3
#' \dontrun{
#'   bvhar_spec <- set_weight_bvhar(
#'     sigma = c(.03, .02, .01), # Sigma = diag(.03^2, .02^2, .01^2)
#'     lambda = .2, eps = 1e-04, # lambda = .2
#'     eps = 1e-04, # eps = 1e-04
#'     daily = rep(.2, 3), # daily1 = .2, daily2 = .2, daily3 = .2
#'     weekly = rep(.1, 3), # weekly1 = .1, weekly2 = .1, weekly3 = .1
#'     monthly = rep(.05, 3) # monthly1 = .05, monthly2 = .05, monthly3 = .05
#'   )
#'   class(bvhar_spec)
#'   str(bvhar_spec)
#' }
#' @order 1
#' @export
set_weight_bvhar <- function(sigma,
                             lambda = .1,
                             eps = 1e-04,
                             daily,
                             weekly,
                             monthly) {
  if (missing(sigma)) sigma <- NULL
  if (missing(daily)) daily <- NULL
  if (missing(weekly)) weekly <- NULL
  if (missing(monthly)) monthly <- NULL
  bvhar_param <- list(
    process = "BVHAR",
    prior = "MN_VHAR",
    sigma = sigma,
    lambda = lambda,
    eps = eps,
    daily = daily,
    weekly = weekly,
    monthly = monthly
  )
  class(bvhar_param) <- "bvharspec"
  bvhar_param
}

#' @rdname set_bvar
#' @param x `bvharspec` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharspec <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(paste0("Model Specification for ", x$process, "\n\n"))
  cat("Parameters: Coefficent matrice and Covariance matrix\n")
  cat(paste0("Prior: ", x$prior, "\n"))
  fit_func <- switch(
    x$prior,
    "Minnesota" = "?bvar_minnesota",
    "Flat" = "?bvar_flat",
    "MN_VAR" = "?bvhar_minnesota",
    "MN_VHAR" = "?bvhar_minnesota",
    stop("Invalid 'x$prior' element")
  )
  cat(paste0("# Type '", fit_func, "' in the console for some help.", "\n"))
  cat("========================================================\n\n")
  param <- x[!(names(x) %in% c("process", "prior"))]
  for (i in seq_along(param)) {
    cat(paste0("Setting for '", names(param)[i], "':\n"))
    if (is.matrix(param[[i]])) {
      type <- "a" # not large
      if (nrow(param[[i]]) > 10 & ncol(param[[i]]) > 5) {
        type <- "b" # both large
      } else if (nrow(param[[i]]) > 10 & ncol(param[[i]]) <= 5) {
        type <- "c" # large row
      } else if (nrow(param[[i]]) <= 10 & ncol(param[[i]]) > 5) {
        type <- "d" # large column
      }
      switch(
        type,
        "a" = {
          print.default(
            param[[i]],
            digits = digits,
            print.gap = 2L,
            quote = FALSE
          )
          cat("\n")
        },
        "b" = {
          cat(
            paste0("# A matrix: "), 
            paste(nrow(param[[i]]), "x", ncol(param[[i]])),
            "\n"
          )
          print.default(
            param[[i]][1:10, 1:5],
            digits = digits,
            print.gap = 2L,
            quote = FALSE
          )
          cat(paste0("# ... with ", nrow(param[[i]]) - 10, " more rows", "\n"))
        },
        "c" = {
          print.default(
            param[[i]][1:10,],
            digits = digits,
            print.gap = 2L,
            quote = FALSE
          )
          cat(paste0("# ... with ", nrow(param[[i]]) - 10, " more rows", "\n"))
        },
        "d" = {
          cat(
            paste0("# A matrix: "), 
            paste(nrow(param[[i]]), "x", ncol(param[[i]])), 
            "\n"
          )
          print.default(
            param[[i]][1:10, 1:5],
            digits = digits,
            print.gap = 2L,
            quote = FALSE
          )
          cat("\n")
        }
      )
    } else {
      print.default(
        param[[i]],
        digits = digits,
        print.gap = 2L,
        quote = FALSE
      )
      cat("\n")
    }
  }
}

#' @rdname set_bvar
#' @param x `bvharspec` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvharspec <- function(x, ...) {
  print(x)
}

registerS3method(
  "knit_print", "bvharspec",
  knit_print.bvharspec,
  envir = asNamespace("knitr")
)
