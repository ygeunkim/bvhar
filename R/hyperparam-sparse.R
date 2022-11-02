#' Stochastic Search Variable Selection (SSVS) Hyperparameter for Coefficients Matrix and Cholesky Factor
#' 
#' Set SSVS hyperparameters for VAR coefficient matrix and Cholesky factor.
#' 
#' @param coef_spike Standard deviance for Spike normal distribution (See Details).
#' @param coef_slab Standard deviance for Slab normal distribution (See Details).
#' @param coef_mixture Bernoulli parameter for sparsity proportion (See Details).
#' @param coef_non Hyperparameter for constant term
#' @param shape Gamma shape parameters for precision matrix (See Details).
#' @param rate Gamma rate parameters for precision matrix (See Details).
#' @param chol_spike Standard deviance for Spike normal distribution, in the cholesky factor (See Details).
#' @param chol_slab Standard deviance for Slab normal distribution, in the cholesky factor (See Details).
#' @param chol_mixture Bernoulli parameter for sparsity proportion, in the cholesky factor (See Details).
#' @details 
#' Let \eqn{\alpha} be the vectorized coefficient, \eqn{\alpha = vec(A)}.
#' Spike-slab prior is given using two normal distributions.
#' \deqn{\alpha_j \mid \gamma_j \sim (1 - \gamma_j) N(0, \tau_{0j}^2) + \gamma_j N(0, \tau_{1j}^2)}
#' As spike-slab prior itself suggests, set \eqn{\tau_{0j}} small (point mass at zero: spike distribution)
#' and set \eqn{\tau_{1j}} large (symmetric by zero: slab distribution).
#' 
#' \eqn{\gamma_j} is the proportion of the nonzero coefficients and it follows
#' \deqn{\gamma_j \sim Bernoulli(p_j)}
#' 
#' * `coef_spike`: \eqn{\tau_{0j}}
#' * `coef_slab`: \eqn{\tau_{1j}}
#' * `coef_mixture`: \eqn{p_j}
#' * \eqn{j = 1, \ldots, mk}: vectorized format corresponding to coefficient matrix
#' * If one value is provided, model function will read it by replicated value.
#' 
#' Next for precision matrix \eqn{\Sigma_e^{-1}}, SSVS applies Cholesky decomposition.
#' \deqn{\Sigma_e^{-1} = \Psi \Psi^T}
#' where \eqn{\Psi = \{\psi_{ij}\}} is upper triangular.
#' 
#' Diagonal components follow the gamma distribution.
#' \deqn{\psi_{jj}^2 \sim Gamma(shape = a_j, rate = b_j)}
#' For each row of off-diagonal (upper-triangular) components, we apply spike-slab prior again.
#' \deqn{\psi_{ij} \mid w_{ij} \sim (1 - w_{ij}) N(0, \kappa_{0,ij}^2) + w_{ij} N(0, \kappa_{1,ij}^2)}
#' \deqn{w_{ij} \sim Bernoulli(q_{ij})}
#' 
#' * `shape`: \eqn{a_j}
#' * `rate`: \eqn{b_j}
#' * `chol_spike`: \eqn{\kappa_{0,ij}}
#' * `chol_slab`: \eqn{\kappa_{1,ij}}
#' * `chol_mixture`: \eqn{q_{ij}}
#' * \eqn{j = 1, \ldots, mk}: vectorized format corresponding to coefficient matrix
#' * \eqn{i = 1, \ldots, j - 1} and \eqn{j = 2, \ldots, m}: \eqn{\eta = (\psi_{12}, \psi_{13}, \psi_{23}, \psi_{14}, \ldots, \psi_{34}, \ldots, \psi_{1m}, \ldots, \psi_{m - 1, m})^T}
#' * `chol_` arguments can be one value for replication, vector, or upper triangular matrix.
#' @references 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
#' 
#' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
#' @order 1
#' @export
set_ssvs <- function(coef_spike = .1, 
                     coef_slab = 5, 
                     coef_mixture = .5,
                     coef_non = .1,
                     shape = .01,
                     rate = .01,
                     chol_spike = .1,
                     chol_slab = 5,
                     chol_mixture = .5) {
  if (!(is.vector(coef_spike) && 
        is.vector(coef_slab) && 
        is.vector(coef_mixture) &&
        is.vector(shape) &&
        is.vector(rate))) {
    stop("'coef_spike', 'coef_slab', 'coef_mixture', 'shape', and 'rate' be a vector.")
  }
  if (length(coef_non) != 1) {
    stop("'coef_non' should be length 1 numeric.")
  }
  if (coef_non < 0) {
    stop("'coef_non' should be positive.")
  }
  if (!(is.numeric(chol_spike) ||
        is.vector(chol_spike) || 
        is.matrix(chol_spike) ||
        is.numeric(chol_slab) ||
        is.vector(chol_slab) ||
        is.matrix(chol_slab) ||
        is.numeric(chol_mixture) ||
        is.vector(chol_mixture) ||
        is.matrix(chol_mixture))) {
    stop("'chol_spike', 'chol_slab', and 'chol_mixture' should be a vector or upper triangular matrix.")
  }
  # coefficients---------------------
  coef_param <- list(
    coef_spike = coef_spike,
    coef_slab = coef_slab,
    coef_mixture = coef_mixture,
    coef_non = coef_non
  )
  len_param <- sapply(coef_param, length)
  if (length(unique(len_param[len_param != 1])) > 1) {
    stop("The length of 'coef_spike', 'coef_slab', and 'coef_mixture' should be the same.")
  }
  # cholesky factor-------------------
  if (is.matrix(chol_spike)) {
    if (any(chol_spike[lower.tri(chol_spike, diag = TRUE)] != 0)) {
      stop("If 'chol_spike' is a matrix, it should be an upper triangular form.")
    }
    chol_spike <- chol_spike[upper.tri(chol_spike, diag = FALSE)]
  }
  if (is.matrix(chol_slab)) {
    if (any(chol_slab[lower.tri(chol_slab, diag = TRUE)] != 0)) {
      stop("If 'chol_slab' is a matrix, it should be an upper triangular form.")
    }
    chol_slab <- chol_slab[upper.tri(chol_slab, diag = FALSE)]
  }
  if (is.matrix(chol_mixture)) {
    if (any(chol_mixture[lower.tri(chol_mixture, diag = TRUE)] != 0)) {
      stop("If 'chol_mixture' is a matrix, it should be an upper triangular form.")
    }
    chol_mixture <- chol_mixture[upper.tri(chol_mixture, diag = FALSE)]
  }
  chol_param <- list(
    process = "BVAR",
    prior = "SSVS",
    shape = shape,
    rate = rate,
    chol_spike = chol_spike, 
    chol_slab = chol_slab,
    chol_mixture = chol_mixture
  )
  len_param <- sapply(chol_param, length)
  len_gamma <- len_param[1:2]
  len_eta <- len_param[3:5]
  if (length(unique(len_gamma[len_gamma != 1])) > 1) {
    stop("The length of 'shape' and 'rate' should be the same.")
  }
  if (length(unique(len_eta[len_eta != 1])) > 1) {
    stop("The size of 'chol_spike', 'chol_slab', and 'chol_mixture' should be the same.")
  }
  res <- append(coef_param, chol_param)
  class(res) <- "ssvsinput"
  res
}

#' Initial Parameters of Stochastic Search Variable Selection (SSVS) Model
#' 
#' Set initial parameters before starting Gibbs sampler for SSVS.
#' 
#' @param init_coef Initial coefficient matrix.
#' @param init_coef_dummy Initial indicator matrix (1-0) corresponding to each component of coefficient.
#' @param init_chol Initial cholesky factor (upper triangular).
#' @param init_chol_dummy Initial indicator matrix (1-0) corresponding to each component of cholesky factor.
#' @details 
#' Set SSVS initialization for the VAR model.
#' 
#' * `init_coef`: (kp + 1) x m \eqn{A} coefficient matrix.
#' * `init_coef_dummy`: kp x m \eqn{\Gamma} dummy matrix to restrict the coefficients.
#' * `init_chol`: k x k \eqn{\Psi} upper triangular cholesky factor, which \eqn{\Psi \Psi^\intercal = \Sigma_e^{-1}}.
#' * `init_chol_dummy`: k x k \eqn{\Omega} upper triangular dummy matrix to restrict the cholesky factor.
#' 
#' Denote that `init_chol` and `init_chol_dummy` should be upper_triangular or the function gives error.
#' @references 
#' George, E. I., Sun, D., & Ni, S. (2008). *Bayesian stochastic search for VAR model restrictions. Journal of Econometrics*, 142(1), 553–580. doi:[10.1016/j.jeconom.2007.08.017](https://doi.org/10.1016/j.jeconom.2007.08.017)
#' 
#' Koop, G., & Korobilis, D. (2009). *Bayesian Multivariate Time Series Methods for Empirical Macroeconomics*. Foundations and Trends® in Econometrics, 3(4), 267–358. doi:[10.1561/0800000013](http://dx.doi.org/10.1561/0800000013)
#' @order 1
#' @export
init_ssvs <- function(init_coef, init_coef_dummy, init_chol, init_chol_dummy) {
  dim_design <- nrow(init_coef) # kp(+1)
  dim_data <- ncol(init_coef) # k = dim
  if (!(nrow(init_coef_dummy) == dim_design && ncol(init_coef_dummy) == dim_data)) {
    if (!(nrow(init_coef_dummy) == dim_design - 1 && ncol(init_coef_dummy) == dim_data)) {
      stop("Invalid dimension of 'init_coef_dummy'.")
    }
  }
  if (!(nrow(init_chol) == dim_data && ncol(init_chol) == dim_data)) {
    stop("Invalid dimension of 'init_chol'.")
  }
  if (any(init_chol[lower.tri(init_chol, diag = TRUE)] != 0)) {
    stop("'init_chol' should be upper triangular matrix.")
  }
  if (!(nrow(init_chol_dummy) == dim_data || ncol(init_chol_dummy) == dim_data)) {
    stop("Invalid dimension of 'init_chol_sparse'.")
  }
  res <- list(
    process = "BVAR",
    prior = "SSVS",
    init_coef = init_coef,
    init_coef_dummy = init_coef_dummy,
    init_chol = init_chol,
    init_chol_dummy = init_chol_dummy
  )
  class(res) <- "ssvsinit"
  res
}

#' @rdname set_ssvs
#' @param x `ssvsinput`
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.ssvsinput <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(paste0("Model Specification for ", x$process, " with ", x$prior, " Prior", "\n\n"))
  cat("Parameters: Coefficent matrix, Cholesky Factor, and Each Restriction Dummy\n")
  cat(paste0("Prior: ", x$prior, "\n"))
  fit_func <- switch(
    x$process,
    "BVAR" = "?bvar_ssvs",
    "BVHAR" = "?bvhar_ssvs",
    stop("Invalid 'x$prior' element")
  )
  cat(paste0("# Type '", fit_func, "' in the console for some help.", "\n"))
  cat("========================================================\n")
  param <- x[!(names(x) %in% c("process", "prior"))]
  for (i in seq_along(param)) {
    cat(paste0("Setting for '", names(param)[i], "':\n"))
    if (is.matrix(param[[i]])) {
      type <- "a"
    } else if (length(param[[i]]) == 1) {
      type <- "b"
    } else {
      type <- "c"
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
      },
      "b" = {
        if (grepl(pattern = "^coef", names(param)[i]) && names(param)[i] != "coef_non") {
          pseudo_param <- paste0("rep(", param[[i]], ", dim^2 * p)") # coef_
        } else if (grepl(pattern = "^chol", names(param)[i])) {
          pseudo_param <- paste0("rep(", param[[i]], ", dim * (dim - 1) / 2)") # chol_
        } else {
          pseudo_param <- paste0("rep(", param[[i]], ", dim)") # shape and rate
        }
        print.default(
          pseudo_param,
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
      },
      "c" = {
        print.default(
          param[[i]],
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
      }
    )
    cat("\n")
  }
  cat("--------------------------------------------------------------\n")
  cat("dim: time series dimension, p: VAR order")
}

#' @rdname set_ssvs
#' @param x `ssvsinput` object
#' @param ... not used
#' @order 3
#' @export
knit_print.ssvsinput <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "ssvsinput",
  knit_print.ssvsinput,
  envir = asNamespace("knitr")
)

#' @rdname init_ssvs
#' @param x `ssvsinit`
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.ssvsinit <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(paste0("Gibbs Sampler Initialization for ", x$process, " with ", x$prior, " Prior", "\n\n"))
  cat("Parameters: Coefficent matrix, Cholesky Factor, and Each Restriction Dummy\n")
  # cat(paste0("Prior: ", x$prior, "\n"))
  fit_func <- switch(
    x$process,
    "BVAR" = "?bvar_ssvs",
    "BVHAR" = "?bvhar_ssvs",
    stop("Invalid 'x$prior' element")
  )
  cat(paste0("# Type '", fit_func, "' in the console for some help.", "\n"))
  cat("========================================================\n")
  param <- x[!(names(x) %in% c("process", "prior"))]
  for (i in seq_along(param)) {
    cat(paste0("Initialization for '", names(param)[i], "':\n"))
    type <- "a" # not large
    if (nrow(param[[i]]) > 7 & ncol(param[[i]]) > 6) {
      type <- "b" # both large
    } else if (nrow(param[[i]]) > 7 & ncol(param[[i]]) <= 6) {
      type <- "c" # large row
    } else if (nrow(param[[i]]) <= 7 & ncol(param[[i]]) > 6) {
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
          param[[i]][1:7, 1:6],
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
        cat(paste0("# ... with ", nrow(param[[i]]) - 7, " more rows", "\n"))
      },
      "c" = {
        print.default(
          param[[i]][1:7,],
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
        cat(paste0("# ... with ", nrow(param[[i]]) - 7, " more rows", "\n"))
      },
      "d" = {
        cat(
          paste0("# A matrix: "), 
          paste(nrow(param[[i]]), "x", ncol(param[[i]])), 
          "\n"
        )
        print.default(
          param[[i]][1:7, 1:6],
          digits = digits,
          print.gap = 2L,
          quote = FALSE
        )
        cat("\n")
      }
    )
  }
}

#' @rdname init_ssvs
#' @param x `ssvsinit` object
#' @param ... not used
#' @order 3
#' @export
knit_print.ssvsinit <- function(x, ...) {
  print(x)
}

#' @export
registerS3method(
  "knit_print", "ssvsinit",
  knit_print.ssvsinit,
  envir = asNamespace("knitr")
)
