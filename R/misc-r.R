#' @noRd
concatenate_colnames <- function(var_name, prefix, include_mean = TRUE) {
  nm <- 
    lapply(
      prefix,
      function(lag) paste(var_name, lag, sep = "_")
    ) %>% 
    unlist()
  if (!include_mean) {
    return(nm)
  }
  c(nm, "const")
}

#' Splitting Coefficient Matrix into List
#' 
#' Split `coefficients` into matrix list.
#' 
#' @param object `bvharmod` object
#' @param ... not used
#' @details 
#' Each result of [var_lm()], [vhar_lm()], [bvar_minnesota()], [bvar_flat()], and [bvhar_minnesota()] is a subclass of `bvharmod`.
#' For example,
#' `c("varlse", "bvharmod")`.
#' @export
split_coef <- function(object, ...) {
  UseMethod("split_coef", object)
}

#' @rdname split_coef
#' @export
split_coef.bvharmod <- function(object, ...) {
  switch(
    object$type,
    "const" = {
      split.data.frame(object$coefficients[-object$df,], gl(object$p, object$m)) %>% 
        lapply(t)
    },
    "none" = {
      split.data.frame(object$coefficients, gl(object$p, object$m)) %>% 
        lapply(t)
    }
  )
}

#' @rdname split_coef
#' @export
split_coef.bvharirf <- function(object, ...) {
  irf_mat <- object$coefficients
  split.data.frame(irf_mat, gl(object$lag_max + 1, ncol(irf_mat)))
}

#' Changing 3d initial array Input to List
#' 
#' This function changes 3d array of [init_ssvs()] function to list.
#' 
#' @param init_array potentially 3d array initial input
#' 
#' @noRd
change_to_list <- function(init_array) {
  if (length(dim(init_array)) == 3) {
    lapply(
      seq_len(dim(init_array)[3]),
      function(k) init_array[,, k]
    )
  } else {
    init_array
  }
}

#' Checking if the Parallel Initial List are not Identical
#' 
#' This function checks if the list of parallel initial matrices are identical.
#' 
#' @param init_list List of parallel initial matrix
#' @param case Check dimension (`"dim"`) or values (`"values"`).
#' 
#' @noRd
isnot_identical <- function(init_list, case = c("dim", "values")) {
  case <- match.arg(case)
  switch(
    case,
    "dim" = {
      if (length(unique(lapply(init_list, dim))) != 1) {
        stop(paste0(
          "Dimension of '",
          deparse(substitute(init_list)),
          "' across every chain should be the same."
        ))
      }
    },
    "values" = {
      if (any(unlist(
        lapply(
          seq_along(init_list)[-1], 
          function(i) identical(init_list[[1]], init_list[[i]])
        )
      ))) {
        warning(paste0(
          "Initial setting of '",
          deparse(substitute(init_list)),
          "' in each chain is recommended to be differed."
        ))
      }
    }
  )
}

#' Processing Multiple Chain Record Result Matrix from `RcppEigen`
#' 
#' Preprocess multiple chain record matrix for [posterior::posterior] package.
#' 
#' @param x Parameter matrix
#' @param chain The number of the chains
#' @param param_name The name of the parameter
#' @details 
#' Internal Gibbs sampler function gives multiple chain results by row-stacked form.
#' This function processes the matrix appropriately for [posterior::draws_array()],
#' i.e. iteration x chain x variable.
#' @noRd
split_paramarray <- function(x, chain, param_name) {
  num_var <- ncol(x) / chain
  res <- 
    split.data.frame(t(x), gl(num_var, 1, ncol(x))) %>%
    lapply(t) %>%
    unlist() %>% 
    array(
      dim = c(nrow(x), chain, num_var),
      dimnames = list(
        iteration = seq_len(nrow(x)),
        chain = seq_len(chain),
        variable = paste0(param_name, "[", seq_len(num_var), "]")
      )
    )
  res
}

#' Processing 3d Matrix from `RcppEigen`
#' 
#' Preprocess 3d record matrix
#' 
#' @param x Parameter matrix
#' @param chain The number of the chains
#' @noRd
split_psirecord <- function(x, chain = 1, varname = "cholesky") {
  res <- 
    x %>% 
    split.data.frame(gl(nrow(x) / ncol(x), ncol(x)))
  if (chain == 1) {
    return(res)
  } else {
    res <- lapply(
      res,
      function(y) {
        num_var <- ncol(y) / chain
        split.data.frame(t(y), gl(num_var, 1, ncol(y))) %>% 
          lapply(t) %>% 
          unlist() %>% 
          array(
            dim = c(nrow(y), chain, num_var),
            dimnames = list(
              iteration = seq_len(nrow(y)),
              chain = seq_len(chain),
              variable = paste0(param_name = varname, "[", seq_len(num_var), "]")
            )
          )
      }
    )
  }
  res
}

#' Get Gamma Distribution Parameters
#' 
#' Compute Gamma distribution parameters from its mode and sd
#' 
#' @param mode Mode of Gamma distribution
#' @param sd Standard deviation of Gamma distribution
#' @details 
#' Parameters of Gamma distribution is computed using [quadratic formula](https://en.wikipedia.org/wiki/Quadratic_formula).
#' @noRd
get_gammaparam <- function(mode, sd) {
  shp <- (
    (2 + mode^2 / sd^2) + 
      sqrt((2 + mode^2 / sd^2)^2 - 4)
  ) / 2
  list(
    shape = shp,
    rate = sqrt(shp) / sd
  )
}
