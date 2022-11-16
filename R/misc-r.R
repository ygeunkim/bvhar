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
#' @param num_iter MCMC iteration number
#' @param dim_data Data dimension
#' @param chain The number of the chains
#' @noRd
split_psirecord <- function(x, chain = 1, varname = "cholesky") {
  # res <- 
  #   x %>% 
  #   split.data.frame(gl(num_iter, dim_data))
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
