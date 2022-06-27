#' @noRd
concatenate_colnames <- function(var_name, prefix) {
  lapply(
    prefix,
    function(lag) paste(var_name, lag, sep = "_")
  ) %>% 
    unlist() %>% 
    c(., "const")
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
