#' @noRd
concatenate_colnames <- function(var_name, prefix) {
  lapply(
    prefix,
    function(lag) paste(var_name, lag, sep = "_")
  ) %>% 
    unlist() %>% 
    c(., "const")
}

#' Split Coefficient Matrix into List
#' 
#' Split `coefficients` into matrix list.
#' 
#' @param object `bvharmod` object
#' @param ... not used
#' 
#' @noRd
split_coef <- function(object, ...) {
  UseMethod("split_coef", object)
}

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
