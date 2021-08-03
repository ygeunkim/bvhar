#' @export
concatenate_colnames <- function(var_name, prefix) {
  lapply(
    prefix,
    function(lag) paste(var_name, lag, sep = "_")
  ) %>% 
    unlist() %>% 
    c(., "const")
}