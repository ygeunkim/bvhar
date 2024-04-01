#' Dynamic Spillover Indices Plot
#'
#' @param object `bvhardynsp` object
#' @param type Index to draw
#' @param row_facet `nrow` of [ggplot2::facet_wrap()]
#' @param col_facet `ncol` of [ggplot2::facet_wrap()]
#' @param ... Additional
#'
#' @importFrom tidyr pivot_longer
#' @importFrom ggplot2 ggplot aes geom_path
#' @export
autoplot.bvhardynsp <- function(object, type = c("tot", "to", "from", "net"), row_facet = NULL, col_facet = NULL, ...) {
  type <- match.arg(type)
  switch(type,
    "tot" = {
      data.frame(
        id = object$index,
        y = object$tot
      ) %>%
        ggplot(aes(x = id, y = y)) +
          geom_path()
    },
    "to" = {
      cbind(id = object$index, object$to) %>%
        pivot_longer(-id, names_to = "series", values_to = "value") %>%
        ggplot(aes(x = id)) +
        # geom_ribbon(aes(ymin = 0, ymax = value)) +
        geom_path(aes(y = value)) +
        facet_wrap(series ~ ., nrow = row_facet, ncol = col_facet)
    },
    "from" = {
      cbind(id = object$index, object$from) %>%
        pivot_longer(-id, names_to = "series", values_to = "value") %>%
        ggplot(aes(x = id)) +
        # geom_ribbon(aes(ymin = 0, ymax = value)) +
        geom_path(aes(y = value)) +
        facet_wrap(series ~ ., nrow = row_facet, ncol = col_facet)
    },
    "net" = {
      cbind(id = object$index, object$net) %>%
        pivot_longer(-id, names_to = "series", values_to = "value") %>%
        ggplot(aes(x = id)) +
        # geom_ribbon(aes(ymin = 0, ymax = value)) +
        geom_path(aes(y = value)) +
        facet_wrap(series ~ ., nrow = row_facet, ncol = col_facet)
    },
    stop("not yet")
  )
}
