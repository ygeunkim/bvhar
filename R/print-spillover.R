#' @rdname spillover
#' @param x `bvharspillover` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvharspillover <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Directional spillovers:\n")
  cat("variables (i) <- shocks (j)\n")
  cat("========================\n")
  if (is.list(x$connect)) {
    connect <- x$connect$mean
    to <- x$to$mean
    from <- x$from$mean
    tot <- x$tot$mean
  } else {
    connect <- x$connect
    to <- x$to
    from <- x$from
    tot <- x$tot
  }
  connect <- rbind(connect, "to_spillovers" = to)
  connect <- cbind(connect, "from_spillovers" = c(from, tot))
  print(
    connect,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\n*Lower right corner: Total spillover\n")
  cat("------------------------\n")
  cat("Net spillovers:\n")
  if (is.list(x$net)) {
    net <- x$net$mean
  } else {
    net <- x$net
  }
  print(
    net,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("\nNet pairwise spillovers:\n")
  if (is.list(x$net_pairwise)) {
    net_pairwise <- x$net_pairwise$mean
  } else {
    net_pairwise <- x$net_pairwise
  }
  print(
    net_pairwise,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname spillover
#' @exportS3Method knitr::knit_print
knit_print.bvharspillover <- function(x, ...) {
  print(x)
}

#' @rdname dynamic_spillover
#' @param x `bvhardynsp` object
#' @param digits digit option to print
#' @param ... not used
#' @importFrom utils head
#' @importFrom dplyr mutate n select
#' @importFrom tidyr pivot_wider
#' @order 2
#' @export
print.bvhardynsp <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Dynamics of spillover effect:\n")
  cat(sprintf("Forecast using %s\n", x$process))
  cat(sprintf("Forecast step: %d\n", x$ahead))
  cat("========================\n")
  is_mcmc <- !is.vector(x$tot)
  cat("Total spillovers:\n")
  if (!is_mcmc) {
    cat(sprintf("# A vector: %d\n", length(x$tot)))
  }
  print(
    head(x$tot),
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("------------------------\n")
  # cat("Directional spillovers:\n")
  # print(
  #   x$directional,
  #   digits = digits,
  #   print.gap = 2L,
  #   quote = FALSE
  # )
  if (is_mcmc) {
    dim_data <- nrow(x$to) / length(x$index)
    to_distn <-
      x$to %>%
      select("series", "mean") %>%
      mutate(id = rep(x$index, each = dim_data)) %>%
      pivot_wider(names_from = "series", values_from = "mean")
    from_distn <-
      x$from %>%
      select("series", "mean") %>%
      mutate(id = rep(x$index, each = dim_data)) %>%
      pivot_wider(names_from = "series", values_from = "mean")
    net_distn <-
      x$net %>%
      select("series", "mean") %>%
      mutate(id = rep(x$index, each = dim_data)) %>%
      pivot_wider(names_from = "series", values_from = "mean")
  } else {
    to_distn <- x$to
    from_distn <- x$from
    net_distn <- x$net
  }
  cat("To spillovers:\n")
  print(
    to_distn,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("------------------------\n")
  cat("From spillovers:\n")
  print(
    from_distn,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
  cat("------------------------\n")
  cat("Net spillovers:\n")
  print(
    net_distn,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname dynamic_spillover
#' @exportS3Method knitr::knit_print
knit_print.bvhardynsp <- function(x, ...) {
  print(x)
}
