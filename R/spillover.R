#' h-step ahead Normalized Spillover
#'
#' This function gives connectedness table with h-step ahead normalized spillover index (a.k.a. variance shares).
#'
#' @param object Model object
#' @param n_ahead step to forecast. By default, 10.
#' @param num_iter Number to sample MNIW distribution
#' @param num_burn Number of burn-in
#' @param thinning Thinning every thinning-th iteration
#' @param ... Not used
#' @references Diebold, F. X., & Yilmaz, K. (2012). *Better to give than to receive: Predictive directional measurement of volatility spillovers*. International Journal of forecasting, 28(1), 57-66.
#' @importFrom tibble rownames_to_column
#' @importFrom tidyr pivot_longer
#' @order 1
#' @export
spillover <- function(object, n_ahead = 10L, num_iter = 10000L, num_burn = floor(num_iter / 2), thinning = 1L, ...) {
  UseMethod("spillover", object)
}

#' @rdname spillover
#' @export 
spillover.bvharmod <- function(object, n_ahead = 10L, num_iter = 5000L, num_burn = floor(num_iter / 2), thinning = 1L, ...) {
  if (object$process == "VAR") {
    mod_type <- "freq_var"
  } else if (object$process == "VHAR") {
    mod_type <- "freq_vhar"
  } else {
    mod_type <- ifelse(grepl(pattern = "^BVAR_", object$process), "var", "vhar")
  }
  dim_data <- object$coefficients
  if (grepl(pattern = "^freq_", mod_type)) {
    res <- compute_ols_spillover(object, n_ahead)
  } else {
    res <- compute_mn_spillover(
      object, step = n_ahead,
      num_iter = num_iter, num_burn = num_burn, thin = thinning,
      seed = sample.int(.Machine$integer.max, size = 1)
    )
  }
  colnames(res$connect) <- colnames(object$coefficients)
  rownames(res$connect) <- colnames(object$coefficients)
  res$df_long <-
    res$connect %>%
    as.data.frame() %>%
    rownames_to_column(var = "series") %>%
    pivot_longer(-"series", names_to = "shock", values_to = "spillover")
  colnames(res$net_pairwise) <- colnames(res$connect)
  rownames(res$net_pairwise) <- rownames(res$connect)
  res$connect <- rbind(res$connect, "to_spillovers" = res$to)
  res$connect <- cbind(res$connect, "from_spillovers" = c(res$from, res$tot))
  res$ahead <- n_ahead
  res$process <- object$process
  class(res) <- "bvharspillover"
  res
}

#' Dynamic Spillover
#'
#' This function gives connectedness table with h-step ahead normalized spillover index (a.k.a. variance shares).
#'
#' @param object Model object
#' @param n_ahead step to forecast. By default, 10.
#' @param ... Not used
#' @references Diebold, F. X., & Yilmaz, K. (2012). *Better to give than to receive: Predictive directional measurement of volatility spillovers*. International Journal of forecasting, 28(1), 57-66.
#' @importFrom tibble as_tibble
#' @order 1
#' @export
dynamic_spillover <- function(object, n_ahead = 10L, ...) {
  UseMethod("dynamic_spillover", object)
}

#' @rdname dynamic_spillover
#' @param window Window size
#' @param num_iter Number to sample MNIW distribution
#' @param num_burn Number of burn-in
#' @param thinning Thinning every thinning-th iteration
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @export
dynamic_spillover.bvharmod <- function(object, n_ahead = 10L, window, num_iter = 5000L, num_burn = floor(num_iter / 2), thinning = 1L, num_thread = 1, ...) {
  num_horizon <- nrow(object$y) - window + 1
  if (num_horizon < 0) {
    stop(sprintf("Invalid 'window' size: Specify as 'window' < 'nrow(y) + 1' = %d", nrow(object$y) + 1))
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_horizon) {
    warning(sprintf("'num_thread' > number of horizon will use not every thread. Specify as 'num_thread' <= 'nrow(y) - window + 1' = %d.", num_horizon))
  }
  model_type <- class(object)[1]
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  if (model_type == "varlse" || model_type == "vharlse") {
    method <- switch(object$method,
      "nor" = 1,
      "chol" = 2,
      "qr" = 3
    )
  }
  sp_list <- switch(model_type,
    "varlse" = {
      dynamic_var_spillover(
        y = object$y, window = window, step = n_ahead,
        lag = object$p, include_mean = include_mean, method = method,
        nthreads = num_thread
      )
    },
    "vharlse" = {
      dynamic_vhar_spillover(
        y = object$y, window = window, step = n_ahead,
        week = object$week, month = object$month, include_mean = include_mean, method = method,
        nthreads = num_thread
      )
    },
    "bvarmn" = {
      dynamic_bvar_spillover(
        y = object$y, window = window, step = n_ahead,
        num_iter = num_iter, num_burn = num_burn, thin = thinning,
        lag = object$p, bayes_spec = object$spec, include_mean = include_mean,
        seed_chain = sample.int(.Machine$integer.max, size = num_horizon), nthreads = num_thread
      )
    },
    "bvharmn" = {
      dynamic_bvhar_spillover(
        y = object$y, window = window, step = n_ahead,
        num_iter = num_iter, num_burn = num_burn, thin = thinning,
        week = object$week, month = object$month, bayes_spec = object$spec, include_mean = include_mean,
        seed_chain = sample.int(.Machine$integer.max, size = num_horizon), nthreads = num_thread
      )
    },
    stop("Not supported model.")
  )
  # colnames(sp_list$to) <- paste(colnames(object$y), "to", sep = "_")
  # colnames(sp_list$from) <- paste(colnames(object$y), "from", sep = "_")
  colnames(sp_list$to) <- colnames(object$y)
  colnames(sp_list$from) <- colnames(object$y)
  colnames(sp_list$net) <- colnames(object$y)
  res <- list(
    tot = sp_list$tot,
    # directional = as_tibble(cbind(sp_list$to, sp_list$from)),
    to = as_tibble(sp_list$to),
    from = as_tibble(sp_list$from),
    net = as_tibble(sp_list$net),
    index = window:nrow(object$y),
    ahead = n_ahead,
    process = object$process
  )
  class(res) <- "bvhardynsp"
  res
}

#' @rdname dynamic_spillover
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @importFrom posterior as_draws_matrix
#' @export
dynamic_spillover.svmod <- function(object, n_ahead = 10L, num_thread = 1, ...) {
  num_design <- nrow(object$y0)
  if (num_design < 0) {
    stop(sprintf("Invalid 'window' size: Specify as 'window' < 'nrow(y) + 1' = %d", nrow(object$y) + 1))
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_design) {
    warning(sprintf("'num_thread' > number of horizon will use not every thread. Specify as 'num_thread' <= 'nrow(y) - p (month)' = %d.", num_design))
  }
  model_type <- class(object)[1]
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  sp_list <- switch(model_type,
    "bvarsv" = {
      dynamic_bvarsv_spillover(
        lag = object$p, step = n_ahead, num_design = num_design,
        alpha_record = as_draws_matrix(object$alpha_record), h_record = as_draws_matrix(object$h_record), a_record = as_draws_matrix(object$a_record),
        nthreads = num_thread
      )
    },
    "bvharsv" = {
      dynamic_bvharsv_spillover(
        week = object$week, month = object$month, step = n_ahead, num_design = num_design,
        phi_record = as_draws_matrix(object$phi_record), h_record = as_draws_matrix(object$h_record), a_record = as_draws_matrix(object$a_record),
        nthreads = num_thread
      )
    },
    stop("Not supported model.")
  )
  # colnames(sp_list$to) <- paste(colnames(object$y), "to", sep = "_")
  # colnames(sp_list$from) <- paste(colnames(object$y), "from", sep = "_")
  colnames(sp_list$to) <- colnames(object$y)
  colnames(sp_list$from) <- colnames(object$y)
  colnames(sp_list$net) <- colnames(object$y)
  res <- list(
    tot = sp_list$tot,
    # directional = as_tibble(cbind(sp_list$to, sp_list$from)),
    to = as_tibble(sp_list$to),
    from = as_tibble(sp_list$from),
    net = as_tibble(sp_list$net),
    index = seq_len(nrow(object$y))[-seq_len(nrow(object$y) - nrow(object$y0))],
    ahead = n_ahead,
    process = object$process
  )
  class(res) <- "bvhardynsp"
  res
}
