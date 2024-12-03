#' h-step ahead Normalized Spillover
#'
#' This function gives connectedness table with h-step ahead normalized spillover index (a.k.a. variance shares).
#'
#' @param object Model object
#' @param n_ahead step to forecast. By default, 10.
#' @param ... Not used
#' @references Diebold, F. X., & Yilmaz, K. (2012). *Better to give than to receive: Predictive directional measurement of volatility spillovers*. International Journal of forecasting, 28(1), 57-66.
#' @importFrom tibble rownames_to_column
#' @importFrom tidyr pivot_longer
#' @order 1
#' @export
spillover <- function(object, n_ahead = 10L, ...) {
  UseMethod("spillover", object)
}

#' @rdname spillover
#' @export
spillover.olsmod <- function(object, n_ahead = 10L, ...) {
  # if (object$process == "VAR") {
  #   mod_type <- "freq_var"
  # } else if (object$process == "VHAR") {
  #   mod_type <- "freq_vhar"
  # } else {
  #   mod_type <- ifelse(grepl(pattern = "^BVAR_", object$process), "var", "vhar")
  # }
  # dim_data <- object$coefficients
  # if (grepl(pattern = "^freq_", mod_type)) {
  #   res <- compute_ols_spillover(object, n_ahead)
  # } else {
  #   res <- compute_mn_spillover(
  #     object,
  #     step = n_ahead,
  #     num_iter = num_iter, num_burn = num_burn, thin = thinning,
  #     seed = sample.int(.Machine$integer.max, size = 1)
  #   )
  # }
  res <- compute_ols_spillover(object, n_ahead)
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

#' @rdname spillover
#' @param num_iter Number to sample MNIW distribution
#' @param num_burn Number of burn-in
#' @param thinning Thinning every thinning-th iteration
#' @export 
spillover.normaliw <- function(object, n_ahead = 10L, num_iter = 5000L, num_burn = floor(num_iter / 2), thinning = 1L, ...) {
  # if (object$process == "VAR") {
  #   mod_type <- "freq_var"
  # } else if (object$process == "VHAR") {
  #   mod_type <- "freq_vhar"
  # } else {
  #   mod_type <- ifelse(grepl(pattern = "^BVAR_", object$process), "var", "vhar")
  # }
  # mod_type <- class(object)[1]
  # dim_data <- object$coefficients
  # if (grepl(pattern = "^freq_", mod_type)) {
  #   res <- compute_ols_spillover(object, n_ahead)
  # } else {
  #   res <- compute_mn_spillover(
  #     object,
  #     step = n_ahead,
  #     num_iter = num_iter, num_burn = num_burn, thin = thinning,
  #     seed = sample.int(.Machine$integer.max, size = 1)
  #   )
  # }
  res <- compute_mn_spillover(
    object,
    step = n_ahead,
    num_iter = num_iter, num_burn = num_burn, thin = thinning,
    seed = sample.int(.Machine$integer.max, size = 1)
  )
  # res <- switch(mod_type,
  #   "bvarmn" = {
  #     compute_bvarmn_spillover(
  #       object$p,
  #       step = n_ahead,
  #       alpha_record = as_draws_matrix(subset_draws(object$param, variable = "alpha")),
  #       sig_record = as_draws_matrix(subset_draws(object$param, variable = "sigma"))
  #     )
  #   },
  #   "bvharmn" = {
  #     compute_bvharmn_spillover(
  #       object$month,
  #       step = n_ahead, har_trans = object$HARtrans,
  #       phi_record = as_draws_matrix(subset_draws(object$param, variable = "phi")),
  #       sig_record = as_draws_matrix(subset_draws(object$param, variable = "sigma"))
  #     )
  #   }
  # )
  # # Preprocess?
  # # 
  # # 
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

#' @rdname spillover
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param sparse `r lifecycle::badge("experimental")` Apply restriction. By default, `FALSE`.
#' @importFrom posterior subset_draws as_draws_matrix
#' @importFrom dplyr left_join
#' @export
spillover.bvarldlt <- function(object, n_ahead = 10L, level = .05, sparse = FALSE, ...) {
  alpha_record <- as_draws_matrix(subset_draws(object$param, variable = "alpha"))
  a_record <- as_draws_matrix(subset_draws(object$param, variable = "a"))
  if (sparse) {
    alpha_record <- as_draws_matrix(subset_draws(object$param, variable = "alpha_sparse"))
    a_record <- as_draws_matrix(subset_draws(object$param, variable = "a_sparse"))
  }
  sp_res <- compute_varldlt_spillover(
    object$p,
    step = n_ahead,
    alpha_record = alpha_record,
    d_record = as_draws_matrix(subset_draws(object$param, variable = "d")),
    a_record = a_record
  )
  dim_data <- object$m
  num_draw <- nrow(alpha_record)
  var_names <- colnames(object$coefficients)
  connect_distn <- process_forecast_draws(
    sp_res$connect,
    n_ahead = dim_data,
    dim_data = dim_data,
    num_draw = num_draw,
    var_names = var_names,
    level = level,
    med = FALSE
  )
  net_pairwise_distn <- process_forecast_draws(
    sp_res$net_pairwise,
    n_ahead = dim_data,
    dim_data = dim_data,
    num_draw = num_draw,
    var_names = var_names,
    level = level,
    med = FALSE
  )
  tot_distn <- process_vector_draws(sp_res$tot, dim_data = 1, level = level, med = FALSE)
  to_distn <- process_vector_draws(sp_res$to, dim_data = dim_data, level = level, med = FALSE)
  from_distn <- process_vector_draws(sp_res$from, dim_data = dim_data, level = level, med = FALSE)
  net_distn <- process_vector_draws(sp_res$net, dim_data = dim_data, level = level, med = FALSE)
  df_long <-
    join_long_spillover(connect_distn, prefix = "spillover") %>%
    left_join(join_long_spillover(net_pairwise_distn, prefix = "net"), by = c("series", "shock"))
  res <- list(
    connect = connect_distn,
    net_pairwise = net_pairwise_distn,
    tot = tot_distn,
    to = to_distn,
    from = from_distn,
    net = net_distn,
    df_long = df_long,
    ahead = n_ahead,
    process = object$process
  )
  class(res) <- "bvharspillover"
  res
}

#' @rdname spillover
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param sparse `r lifecycle::badge("experimental")` Apply restriction. By default, `FALSE`.
#' @importFrom posterior subset_draws as_draws_matrix
#' @importFrom dplyr left_join
#' @export
spillover.bvharldlt <- function(object, n_ahead = 10L, level = .05, sparse = FALSE, ...) {
  phi_record <- as_draws_matrix(subset_draws(object$param, variable = "phi"))
  a_record <- as_draws_matrix(subset_draws(object$param, variable = "a"))
  if (sparse) {
    phi_record <- as_draws_matrix(subset_draws(object$param, variable = "phi_sparse"))
    a_record <- as_draws_matrix(subset_draws(object$param, variable = "a_sparse"))
  }
  sp_res <- compute_vharldlt_spillover(
    object$week, object$month,
    step = n_ahead,
    phi_record = phi_record,
    d_record = as_draws_matrix(subset_draws(object$param, variable = "d")),
    a_record = a_record
  )
  dim_data <- object$m
  num_draw <- nrow(phi_record)
  var_names <- colnames(object$coefficients)
  connect_distn <- process_forecast_draws(
    sp_res$connect,
    n_ahead = dim_data,
    dim_data = dim_data,
    num_draw = num_draw,
    var_names = var_names,
    level = level,
    med = FALSE
  )
  net_pairwise_distn <- process_forecast_draws(
    sp_res$net_pairwise,
    n_ahead = dim_data,
    dim_data = dim_data,
    num_draw = num_draw,
    var_names = var_names,
    level = level,
    med = FALSE
  )
  tot_distn <- process_vector_draws(sp_res$tot, dim_data = 1, level = level, med = FALSE)
  to_distn <- process_vector_draws(sp_res$to, dim_data = dim_data, level = level, med = FALSE)
  from_distn <- process_vector_draws(sp_res$from, dim_data = dim_data, level = level, med = FALSE)
  net_distn <- process_vector_draws(sp_res$net, dim_data = dim_data, level = level, med = FALSE)
  df_long <-
    join_long_spillover(connect_distn, prefix = "spillover") %>%
    left_join(join_long_spillover(net_pairwise_distn, prefix = "net"), by = c("series", "shock"))
  res <- list(
    connect = connect_distn,
    net_pairwise = net_pairwise_distn,
    tot = tot_distn,
    to = to_distn,
    from = from_distn,
    net = net_distn,
    df_long = df_long,
    ahead = n_ahead,
    process = object$process
  )
  # colnames(res$net_pairwise) <- colnames(res$connect)
  # rownames(res$net_pairwise) <- rownames(res$connect)
  # res$connect <- rbind(res$connect, "to_spillovers" = res$to)
  # res$connect <- cbind(res$connect, "from_spillovers" = c(res$from, res$tot))
  # res$ahead <- n_ahead
  # res$process <- object$process
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
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @export
dynamic_spillover.olsmod <- function(object, n_ahead = 10L, window, num_thread = 1, ...) {
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
  # if (model_type == "varlse" || model_type == "vharlse") {
  #   method <- switch(object$method,
  #     "nor" = 1,
  #     "chol" = 2,
  #     "qr" = 3
  #   )
  # }
  method <- switch(object$method,
    "nor" = 1,
    "chol" = 2,
    "qr" = 3
  )
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
    }
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
#' @param window Window size
#' @param num_iter Number to sample MNIW distribution
#' @param num_burn Number of burn-in
#' @param thinning Thinning every thinning-th iteration
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @export
dynamic_spillover.normaliw <- function(object, n_ahead = 10L, window,
                                       num_iter = 1000L, num_burn = floor(num_iter / 2), thinning = 1,
                                       num_thread = 1, ...) {
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
  # num_chains <- object$chain
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  # if (model_type == "varlse" || model_type == "vharlse") {
  #   method <- switch(object$method,
  #     "nor" = 1,
  #     "chol" = 2,
  #     "qr" = 3
  #   )
  # }
  sp_list <- switch(model_type,
    "bvarmn" = {
      dynamic_bvar_spillover(
        y = object$y, window = window, step = n_ahead,
        num_iter = num_iter, num_burn = num_burn, thin = thinning,
        # num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thin = object$thin,
        lag = object$p, bayes_spec = object$spec, include_mean = include_mean,
        seed_chain = sample.int(.Machine$integer.max, size = num_horizon),
        # seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) %>% matrix(ncol = num_chains),
        nthreads = num_thread
      )
    },
    "bvharmn" = {
      dynamic_bvhar_spillover(
        y = object$y, window = window, step = n_ahead,
        num_iter = num_iter, num_burn = num_burn, thin = thinning,
        # num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thin = object$thin,
        week = object$week, month = object$month, bayes_spec = object$spec, include_mean = include_mean,
        seed_chain = sample.int(.Machine$integer.max, size = num_horizon),
        # seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) %>% matrix(ncol = num_chains),
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
    index = window:nrow(object$y),
    ahead = n_ahead,
    process = object$process
  )
  class(res) <- "bvhardynsp"
  res
}

#' @rdname dynamic_spillover
#' @param window Window size
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param sparse `r lifecycle::badge("experimental")` Apply restriction. By default, `FALSE`.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @importFrom dplyr mutate
#' @export
dynamic_spillover.ldltmod <- function(object, n_ahead = 10L, window, level = .05, sparse = FALSE, num_thread = 1, ...) {
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
  # num_chains <- object$chain
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  # prior_nm <- object$spec$prior
  prior_nm <- ifelse(
    object$spec$prior == "MN_VAR" || object$spec$prior == "MN_VHAR" || object$spec$prior == "MN_Hierarchical",
    "Minnesota",
    object$spec$prior
  )
  if (prior_nm == "Minnesota") {
    param_prior <- append(object$spec, list(p = object$p))
    if (object$spec$hierarchical) {
      param_prior$shape <- object$spec$lambda$param[1]
      param_prior$rate <- object$spec$lambda$param[2]
      param_prior$grid_size <- object$spec$lambda$grid_size
      prior_nm <- "MN_Hierarchical"
    }
  } else if (prior_nm == "SSVS") {
    param_prior <- object$spec
  } else if (prior_nm == "Horseshoe") {
    param_prior <- list()
  } else if (prior_nm == "NG") {
    param_prior <- object$spec
  } else if (prior_nm == "DL") {
    param_prior <- object$spec
  }
  prior_type <- switch(prior_nm,
    "Minnesota" = 1,
    "SSVS" = 2,
    "Horseshoe" = 3,
    "MN_Hierarchical" = 4,
    "NG" = 5,
    "DL" = 6,
    "GDP" = 7
  )
  grp_id <- unique(c(object$group))
  if (length(grp_id) > 1) {
    own_id <- 2
    cross_id <- seq_len(object$p + 1)[-2]
  } else {
    own_id <- 1
    cross_id <- 2
  }
  num_chains <- object$chain
  # chunk_size <- num_horizon * num_chains %/% num_thread # default setting of OpenMP schedule(static)
  sp_list <- switch(model_type,
    "bvarldlt" = {
      dynamic_bvarldlt_spillover(
        y = object$y, window = window, step = n_ahead,
        num_chains = num_chains,
        num_iter = object$iter, num_burn = object$burn, thin = object$thin, sparse = sparse,
        lag = object$p,
        param_reg = object$sv[c("shape", "scale")],
        param_prior = param_prior,
        param_intercept = object$intercept[c("mean_non", "sd_non")],
        # param_init = object$init[[1]], # should add multiple chain later
        param_init = object$init,
        prior_type = prior_type,
        ggl = object$ggl,
        grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = object$group,
        include_mean = include_mean,
        # seed_chain = sample.int(.Machine$integer.max, size = num_horizon),
        seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) %>% matrix(ncol = num_chains),
        nthreads = num_thread
      )
    },
    "bvharldlt" = {
      dynamic_bvharldlt_spillover(
        y = object$y, window = window, step = n_ahead,
        num_chains = num_chains,
        num_iter = object$iter, num_burn = object$burn, thin = object$thin, sparse = sparse,
        week = object$p, month = object$month,
        param_reg = object$sv[c("shape", "scale")],
        param_prior = param_prior,
        param_intercept = object$intercept[c("mean_non", "sd_non")],
        param_init = object$init,
        prior_type = prior_type,
        ggl = object$ggl,
        grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = object$group,
        include_mean = include_mean,
        seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) %>% matrix(ncol = num_chains),
        nthreads = num_thread
      )
    },
    stop("Not supported model.")
  )
  dim_data <- object$m
  var_names <- colnames(object$coefficients)
  to_distn <- process_dynamic_spdraws(sp_list$to, dim_data = dim_data, level = level, med = FALSE, var_names = var_names)
  from_distn <- process_dynamic_spdraws(sp_list$from, dim_data = dim_data, level = level, med = FALSE, var_names = var_names)
  net_distn <- process_dynamic_spdraws(sp_list$net, dim_data = dim_data, level = level, med = FALSE, var_names = var_names)
  tot_distn <-
    lapply(
      sp_list$tot,
      function(x) {
        process_vector_draws(unlist(x), dim_data = 1, level = level, med = FALSE) %>%
          do.call(cbind, .)
      }
    ) %>% 
    do.call(rbind, .)
  # sp_list <- lapply(sp_list, function(x) {
  #   if (is.matrix(x)) {
  #     return(apply(x, 1, mean))
  #   }
  #   Reduce("+", x) / length(x)
  # })
  # colnames(sp_list$to) <- paste(colnames(object$y), "to", sep = "_")
  # colnames(sp_list$from) <- paste(colnames(object$y), "from", sep = "_")
  # colnames(sp_list$to) <- colnames(object$y)
  # colnames(sp_list$from) <- colnames(object$y)
  # colnames(sp_list$net) <- colnames(object$y)
  res <- list(
    # tot = sp_list$tot,
    tot = tot_distn,
    # directional = as_tibble(cbind(sp_list$to, sp_list$from)),
    # to = as_tibble(sp_list$to),
    # from = as_tibble(sp_list$from),
    # net = as_tibble(sp_list$net),
    to = to_distn,
    from = from_distn,
    net = net_distn,
    index = window:nrow(object$y),
    ahead = n_ahead,
    process = object$process
  )
  class(res) <- "bvhardynsp"
  res
}

#' @rdname dynamic_spillover
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param sparse `r lifecycle::badge("experimental")` Apply restriction. By default, `FALSE`.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @importFrom posterior subset_draws as_draws_matrix
#' @export
dynamic_spillover.svmod <- function(object, n_ahead = 10L, level = .05, sparse = FALSE, num_thread = 1, ...) {
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
      alpha_record <- as_draws_matrix(subset_draws(object$param, variable = "alpha"))
      a_record <- as_draws_matrix(subset_draws(object$param, variable = "a"))
      if (sparse) {
        alpha_record <- as_draws_matrix(subset_draws(object$param, variable = "alpha_sparse"))
        a_record <- as_draws_matrix(subset_draws(object$param, variable = "a_sparse"))
      }
      dynamic_bvarsv_spillover(
        lag = object$p, step = n_ahead, num_design = num_design,
        alpha_record = alpha_record,
        h_record = as_draws_matrix(subset_draws(object$param, variable = "h")),
        a_record = a_record,
        nthreads = num_thread
      )
    },
    "bvharsv" = {
      phi_record <- as_draws_matrix(subset_draws(object$param, variable = "phi"))
      a_record <- as_draws_matrix(subset_draws(object$param, variable = "a"))
      if (sparse) {
        phi_record <- as_draws_matrix(subset_draws(object$param, variable = "phi_sparse"))
        a_record <- as_draws_matrix(subset_draws(object$param, variable = "a_sparse"))
      }
      dynamic_bvharsv_spillover(
        week = object$week, month = object$month, step = n_ahead, num_design = num_design,
        phi_record = phi_record,
        h_record = as_draws_matrix(subset_draws(object$param, variable = "h")),
        a_record = a_record,
        nthreads = num_thread
      )
    },
    stop("Not supported model.")
  )
  dim_data <- object$m
  var_names <- colnames(object$coefficients)
  to_distn <- process_dynamic_spdraws(sp_list$to, dim_data = dim_data, level = level, med = FALSE, var_names = var_names)
  from_distn <- process_dynamic_spdraws(sp_list$from, dim_data = dim_data, level = level, med = FALSE, var_names = var_names)
  net_distn <- process_dynamic_spdraws(sp_list$net, dim_data = dim_data, level = level, med = FALSE, var_names = var_names)
  tot_distn <-
    lapply(
      sp_list$tot,
      function(x) {
        process_vector_draws(unlist(x), dim_data = 1, level = level, med = FALSE) %>%
          do.call(cbind, .)
      }
    ) %>%
    do.call(rbind, .)
  # colnames(sp_list$to) <- paste(colnames(object$y), "to", sep = "_")
  # colnames(sp_list$from) <- paste(colnames(object$y), "from", sep = "_")
  # colnames(sp_list$to) <- colnames(object$y)
  # colnames(sp_list$from) <- colnames(object$y)
  # colnames(sp_list$net) <- colnames(object$y)
  res <- list(
    # tot = sp_list$tot,
    tot = tot_distn,
    # directional = as_tibble(cbind(sp_list$to, sp_list$from)),
    # to = as_tibble(sp_list$to),
    # from = as_tibble(sp_list$from),
    # net = as_tibble(sp_list$net),
    to = to_distn,
    from = from_distn,
    net = net_distn,
    index = seq_len(nrow(object$y))[-seq_len(nrow(object$y) - nrow(object$y0))],
    ahead = n_ahead,
    process = object$process
  )
  class(res) <- "bvhardynsp"
  res
}
