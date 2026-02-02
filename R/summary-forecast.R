#' Split a Time Series Dataset into Train-Test Set
#' 
#' Split a given time series dataset into train and test set for evaluation.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param n_ahead step to evaluate
#' @return List of two datasets, train and test.
#' @importFrom stats setNames
#' @export
divide_ts <- function(y, n_ahead) {
  num_ts <- nrow(y)
  fac_train <- rep(1, num_ts - n_ahead)
  fac_test <- rep(2, n_ahead)
  y |> 
    split.data.frame(
      factor(c(fac_train, fac_test))
    ) |> 
    setNames(c("train", "test"))
}

#' Out-of-sample Forecasting based on Rolling Window
#' 
#' This function conducts rolling window forecasting.
#' 
#' @param object Model object
#' @param n_ahead Step to forecast in rolling window scheme
#' @param y_test Test data to be compared. Use [divide_ts()] if you don't have separate evaluation dataset.
#' Use numeric, the length of test period, if `object` is the full sample result.
#' Numeric `y_test` will automatically splits `object$y`.
#' In this case, `use_fit` is forced to become `FALSE`.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param newxreg New values for exogenous variables.
#' Should have the same row numbers as `y_test`.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @param ... Additional arguments
#' @details 
#' Rolling windows forecasting fixes window size.
#' It moves the window ahead and forecast h-ahead in `y_test` set.
#' @return `predbvhar_roll` [class]
#' @references Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTEXTS.
#' @order 1
#' @export
forecast_roll <- function(object, n_ahead, y_test, level = .05, newxreg = NULL, num_thread = 1, ...) {
  UseMethod("forecast_roll", object)
}

#' @rdname forecast_roll
#' @export
forecast_roll.olsmod <- function(object, n_ahead, y_test, level = .05, newxreg = NULL, num_thread = 1, ...) {
  y <- object$y
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0(
      "y",
      seq_len(ncol(y))
    )
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  if (!is.matrix(y_test)) {
    y_test <- as.matrix(y_test)
  }
  model_type <- class(object)[1]
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  num_horizon <- nrow(y_test) - n_ahead + 1
  method <- switch(object$method,
    "nor" = 1,
    "chol" = 2,
    "qr" = 3
  )
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_horizon) {
    warning(sprintf("'num_thread' > number of horizon will use not every thread. Specify as 'num_thread' <= 'nrow(y_test) - n_ahead + 1' = %d.", num_horizon))
  }
  is_exogen <- !is.null(eval.parent(object$call$exogen))
  if (is_exogen) {
    newxreg <- validate_newxreg(newxreg = newxreg, n_ahead = nrow(y_test))
  }
  res_mat <- switch(model_type,
    "varlse" = {
      if (is_exogen) {
        roll_varx(y, object$p, include_mean, n_ahead, y_test, method, num_thread, rbind(object$exogen_data, newxreg), object$s)
      } else {
        roll_var(y, object$p, include_mean, n_ahead, y_test, method, num_thread)
      }
    },
    "vharlse" = {
      if (is_exogen) {
        roll_vharx(y, object$week, object$month, include_mean, n_ahead, y_test, method, num_thread, rbind(object$exogen_data, newxreg), object$s)
      } else {
        roll_vhar(y, object$week, object$month, include_mean, n_ahead, y_test, method, num_thread)
      }
    }
  )
  colnames(res_mat) <- name_var
  res <- list(
    process = object$process,
    forecast = res_mat,
    eval_id = n_ahead:nrow(y_test),
    y = y
  )
  class(res) <- c("predbvhar_roll", "bvharcv")
  res
}

#' @rdname forecast_roll
#' @param use_fit `r lifecycle::badge("experimental")` Use `object` result for the first window. By default, `TRUE`.
#' @export
forecast_roll.normaliw <- function(object, n_ahead, y_test, level = .05, newxreg = NULL, num_thread = 1, use_fit = TRUE, ...) {
  y <- object$y
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0(
      "y",
      seq_len(ncol(y))
    )
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  if (!is.matrix(y_test)) {
    y_test <- as.matrix(y_test)
  }
  model_type <- class(object)[1]
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  # num_chains <- object$chain
  num_horizon <- nrow(y_test) - n_ahead + 1
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  # if (num_thread > num_horizon * num_chains) {
  #   warning(sprintf("'num_thread' > (number of horizon * number of chains) will use not every thread. Specify as 'num_thread' <= '(nrow(y_test) - n_ahead + 1) * object$chain' = %d.", num_horizon * num_chains))
  # }
  if (num_thread > num_horizon) {
    warning(sprintf("'num_thread' > number of horizon will use not every thread. Specify as 'num_thread' <= 'nrow(y_test) - n_ahead + 1' = %d.", num_horizon))
  }
  # if (num_horizon * num_chains %% num_thread != 0) {
  #   warning(sprintf("OpenMP cannot divide the iterations as integer. Use divisor of ('nrow(y_test) - n_ahead + 1') * 'num_thread' <= 'object$chain' = %d", num_horizon * num_chains))
  # }
  # chunk_size <- num_horizon * num_chains %/% num_thread # default setting of OpenMP schedule(static)
  # if (chunk_size == 0) {
  #   chunk_size <- 1
  # }
  # if (num_horizon > num_chains && chunk_size > num_chains) {
  #   chunk_size <- min(
  #     num_chains,
  #     (num_horizon %/% num_thread) * num_chains
  #   )
  #   if (chunk_size == 0) {
  #     chunk_size <- 1
  #   }
  # }

  # if (model_type != "bvarflat") {
  #   if (num_thread > get_maxomp()) {
  #     warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  #   }
  #   if (num_thread > get_maxomp()) {
  #     warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  #   }
  #   if (num_thread > num_horizon) {
  #     warning(sprintf("'num_thread' > number of horizon will use not every thread. Specify as 'num_thread' <= 'nrow(y_test) - n_ahead + 1' = %d.", num_horizon))
  #   }
  # }
  # fit_ls <- list()
  # if (use_fit) {
  #   fit_ls <- lapply(
  #     object$param_names,
  #     function(x) {
  #       subset_draws(object$param, variable = x) |>
  #         as_draws_matrix() |>
  #         split.data.frame(gl(num_chains, nrow(object$param) / num_chains))
  #     }
  #   ) |>
  #     setNames(paste(object$param_names, "record", sep = "_"))
  # }
  res_mat <- switch(model_type,
    "bvarmn" = {
      roll_bvar(y, object$p, object$spec, include_mean, n_ahead, y_test, sample.int(.Machine$integer.max, size = num_horizon), num_thread)
      # roll_bvar(
      #   y, object$p, num_chains, object$iter, object$burn, object$thin,
      #   fit_ls,
      #   object$spec, include_mean, n_ahead, y_test,
      #   sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
      #   num_thread, chunk_size
      # )
    },
    "bvarflat" = {
      roll_bvarflat(y, object$p, object$spec$U, include_mean, n_ahead, y_test, sample.int(.Machine$integer.max, size = num_horizon), num_thread)
      # roll_bvarflat(
      #   y, object$p, num_chains, object$iter, object$burn, object$thin,
      #   fit_ls,
      #   object$spec$U, include_mean, n_ahead, y_test,
      #   sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
      #   num_thread, chunk_size
      # )
    },
    "bvharmn" = {
      roll_bvhar(y, object$week, object$month, object$spec, include_mean, n_ahead, y_test, sample.int(.Machine$integer.max, size = num_horizon), num_thread)
      # roll_bvhar(
      #   y, object$week, object$month, num_chains, object$iter, object$burn, object$thin,
      #   fit_ls,
      #   object$spec, include_mean, n_ahead, y_test,
      #   sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
      #   num_thread, chunk_size
      # )
    }
  )
  # num_draw <- nrow(object$a_record) # concatenate multiple chains
  # res_mat <-
  #   res_mat |>
  #   lapply(function(res) {
  #     unlist(res) |>
  #       array(dim = c(1, object$m, num_draw)) |>
  #       apply(c(1, 2), mean)
  #   }) |>
  #   do.call(rbind, .)
  colnames(res_mat) <- name_var
  res <- list(
    process = object$process,
    forecast = res_mat,
    eval_id = n_ahead:nrow(y_test),
    y = y
  )
  class(res) <- c("predbvhar_roll", "bvharcv")
  res
}

#' @rdname forecast_roll
#' @param stable `r lifecycle::badge("experimental")` Filter only stable coefficient draws in MCMC records.
#' @param sparse `r lifecycle::badge("experimental")` Apply restriction. By default, `FALSE`.
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param lpl `r lifecycle::badge("experimental")` Compute log-predictive likelihood (LPL). By default, `FALSE`.
#' @param mcmc `r lifecycle::badge("experimental")` If `TRUE`, run new MCMC in new windows. By default, `TRUE`.
#' @param use_fit `r lifecycle::badge("experimental")` Use `object` result for the first window. By default, `TRUE`.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @export
forecast_roll.ldltmod <- function(object, n_ahead, y_test,
                                  level = .05,
                                  newxreg = NULL,
                                  num_thread = 1,
                                  stable = FALSE,
                                  sparse = FALSE,
                                  med = FALSE,
                                  lpl = FALSE,
                                  mcmc = TRUE,
                                  use_fit = TRUE,
                                  verbose = FALSE, ...) {
  y <- object$y
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0(
      "y",
      seq_len(ncol(y))
    )
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  size_factor <- 0
  factor_lag <- 0
  if (!is.null(object$spec_factor)) {
    size_factor <- object$factor_size
    factor_lag <- object$factor_lag
  }
  is_full <- is.numeric(y_test) && length(y_test) == 1
  if (is_full) {
    num_test <- y_test
    y_split <- divide_ts(y, num_test)
    y <- y_split$train
    y_test <- y_split$test
    use_fit <- FALSE
  }
  if (!is.matrix(y_test)) {
    y_test <- as.matrix(y_test)
  }
  model_type <- class(object)[1]
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  num_chains <- object$chain
  num_horizon <- nrow(y_test) - n_ahead + 1
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_horizon * num_chains) {
    warning(sprintf("'num_thread' > (number of horizon * number of chains) will use not every thread. Specify as 'num_thread' <= '(nrow(y_test) - n_ahead + 1) * object$chain' = %d.", num_horizon * num_chains))
  }
  ci_lev <- 0
  if (is.numeric(sparse)) {
    ci_lev <- sparse
    sparse <- FALSE
  }
  fit_ls <- list()
  if (use_fit) {
    fit_ls <- get_records(object, TRUE)
  }
  param_prior <- get_coefspec(object)
  prior_type <- enumerate_prior(object$spec_coef$prior)
  contem_prior <- get_contemspec(object)
  contem_prior_type <- enumerate_prior(object$spec_contem$prior)
  is_exogen <- !is.null(eval.parent(object$call$exogen))
  if (is_exogen) {
    if (is_full) {
      exogen_split <- divide_ts(object$exogen_data, nrow(y_test))
      object$exogen_data <- exogen_split$train
      newxreg <- exogen_split$test
    }
    newxreg <- validate_newxreg(newxreg = newxreg, n_ahead = nrow(y_test))
    exogen_prior <- get_exogenspec(object)
    exogen_prior_type <- enumerate_prior(object$spec_exogen$prior)
  }
  factor_prior <- list()
  factor_prior_type <- 0
  factor_init <- list()
  if (!is.null(object$spec_factor)) {
    factor_prior <- get_factorspec(object)
    factor_prior_type <- enumerate_prior(object$spec_factor$prior)
    factor_init <- object$init_factor
  }
  res_mat <- switch(model_type,
    "bvarldlt" = {
      grp_mat <- object$group
      grp_id <- unique(c(grp_mat))
      own_id <- 2
      cross_id <- seq_len(object$p + 1)[-2]
      if (is_exogen) {
        roll_bvarxldlt(
          y = y, lag = object$p, num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_reg = object$sv[c("shape", "scale")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test, get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread,
          # exogen = rbind(tail(object$exogen_data, object$s), newxreg), exogen_lag = object$s,
          exogen = rbind(object$exogen_data, newxreg), exogen_lag = object$s,
          exogen_prior = exogen_prior, exogen_init = object$init_exogen, exogen_prior_type = exogen_prior_type
        )
      } else {
        roll_bvarldlt(
          y = y, lag = object$p, num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_reg = object$sv[c("shape", "scale")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test, get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread
        )
      }
    },
    "bvharldlt" = {
      grp_mat <- object$group
      grp_id <- unique(c(grp_mat))
      if (length(grp_id) == 6) {
        own_id <- c(2, 4, 6)
        cross_id <- c(1, 3, 5)
      } else {
        own_id <- 2
        cross_id <- c(1, 3, 4)
      }
      if (is_exogen) {
        roll_bvharxldlt(
          y = y, week = object$week, month = object$month,
          num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_reg = object$sv[c("shape", "scale")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test,
          get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread,
          exogen = rbind(object$exogen_data, newxreg), exogen_lag = object$s,
          exogen_prior = exogen_prior, exogen_init = object$init_exogen, exogen_prior_type = exogen_prior_type
        )
      } else {
        roll_bvharldlt(
          y = y, week = object$week, month = object$month,
          num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_reg = object$sv[c("shape", "scale")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test,
          get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread
        )
      }
    }
  )
  num_draw <- nrow(object$param) # concatenate multiple chains
  # num_draw <- nrow(object$param) / num_chains
  if (lpl) {
    # lpl_val <- res_mat$lpl
    if (med) {
      lpl_val <- apply(res_mat$lpl, 1, median)
    } else {
      lpl_val <- rowMeans(res_mat$lpl)
    }
    res_mat$lpl <- NULL
  }
  # If path, res_mat$forecast = list(windows) -> list(chains) -> matrix(n_ahead, dim * draws)
  # If last point, res_mat$forecast = list(windows) -> list(chains) -> vector(dim * draws)
  y_distn <- process_forecast_draws(
    draws = res_mat$forecast,
    n_ahead = n_ahead, # 1 when isPath = false while n_ahead when isPath = true
    dim_data = object$m,
    num_draw = num_draw,
    var_names = name_var,
    roll = TRUE,
    med = med
  )
  # Temporarily same results when isPath = false
  y_distn <- lapply(
    y_distn,
    function(x) x[,,dim(x)[3]]
  )
  res <- list(
    process = object$process,
    draws = res_mat$forecast,
    forecast = y_distn$mean,
    se = y_distn$sd,
    lower = y_distn$lower,
    upper = y_distn$upper,
    lower_joint = y_distn$lower,
    upper_joint = y_distn$upper,
    med = med,
    eval_id = n_ahead:nrow(y_test),
    y = y
  )
  if (lpl) {
    res$lpl <- lpl_val
  }
  class(res) <- c("predbvhar_roll", "bvharcv")
  res
}

#' @rdname forecast_roll
#' @param use_sv Use SV term
#' @param stable `r lifecycle::badge("experimental")` Filter only stable coefficient draws in MCMC records.
#' @param sparse `r lifecycle::badge("experimental")` Apply restriction. By default, `FALSE`.
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param lpl `r lifecycle::badge("experimental")` Compute log-predictive likelihood (LPL). By default, `FALSE`.
#' @param mcmc `r lifecycle::badge("experimental")` If `TRUE`, run new MCMC in new windows. By default, `TRUE`.
#' @param use_fit `r lifecycle::badge("experimental")` Use `object` result for the first window. By default, `TRUE`.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @export
forecast_roll.svmod <- function(object, n_ahead, y_test,
                                level = .05,
                                newxreg = NULL,
                                num_thread = 1,
                                use_sv = TRUE,
                                stable = FALSE,
                                sparse = FALSE,
                                med = FALSE,
                                lpl = FALSE,
                                mcmc = TRUE,
                                use_fit = TRUE,
                                verbose = FALSE, ...) {
  y <- object$y
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0(
      "y",
      seq_len(ncol(y))
    )
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  size_factor <- 0
  factor_lag <- 0
  if (!is.null(object$spec_factor)) {
    size_factor <- object$factor_size
    factor_lag <- object$factor_lag
  }
  is_full <- is.numeric(y_test) && length(y_test) == 1
  if (is_full) {
    num_test <- y_test
    y_split <- divide_ts(y, num_test)
    y <- y_split$train
    y_test <- y_split$test
    use_fit <- FALSE
    object$init_coef <- lapply(
      object$init_coef,
      function(init) {
        init$lvol <- head(init$lvol, nrow(object$design) - num_test)
        init
      }
    )
  }
  if (!is.matrix(y_test)) {
    y_test <- as.matrix(y_test)
  }
  model_type <- class(object)[1]
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  num_chains <- object$chain
  num_horizon <- nrow(y_test) - n_ahead + 1
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_horizon * num_chains) {
    warning(sprintf("'num_thread' > (number of horizon * number of chains) will use not every thread. Specify as 'num_thread' <= '(nrow(y_test) - n_ahead + 1) * object$chain' = %d.", num_horizon * num_chains))
  }
  # if (num_thread > num_chains && num_chains != 1) {
  #   warning(sprintf("'num_thread' > MCMC chain will use not every thread. Specify as 'num_thread' <= 'object$chain' = %d.", num_chains))
  # }
  # if (num_horizon * num_chains %/% num_thread == 0) {
  #   warning(sprintf("OpenMP cannot divide the iterations as integer. Use divisor of ('nrow(y_test) - n_ahead + 1') * 'num_thread' <= 'object$chain' = %d", num_horizon * num_chains))
  # }
  ci_lev <- 0
  if (is.numeric(sparse)) {
    ci_lev <- sparse
    sparse <- FALSE
  }
  fit_ls <- list()
  if (use_fit) {
    fit_ls <- get_records(object, TRUE)
  }
  param_prior <- get_coefspec(object)
  prior_type <- enumerate_prior(object$spec_coef$prior)
  contem_prior <- get_contemspec(object)
  contem_prior_type <- enumerate_prior(object$spec_contem$prior)
  is_exogen <- !is.null(eval.parent(object$call$exogen))
  if (is_exogen) {
    if (is_full) {
      exogen_split <- divide_ts(object$exogen_data, nrow(y_test))
      object$exogen_data <- exogen_split$train
      newxreg <- exogen_split$test
    }
    newxreg <- validate_newxreg(newxreg = newxreg, n_ahead = nrow(y_test))
    exogen_prior <- get_exogenspec(object)
    exogen_prior_type <- enumerate_prior(object$spec_exogen$prior)
  }
  factor_prior <- list()
  factor_prior_type <- 0
  factor_init <- list()
  if (!is.null(object$spec_factor)) {
    factor_prior <- get_factorspec(object)
    factor_prior_type <- enumerate_prior(object$spec_factor$prior)
    factor_init <- object$init_factor
  }
  res_mat <- switch(model_type,
    "bvarsv" = {
      grp_mat <- object$group
      grp_id <- unique(c(grp_mat))
      own_id <- 2
      cross_id <- seq_len(object$p + 1)[-2]
      if (is_exogen) {
        roll_bvarxsv(
          y = y, lag = object$p, num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sv = use_sv, sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_sv = object$sv[c("shape", "scale", "initial_mean", "initial_prec")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test, get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread,
          # exogen = rbind(tail(object$exogen_data, object$s), newxreg), exogen_lag = object$s,
          exogen = rbind(object$exogen_data, newxreg), exogen_lag = object$s,
          exogen_prior = exogen_prior, exogen_init = object$init_exogen, exogen_prior_type = exogen_prior_type
        )
      } else {
        roll_bvarsv(
          y = y, lag = object$p, num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sv = use_sv, sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_sv = object$sv[c("shape", "scale", "initial_mean", "initial_prec")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test, get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread
        )
      }
    },
    "bvharsv" = {
      grp_mat <- object$group
      grp_id <- unique(c(grp_mat))
      if (length(grp_id) == 6) {
        own_id <- c(2, 4, 6)
        cross_id <- c(1, 3, 5)
      } else {
        own_id <- 2
        cross_id <- c(1, 3, 4)
      }
      if (is_exogen) {
        roll_bvharxsv(
          y = y, week = object$week, month = object$month, num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sv = use_sv, sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_sv = object$sv[c("shape", "scale", "initial_mean", "initial_prec")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test, get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread,
          exogen = rbind(object$exogen_data, newxreg), exogen_lag = object$s,
          exogen_prior = exogen_prior, exogen_init = object$init_exogen, exogen_prior_type = exogen_prior_type
        )
      } else {
        roll_bvharsv(
          y = y, week = object$week, month = object$month, num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sv = use_sv, sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_sv = object$sv[c("shape", "scale", "initial_mean", "initial_prec")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test, get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread
        )
      }
    }
  )
  num_draw <- nrow(object$param) # concatenate multiple chains
  # num_draw <- nrow(object$param) / num_chains
  if (lpl) {
    # lpl_val <- res_mat$lpl
    if (med) {
      lpl_val <- apply(res_mat$lpl, 1, median)
    } else {
      lpl_val <- rowMeans(res_mat$lpl)
    }
    res_mat$lpl <- NULL
  }
  y_distn <- process_forecast_draws(
    draws = res_mat$forecast,
    n_ahead = n_ahead,
    dim_data = object$m,
    num_draw = num_draw,
    var_names = name_var,
    roll = TRUE,
    med = med
  )
  res <- list(
    process = object$process,
    draws = res_mat$forecast,
    forecast = y_distn$mean,
    se = y_distn$sd,
    lower = y_distn$lower,
    upper = y_distn$upper,
    lower_joint = y_distn$lower,
    upper_joint = y_distn$upper,
    med = med,
    eval_id = n_ahead:nrow(y_test),
    y = y
  )
  if (lpl) {
    res$lpl <- lpl_val
  }
  class(res) <- c("predbvhar_roll", "bvharcv")
  res
}

#' Out-of-sample Forecasting based on Expanding Window
#'
#' This function conducts expanding window forecasting.
#'
#' @param object Model object
#' @param n_ahead Step to forecast in rolling window scheme
#' @param y_test Test data to be compared. Use [divide_ts()] if you don't have separate evaluation dataset.
#' Use numeric, the length of test period, if `object` is the full sample result.
#' Numeric `y_test` will automatically splits `object$y`.
#' In this case, `use_fit` is forced to become `FALSE`.
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param newxreg New values for exogenous variables.
#' Should have the same row numbers as `y_test`.
#' @param num_thread `r lifecycle::badge("experimental")` Number of threads
#' @param ... Additional arguments.
#' @details
#' Expanding windows forecasting fixes the starting period.
#' It moves the window ahead and forecast h-ahead in `y_test` set.
#' @return `predbvhar_expand` [class]
#' @references Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTEXTS. [https://otexts.com/fpp3/](https://otexts.com/fpp3/)
#' @order 1
#' @export
forecast_expand <- function(object, n_ahead, y_test, level = .05, newxreg = NULL, num_thread = 1, ...) {
  UseMethod("forecast_expand", object)
}

#' @rdname forecast_expand
#' @export
forecast_expand.olsmod <- function(object, n_ahead, y_test, level = .05, newxreg = NULL, num_thread = 1, ...) {
  y <- object$y
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0(
      "y",
      seq_len(ncol(y))
    )
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  if (!is.matrix(y_test)) {
    y_test <- as.matrix(y_test)
  }
  model_type <- class(object)[1]
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  method <- switch(object$method,
    "nor" = 1,
    "chol" = 2,
    "qr" = 3
  )
  num_horizon <- nrow(y_test) - n_ahead + 1
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_horizon) {
    warning(sprintf("'num_thread' > number of horizon will use not every thread. Specify as 'num_thread' <= 'nrow(y_test) - n_ahead + 1' = %d.", num_horizon))
  }
  is_exogen <- !is.null(eval.parent(object$call$exogen))
  if (is_exogen) {
    newxreg <- validate_newxreg(newxreg = newxreg, n_ahead = nrow(y_test))
  }
  res_mat <- switch(model_type,
    "varlse" = {
      if (is_exogen) {
        expand_varx(y, object$p, include_mean, n_ahead, y_test, method, num_thread, rbind(object$exogen_data, newxreg), object$s)
      } else {
        expand_var(y, object$p, include_mean, n_ahead, y_test, method, num_thread)
      }
    },
    "vharlse" = {
      if (is_exogen) {
        expand_vharx(y, object$week, object$month, include_mean, n_ahead, y_test, method, num_thread, rbind(object$exogen_data, newxreg), object$s)
      } else {
        expand_vhar(y, object$week, object$month, include_mean, n_ahead, y_test, method, num_thread)
      }
    }
  )
  colnames(res_mat) <- name_var
  res <- list(
    process = object$process,
    forecast = res_mat,
    eval_id = n_ahead:nrow(y_test),
    y = y
  )
  class(res) <- c("predbvhar_expand", "bvharcv")
  res
}

#' @rdname forecast_expand
#' @param use_fit `r lifecycle::badge("experimental")` Use `object` result for the first window. By default, `TRUE`.
#' @export
forecast_expand.normaliw <- function(object, n_ahead, y_test, level = .05, newxreg = NULL, num_thread = 1, use_fit = TRUE, ...) {
  y <- object$y
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0(
      "y",
      seq_len(ncol(y))
    )
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  if (!is.matrix(y_test)) {
    y_test <- as.matrix(y_test)
  }
  model_type <- class(object)[1]
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  # num_chains <- object$chain
  num_horizon <- nrow(y_test) - n_ahead + 1
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  # if (num_thread > num_horizon * num_chains) {
  #   warning(sprintf("'num_thread' > (number of horizon * number of chains) will use not every thread. Specify as 'num_thread' <= '(nrow(y_test) - n_ahead + 1) * object$chain' = %d.", num_horizon * num_chains))
  # }
  if (num_thread > num_horizon) {
    warning(sprintf("'num_thread' > number of horizon will use not every thread. Specify as 'num_thread' <= 'nrow(y_test) - n_ahead + 1' = %d.", num_horizon))
  }
  res_mat <- switch(
    model_type,
    "bvarmn" = {
      expand_bvar(y, object$p, object$spec, include_mean, n_ahead, y_test, sample.int(.Machine$integer.max, size = num_horizon), num_thread)
      # expand_bvar(
      #   y, object$p, num_chains, object$iter, object$burn, object$thin,
      #   fit_ls,
      #   object$spec, include_mean, n_ahead, y_test,
      #   sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
      #   num_thread, chunk_size
      # )
    },
    "bvarflat" = {
      expand_bvarflat(y, object$p, object$spec$U, include_mean, n_ahead, y_test, sample.int(.Machine$integer.max, size = num_horizon), num_thread)
      # expand_bvarflat(
      #   y, object$p, num_chains, object$iter, object$burn, object$thin,
      #   fit_ls,
      #   object$spec$U, include_mean, n_ahead, y_test,
      #   sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
      #   num_thread, chunk_size
      # )
    },
    "bvharmn" = {
      expand_bvhar(y, object$week, object$month, object$spec, include_mean, n_ahead, y_test, sample.int(.Machine$integer.max, size = num_horizon), num_thread)
      # expand_bvhar(
      #   y, object$week, object$month, num_chains, object$iter, object$burn, object$thin,
      #   fit_ls,
      #   object$spec, include_mean, n_ahead, y_test,
      #   sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
      #   num_thread, chunk_size
      # )
    }
  )
  # num_draw <- nrow(object$a_record) # concatenate multiple chains
  # res_mat <-
  #   res_mat |>
  #   lapply(function(res) {
  #     unlist(res) |>
  #       array(dim = c(1, object$m, num_draw)) |>
  #       apply(c(1, 2), mean)
  #   }) |>
  #   do.call(rbind, .)
  colnames(res_mat) <- name_var
  res <- list(
    process = object$process,
    forecast = res_mat,
    eval_id = n_ahead:nrow(y_test),
    y = y
  )
  class(res) <- c("predbvhar_expand", "bvharcv")
  res
}

#' @rdname forecast_expand
#' @param stable `r lifecycle::badge("experimental")` Filter only stable coefficient draws in MCMC records.
#' @param sparse `r lifecycle::badge("experimental")` Apply restriction. By default, `FALSE`.
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param lpl `r lifecycle::badge("experimental")` Compute log-predictive likelihood (LPL). By default, `FALSE`.
#' @param mcmc `r lifecycle::badge("experimental")` If `TRUE`, run new MCMC in new windows. By default, `TRUE`.
#' @param use_fit `r lifecycle::badge("experimental")` Use `object` result for the first window. By default, `TRUE`.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @export
forecast_expand.ldltmod <- function(object, n_ahead, y_test,
                                    level = .05,
                                    newxreg = NULL,
                                    num_thread = 1,
                                    stable = FALSE,
                                    sparse = FALSE,
                                    med = FALSE,
                                    lpl = FALSE,
                                    mcmc = TRUE,
                                    use_fit = TRUE,
                                    verbose = FALSE, ...) {
  y <- object$y
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0(
      "y",
      seq_len(ncol(y))
    )
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  size_factor <- 0
  factor_lag <- 0
  if (!is.null(object$spec_factor)) {
    size_factor <- object$factor_size
    factor_lag <- object$factor_lag
  }
  is_full <- is.numeric(y_test) && length(y_test) == 1
  if (is_full) {
    num_test <- y_test
    y_split <- divide_ts(y, num_test)
    y <- y_split$train
    y_test <- y_split$test
    use_fit <- FALSE
  }
  if (!is.matrix(y_test)) {
    y_test <- as.matrix(y_test)
  }
  model_type <- class(object)[1]
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  num_chains <- object$chain
  num_horizon <- nrow(y_test) - n_ahead + 1
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_horizon * num_chains) {
    warning(sprintf("'num_thread' > (number of horizon * number of chains) will use not every thread. Specify as 'num_thread' <= '(nrow(y_test) - n_ahead + 1) * object$chain' = %d.", num_horizon * num_chains))
  }
  ci_lev <- 0
  if (is.numeric(sparse)) {
    ci_lev <- sparse
    sparse <- FALSE
  }
  fit_ls <- list()
  if (use_fit) {
    fit_ls <- get_records(object, TRUE)
  }
  param_prior <- get_coefspec(object)
  prior_type <- enumerate_prior(object$spec_coef$prior)
  contem_prior <- get_contemspec(object)
  contem_prior_type <- enumerate_prior(object$spec_contem$prior)
  is_exogen <- !is.null(eval.parent(object$call$exogen))
  if (is_exogen) {
    if (is_full) {
      exogen_split <- divide_ts(object$exogen_data, nrow(y_test))
      object$exogen_data <- exogen_split$train
      newxreg <- exogen_split$test
    }
    newxreg <- validate_newxreg(newxreg = newxreg, n_ahead = nrow(y_test))
    exogen_prior <- get_exogenspec(object)
    exogen_prior_type <- enumerate_prior(object$spec_exogen$prior)
  }
  factor_prior <- list()
  factor_prior_type <- 0
  factor_init <- list()
  if (!is.null(object$spec_factor)) {
    factor_prior <- get_factorspec(object)
    factor_prior_type <- enumerate_prior(object$spec_factor$prior)
    factor_init <- object$init_factor
  }
  res_mat <- switch(model_type,
    "bvarldlt" = {
      grp_mat <- object$group
      grp_id <- unique(c(grp_mat))
      own_id <- 2
      cross_id <- seq_len(object$p + 1)[-2]
      if (is_exogen) {
        expand_bvarxldlt(
          y = y, lag = object$p, num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_reg = object$sv[c("shape", "scale")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test, get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread,
          # exogen = rbind(tail(object$exogen_data, object$s), newxreg), exogen_lag = object$s,
          exogen = rbind(object$exogen_data, newxreg), exogen_lag = object$s,
          exogen_prior = exogen_prior, exogen_init = object$init_exogen, exogen_prior_type = exogen_prior_type
        )
      } else {
        expand_bvarldlt(
          y = y, lag = object$p, num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_reg = object$sv[c("shape", "scale")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test, get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread
        )
      }
    },
    "bvharldlt" = {
      grp_mat <- object$group
      grp_id <- unique(c(grp_mat))
      if (length(grp_id) == 6) {
        own_id <- c(2, 4, 6)
        cross_id <- c(1, 3, 5)
      } else {
        own_id <- 2
        cross_id <- c(1, 3, 4)
      }
      if (is_exogen) {
        expand_bvharxldlt(
          y = y, week = object$week, month = object$month,
          num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_reg = object$sv[c("shape", "scale")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test,
          get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread,
          exogen = rbind(object$exogen_data, newxreg), exogen_lag = object$s,
          exogen_prior = exogen_prior, exogen_init = object$init_exogen, exogen_prior_type = exogen_prior_type
        )
      } else {
        expand_bvharldlt(
          y = y, week = object$week, month = object$month,
          num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_reg = object$sv[c("shape", "scale")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test,
          get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread
        )
      }
    }
  )
  num_draw <- nrow(object$param) # concatenate multiple chains
  # num_draw <- nrow(object$param) / num_chains
  if (lpl) {
    # lpl_val <- res_mat$lpl
    if (med) {
      lpl_val <- apply(res_mat$lpl, 1, median)
    } else {
      lpl_val <- rowMeans(res_mat$lpl)
    }
    res_mat$lpl <- NULL
  }
  y_distn <- process_forecast_draws(
    draws = res_mat$forecast,
    n_ahead = n_ahead,
    dim_data = object$m,
    num_draw = num_draw,
    var_names = name_var,
    roll = TRUE,
    med = med
  )
  res <- list(
    process = object$process,
    draws = res_mat$forecast,
    forecast = y_distn$mean,
    se = y_distn$sd,
    lower = y_distn$lower,
    upper = y_distn$upper,
    lower_joint = y_distn$lower,
    upper_joint = y_distn$upper,
    med = med,
    eval_id = n_ahead:nrow(y_test),
    y = y
  )
  if (lpl) {
    res$lpl <- lpl_val
  }
  class(res) <- c("predbvhar_expand", "bvharcv")
  res
}

#' @rdname forecast_expand
#' @param use_sv Use SV term
#' @param stable `r lifecycle::badge("experimental")` Filter only stable coefficient draws in MCMC records.
#' @param sparse `r lifecycle::badge("experimental")` Apply restriction. By default, `FALSE`.
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @param lpl `r lifecycle::badge("experimental")` Compute log-predictive likelihood (LPL). By default, `FALSE`.
#' @param mcmc `r lifecycle::badge("experimental")` If `TRUE`, run new MCMC in new windows. By default, `TRUE`.
#' @param use_fit `r lifecycle::badge("experimental")` Use `object` result for the first window. By default, `TRUE`.
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @export
forecast_expand.svmod <- function(object, n_ahead, y_test,
                                  level = .05,
                                  newxreg = NULL,
                                  num_thread = 1,
                                  use_sv = TRUE,
                                  stable = FALSE,
                                  sparse = FALSE,
                                  med = FALSE,
                                  lpl = FALSE,
                                  mcmc = TRUE,
                                  use_fit = TRUE,
                                  verbose = FALSE, ...) {
  y <- object$y
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0(
      "y",
      seq_len(ncol(y))
    )
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  size_factor <- 0
  factor_lag <- 0
  if (!is.null(object$spec_factor)) {
    size_factor <- object$factor_size
    factor_lag <- object$factor_lag
  }
  is_full <- is.numeric(y_test) && length(y_test) == 1
  if (is_full) {
    num_test <- y_test
    y_split <- divide_ts(y, num_test)
    y <- y_split$train
    y_test <- y_split$test
    use_fit <- FALSE
    object$init_coef <- lapply(
      object$init_coef,
      function(init) {
        init$lvol <- head(init$lvol, nrow(object$design) - num_test)
        init
      }
    )
  }
  if (!is.matrix(y_test)) {
    y_test <- as.matrix(y_test)
  }
  model_type <- class(object)[1]
  include_mean <- ifelse(object$type == "const", TRUE, FALSE)
  num_chains <- object$chain
  num_horizon <- nrow(y_test) - n_ahead + 1
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > get_maxomp()) {
    warning("'num_thread' is greater than 'omp_get_max_threads()'. Check with bvhar:::get_maxomp(). Check OpenMP support of your machine with bvhar:::check_omp().")
  }
  if (num_thread > num_horizon * num_chains) {
    warning(sprintf("'num_thread' > (number of horizon * number of chains) will use not every thread. Specify as 'num_thread' <= '(nrow(y_test) - n_ahead + 1) * object$chain' = %d.", num_horizon * num_chains))
  }
  ci_lev <- 0
  if (is.numeric(sparse)) {
    ci_lev <- sparse
    sparse <- FALSE
  }
  fit_ls <- list()
  if (use_fit) {
    fit_ls <- get_records(object, TRUE)
  }
  param_prior <- get_coefspec(object)
  prior_type <- enumerate_prior(object$spec_coef$prior)
  contem_prior <- get_contemspec(object)
  contem_prior_type <- enumerate_prior(object$spec_contem$prior)
  is_exogen <- !is.null(eval.parent(object$call$exogen))
  if (is_exogen) {
    if (is_full) {
      exogen_split <- divide_ts(object$exogen_data, nrow(y_test))
      object$exogen_data <- exogen_split$train
      newxreg <- exogen_split$test
    }
    newxreg <- validate_newxreg(newxreg = newxreg, n_ahead = nrow(y_test))
    exogen_prior <- get_exogenspec(object)
    exogen_prior_type <- enumerate_prior(object$spec_exogen$prior)
  }
  factor_prior <- list()
  factor_prior_type <- 0
  factor_init <- list()
  if (!is.null(object$spec_factor)) {
    factor_prior <- get_factorspec(object)
    factor_prior_type <- enumerate_prior(object$spec_factor$prior)
    factor_init <- object$init_factor
  }
  res_mat <- switch(model_type,
    "bvarsv" = {
      grp_mat <- object$group
      grp_id <- unique(c(grp_mat))
      own_id <- 2
      cross_id <- seq_len(object$p + 1)[-2]
      if (is_exogen) {
        expand_bvarxsv(
          y = y, lag = object$p, num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sv = use_sv, sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_sv = object$sv[c("shape", "scale", "initial_mean", "initial_prec")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test, get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread,
          # exogen = rbind(tail(object$exogen_data, object$s), newxreg), exogen_lag = object$s,
          exogen = rbind(object$exogen_data, newxreg), exogen_lag = object$s,
          exogen_prior = exogen_prior, exogen_init = object$init_exogen, exogen_prior_type = exogen_prior_type
        )
      } else {
        expand_bvarsv(
          y = y, lag = object$p, num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sv = use_sv, sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_sv = object$sv[c("shape", "scale", "initial_mean", "initial_prec")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test, get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread
        )
      }
    },
    "bvharsv" = {
      grp_mat <- object$group
      grp_id <- unique(c(grp_mat))
      if (length(grp_id) == 6) {
        own_id <- c(2, 4, 6)
        cross_id <- c(1, 3, 5)
      } else {
        own_id <- 2
        cross_id <- c(1, 3, 4)
      }
      if (is_exogen) {
        expand_bvharxsv(
          y = y, week = object$week, month = object$month, num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sv = use_sv, sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_sv = object$sv[c("shape", "scale", "initial_mean", "initial_prec")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test, get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread,
          exogen = rbind(object$exogen_data, newxreg), exogen_lag = object$s,
          exogen_prior = exogen_prior, exogen_init = object$init_exogen, exogen_prior_type = exogen_prior_type
        )
      } else {
        expand_bvharsv(
          y = y, week = object$week, month = object$month, num_chains = num_chains, num_iter = object$iter, num_burn = object$burn, thinning = object$thin,
          sv = use_sv, sparse = sparse, level = ci_lev, fit_record = fit_ls, run_mcmc = mcmc,
          param_sv = object$sv[c("shape", "scale", "initial_mean", "initial_prec")],
          param_prior = param_prior, param_intercept = object$intercept, param_init = object$init_coef, prior_type = prior_type, ggl = object$ggl,
          contem_prior = contem_prior, contem_init = object$init_contem, contem_prior_type = contem_prior_type,
          factor_prior = factor_prior, factor_init = factor_init, factor_prior_type = factor_prior_type, size_factor = size_factor, factor_lag = factor_lag,
          grp_id = grp_id, own_id = own_id, cross_id = cross_id, grp_mat = grp_mat,
          include_mean = include_mean, stable = stable, step = n_ahead, y_test = y_test, get_lpl = lpl, use_fit = use_fit,
          seed_chain = sample.int(.Machine$integer.max, size = num_chains * num_horizon) |> matrix(ncol = num_chains),
          seed_forecast = sample.int(.Machine$integer.max, size = num_chains),
          display_progress = verbose, nthreads = num_thread
        )
      }
    }
  )
  num_draw <- nrow(object$param) # concatenate multiple chains
  # num_draw <- nrow(object$param) / num_chains
  if (lpl) {
    # lpl_val <- res_mat$lpl
    if (med) {
      lpl_val <- apply(res_mat$lpl, 1, median)
    } else {
      lpl_val <- rowMeans(res_mat$lpl)
    }
    res_mat$lpl <- NULL
  }
  y_distn <- process_forecast_draws(
    draws = res_mat$forecast,
    n_ahead = n_ahead,
    dim_data = object$m,
    num_draw = num_draw,
    var_names = name_var,
    roll = TRUE,
    med = med
  )
  res <- list(
    process = object$process,
    draws = res_mat$forecast,
    forecast = y_distn$mean,
    se = y_distn$sd,
    lower = y_distn$lower,
    upper = y_distn$upper,
    lower_joint = y_distn$lower,
    upper_joint = y_distn$upper,
    med = med,
    eval_id = n_ahead:nrow(y_test),
    y = y
  )
  if (lpl) {
    res$lpl <- lpl_val
  }
  class(res) <- c("predbvhar_expand", "bvharcv")
  res
}

#' Evaluate the Model Based on MSE (Mean Square Error)
#' 
#' This function computes MSE given prediction result versus evaluation set.
#' 
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data and `predict$forecast`.
#' @param ... not used
#' @return MSE vector corresponding to each variable.
#' @export
mse <- function(x, y, ...) {
  UseMethod("mse", x)
}

#' @rdname mse
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}. Then
#' \deqn{MSE = mean(e_t^2)}
#' MSE is the most used accuracy measure.
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679-688.
#' @export
mse.predbvhar <- function(x, y, ...) {
  (y - x$forecast)^2 |> 
    colMeans()
}

#' @rdname mse
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
mse.bvharcv <- function(x, y, ...) {
  y_test <- y[x$eval_id, ]
  # if (length(dim(x$forecast)) == 3) {
  #   return(
  #     apply(
  #       x$forecast, 3,
  #       function(est) colMeans((est - y_test)^2)
  #     ) |>
  #       rowMeans()
  #   )
  # }
  (y_test - x$forecast)^2 |> 
    colMeans()
}

#' Evaluate the Model Based on MAE (Mean Absolute Error)
#' 
#' This function computes MAE given prediction result versus evaluation set.
#' 
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return MAE vector corressponding to each variable.
#' @export
mae <- function(x, y, ...) {
  UseMethod("mae", x)
}

#' @rdname mae
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' MAE is defined by
#' 
#' \deqn{MSE = mean(\lvert e_t \rvert)}
#' 
#' Some researchers prefer MAE to MSE because it is less sensitive to outliers.
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679-688.
#' @export
mae.predbvhar <- function(x, y, ...) {
  apply(
    y - x$forecast, 
    2, 
    function(e_t) {
      mean(abs(e_t))
    }
  )
}

#' @rdname mae
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
mae.bvharcv <- function(x, y, ...) {
  y_test <- y[x$eval_id, ]
  # if (length(dim(x$forecast)) == 3) {
  #   return(
  #     apply(
  #       x$forecast, 3,
  #       function(est) colMeans(abs(est - y_test))
  #     ) |>
  #       rowMeans()
  #   )
  # }
  apply(
    y_test - x$forecast,
    2,
    function(e_t) {
      mean(abs(e_t))
    }
  )
}

#' Evaluate the Model Based on MAPE (Mean Absolute Percentage Error)
#' 
#' This function computes MAPE given prediction result versus evaluation set.
#' 
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return MAPE vector corresponding to each variable.
#' @export
mape <- function(x, y, ...) {
  UseMethod("mape", x)
}

#' @rdname mape
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' Percentage error is defined by \eqn{p_t = 100 e_t / Y_t} (100 can be omitted since comparison is the focus).
#' 
#' \deqn{MAPE = mean(\lvert p_t \rvert)}
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679-688.
#' @export
mape.predbvhar <- function(x, y, ...) {
  apply(
    100 * (y - x$forecast) / y, 
    2, 
    function(p_t) {
      mean(abs(p_t))
    }
  )
}

#' @rdname mape
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
mape.bvharcv <- function(x, y, ...) {
  y_test <- y[x$eval_id, ]
  # if (length(dim(x$forecast)) == 3) {
  #   return(
  #     apply(
  #       x$forecast, 3,
  #       function(est) colMeans(abs(100 * (est - y_test) / y_test))
  #     ) |>
  #       rowMeans()
  #   )
  # }
  apply(
    100 * (y_test - x$forecast) / y_test,
    2,
    function(p_t) {
      mean(abs(p_t))
    }
  )
}

#' Evaluate the Model Based on MASE (Mean Absolute Scaled Error)
#' 
#' This function computes MASE given prediction result versus evaluation set.
#' 
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return MASE vector corresponding to each variable.
#' @export
mase <- function(x, y, ...) {
  UseMethod("mase", x)
}

#' @rdname mase
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' Scaled error is defined by
#' \deqn{q_t = \frac{e_t}{\sum_{i = 2}^{n} \lvert Y_i - Y_{i - 1} \rvert / (n - 1)}}
#' so that the error can be free of the data scale.
#' Then
#' 
#' \deqn{MASE = mean(\lvert q_t \rvert)}
#' 
#' Here, \eqn{Y_i} are the points in the sample, i.e. errors are scaled by the in-sample mean absolute error (\eqn{mean(\lvert e_t \rvert)}) from the naive random walk forecasting.
#' 
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679-688.
#' @export
mase.predbvhar <- function(x, y, ...) {
  scaled_err <- 
    x$y |> 
    diff() |> 
    abs() |> 
    colMeans()
  apply(
    100 * (y - x$forecast) / scaled_err, 
    2, 
    function(q_t) {
      mean(abs(q_t))
    }
  )
}

#' @rdname mase
#' @param x Forecasting object
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
mase.bvharcv <- function(x, y, ...) {
  scaled_err <-
    x$y |>
    diff() |>
    abs() |>
    colMeans()
  y_test <- y[x$eval_id,]
  # if (length(dim(x$forecast)) == 3) {
  #   return(
  #     apply(
  #       x$forecast, 3,
  #       function(est) colMeans(abs(100 * (y_test - est) / scaled_err))
  #     ) |>
  #       rowMeans()
  #   )
  # }
  apply(
    100 * (y_test - x$forecast) / scaled_err, 
    2, 
    function(q_t) {
      mean(abs(q_t))
    }
  )
}

#' Evaluate the Model Based on MRAE (Mean Relative Absolute Error)
#' 
#' This function computes MRAE given prediction result versus evaluation set.
#' 
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return MRAE vector corresponding to each variable.
#' @export
mrae <- function(x, pred_bench, y, ...) {
  UseMethod("mrae", x)
}

#' @rdname mrae
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' MRAE implements benchmark model as scaling method.
#' Relative error is defined by
#' \deqn{r_t = \frac{e_t}{e_t^{\ast}}}
#' where \eqn{e_t^\ast} is the error from the benchmark method.
#' Then
#' 
#' \deqn{MRAE = mean(\lvert r_t \rvert)}
#' 
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679-688.
#' @export
mrae.predbvhar <- function(x, pred_bench, y, ...) {
  if (!is.predbvhar(pred_bench)) {
    stop("'pred_bench' should be 'predbvhar' class.")
  }
  apply(
    (y - x$forecast) / (y - pred_bench$forecast), 
    2, 
    function(r_t) {
      mean(abs(r_t))
    }
  )
}

#' @rdname mrae
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
mrae.bvharcv <- function(x, pred_bench, y, ...) {
  if (!is.bvharcv(pred_bench)) {
    stop("'pred_bench' should be 'bvharcv' class.")
  }
  y_test <- y[x$eval_id,]
  # if (length(dim(x$forecast)) == 3) {
  #   stop("Not defined yet")
  # }
  apply(
    (y_test - x$forecast) / (y_test - pred_bench$forecast),
    2,
    function(r_t) {
      mean(abs(r_t))
    }
  )
}

#' Evaluate the Density Forecast Based on Average Log Predictive Likelihood (APLP)
#'
#' This function computes ALPL given forecasting of Bayesian models.
#'
#' @param x Out-of-sample forecasting object to use
#' @param ... Not used
#' @export
alpl <- function(x, ...) {
  UseMethod("alpl", x)
}

#' @rdname alpl
#' @export 
alpl.bvharcv <- function(x, ...) {
  if (x$med) {
    return(median(x$lpl))
  }
  mean(x$lpl)
}

#' Evaluate the Model Based on RelMAE (Relative MAE)
#' 
#' This function computes RelMAE given prediction result versus evaluation set.
#' 
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return RelMAE vector corresponding to each variable.
#' @export
relmae <- function(x, pred_bench, y, ...) {
  UseMethod("relmae", x)
}

#' @rdname relmae
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' RelMAE implements MAE of benchmark model as relative measures.
#' Let \eqn{MAE_b} be the MAE of the benchmark model.
#' Then
#' 
#' \deqn{RelMAE = \frac{MAE}{MAE_b}}
#' 
#' where \eqn{MAE} is the MAE of our model.
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679-688.
#' @export
relmae.predbvhar <- function(x, pred_bench, y, ...) {
  mae(x, y) / mae(pred_bench, y)
}

#' @rdname relmae
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
relmae.bvharcv <- function(x, pred_bench, y, ...) {
  mae(x, y) / mae(pred_bench, y)
}

#' Evaluate the Model Based on RMAPE (Relative MAPE)
#' 
#' This function computes RMAPE given prediction result versus evaluation set.
#' 
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return RMAPE vector corresponding to each variable.
#' @export
rmape <- function(x, pred_bench, y, ...) {
  UseMethod("rmape", x)
}

#' @rdname rmape
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' RMAPE is the ratio of MAPE of given model and the benchmark one.
#' Let \eqn{MAPE_b} be the MAPE of the benchmark model.
#' Then
#' 
#' \deqn{RMAPE = \frac{mean(MAPE)}{mean(MAPE_b)}}
#' 
#' where \eqn{MAPE} is the MAPE of our model.
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679-688.
#' @export
rmape.predbvhar <- function(x, pred_bench, y, ...) {
  mean(mape(x, y)) / mean(mape(pred_bench, y))
}

#' @rdname rmape
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
rmape.bvharcv <- function(x, pred_bench, y, ...) {
  mean(mape(x, y)) / mean(mape(pred_bench, y))
}

#' Evaluate the Model Based on RMASE (Relative MASE)
#' 
#' This function computes RMASE given prediction result versus evaluation set.
#' 
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return RMASE vector corresponding to each variable.
#' @export
rmase <- function(x, pred_bench, y, ...) {
  UseMethod("rmase", x)
}

#' @rdname rmase
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' RMASE is the ratio of MAPE of given model and the benchmark one.
#' Let \eqn{MASE_b} be the MAPE of the benchmark model.
#' Then
#' 
#' \deqn{RMASE = \frac{mean(MASE)}{mean(MASE_b)}}
#' 
#' where \eqn{MASE} is the MASE of our model.
#' @references Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679-688.
#' @export
rmase.predbvhar <- function(x, pred_bench, y, ...) {
  mean(mase(x, y)) / mean(mase(pred_bench, y))
}

#' @rdname rmase
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
rmase.bvharcv <- function(x, pred_bench, y, ...) {
  mean(mase(x, y)) / mean(mase(pred_bench, y))
}

#' Evaluate the Model Based on RMSFE
#' 
#' This function computes RMSFE (Mean Squared Forecast Error Relative to the Benchmark)
#' 
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return RMSFE vector corresponding to each variable.
#' @export
rmsfe <- function(x, pred_bench, y, ...) {
  UseMethod("rmsfe", x)
}

#' @rdname rmsfe
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' RMSFE is the ratio of L2 norm of \eqn{e_t} from forecasting object and from benchmark model.
#' 
#' \deqn{RMSFE = \frac{sum(\lVert e_t \rVert)}{sum(\lVert e_t^{(b)} \rVert)}}
#' 
#' where \eqn{e_t^{(b)}} is the error from the benchmark model.
#' @references 
#' Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679-688.
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1).
#' 
#' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526).
#' @export
rmsfe.predbvhar <- function(x, pred_bench, y, ...) {
  sum(mse(x, y)) / sum(mse(pred_bench, y))
}

#' @rdname rmsfe
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
rmsfe.bvharcv <- function(x, pred_bench, y, ...) {
  sum(mse(x, y)) / sum(mse(pred_bench, y))
}

#' Evaluate the Model Based on RMAFE
#' 
#' This function computes RMAFE (Mean Absolute Forecast Error Relative to the Benchmark)
#' 
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @return RMAFE vector corresponding to each variable.
#' @export
rmafe <- function(x, pred_bench, y, ...) {
  UseMethod("rmafe", x)
}

#' @rdname rmafe
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @details 
#' Let \eqn{e_t = y_t - \hat{y}_t}.
#' RMAFE is the ratio of L1 norm of \eqn{e_t} from forecasting object and from benchmark model.
#' 
#' \deqn{RMAFE = \frac{sum(\lVert e_t \rVert)}{sum(\lVert e_t^{(b)} \rVert)}}
#' 
#' where \eqn{e_t^{(b)}} is the error from the benchmark model.
#' 
#' @references 
#' Hyndman, R. J., & Koehler, A. B. (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting, 22(4), 679-688.
#' 
#' Bańbura, M., Giannone, D., & Reichlin, L. (2010). *Large Bayesian vector auto regressions*. Journal of Applied Econometrics, 25(1).
#' 
#' Ghosh, S., Khare, K., & Michailidis, G. (2018). *High-Dimensional Posterior Consistency in Bayesian Vector Autoregressive Models*. Journal of the American Statistical Association, 114(526).
#' @export
rmafe.predbvhar <- function(x, pred_bench, y, ...) {
  sum(mae(x, y)) / sum(mae(pred_bench, y))
}

#' @rdname rmafe
#' @param x Forecasting object to use
#' @param pred_bench The same forecasting object from benchmark model
#' @param y Test data to be compared. should be the same format with the train data.
#' @param ... not used
#' @export
rmafe.bvharcv <- function(x, pred_bench, y, ...) {
  sum(mae(x, y)) / sum(mae(pred_bench, y))
}
