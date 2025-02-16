#' Validate Input matrix
#' 
#' @param y Time series data of which columns indicate the variables
#' 
#' @noRd
validate_input <- function(y) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    return(as.matrix(y))
  }
  y
}

#' Validate Bayesian configuration input
#'
#' @param bayes_spec A BVHAR model specification by [set_bvhar()] (default) [set_weight_bvhar()], [set_ssvs()], or [set_horseshoe()].
#' @param cov_spec SV specification by [set_sv()].
#' @param intercept Prior for the constant term by [set_intercept()].
#' @noRd
validate_spec <- function(y, dim_data, p, num_grp, grp_id, own_id, cross_id,
                          bayes_spec, cov_spec, intercept, prior_nm, process = c("BVAR", "BVHAR")) {
  process <- match.arg(process)
  if (!(
    is.bvharspec(bayes_spec) ||
      is.ssvsinput(bayes_spec) ||
      is.horseshoespec(bayes_spec) ||
      is.ngspec(bayes_spec) ||
      is.dlspec(bayes_spec) ||
      is.gdpspec(bayes_spec)
  )) {
    stop("Provide 'bvharspec', 'ssvsinput', 'horseshoespec', 'ngspec', 'dlspec', or 'gdpspec' for 'bayes_spec'.")
  }
  # Covariance-----------------
  if (!is.covspec(cov_spec)) {
    stop("Provide 'covspec' for 'cov_spec'.")
  }
  if (!is.interceptspec(intercept)) {
    stop("Provide 'interceptspec' for 'intercept'.")
  }
  if (prior_nm == "Minnesota" || prior_nm == "MN_Hierarchical") {
    if (process == "BVAR") {
      if (bayes_spec$process != process) {
        stop("'bayes_spec' must be the result of 'set_bvar()'.")
      }
      if (is.null(bayes_spec$delta)) {
        bayes_spec$delta <- rep(0, dim_data)
      }
      if (length(bayes_spec$delta) == 1) {
        bayes_spec$delta <- rep(bayes_spec$delta, dim_data)
      }
    } else {
      if (bayes_spec$process != process) {
        stop("'bayes_spec' must be the result of 'set_bvhar()' or 'set_weight_bvhar()'.")
      }
      if ("delta" %in% names(bayes_spec)) {
        if (is.null(bayes_spec$delta)) {
          bayes_spec$delta <- rep(0, dim_data)
        }
        if (length(bayes_spec$delta) == 1) {
          bayes_spec$delta <- rep(bayes_spec$delta, dim_data)
        }
      } else {
        if (is.null(bayes_spec$daily)) {
          bayes_spec$daily <- rep(0, dim_data)
        }
        if (is.null(bayes_spec$weekly)) {
          bayes_spec$weekly <- rep(0, dim_data)
        }
        if (is.null(bayes_spec$monthly)) {
          bayes_spec$monthly <- rep(0, dim_data)
        }
        if (length(bayes_spec$daily) == 1) {
          bayes_spec$daily <- rep(bayes_spec$daily, dim_data)
        }
        if (length(bayes_spec$weekly) == 1) {
          bayes_spec$weekly <- rep(bayes_spec$weekly, dim_data)
        }
        if (length(bayes_spec$monthly) == 1) {
          bayes_spec$monthly <- rep(bayes_spec$monthly, dim_data)
        }
      }
      p <- 3
    }
    if (is.null(bayes_spec$sigma)) {
      bayes_spec$sigma <- apply(y, 2, sd)
    }
    param_prior <- append(bayes_spec, list(p = p))
    if (bayes_spec$hierarchical) {
      param_prior$shape <- bayes_spec$lambda$param[1]
      param_prior$rate <- bayes_spec$lambda$param[2]
      param_prior$grid_size <- bayes_spec$lambda$grid_size
    }
  } else if (prior_nm == "SSVS") {
    if (length(bayes_spec$coef_s1) == 2) {
      coef_s1 <- numeric(num_grp)
      coef_s1[grp_id %in% own_id] <- bayes_spec$coef_s1[1]
      coef_s1[grp_id %in% cross_id] <- bayes_spec$coef_s1[2]
      bayes_spec$coef_s1 <- coef_s1
    }
    if (length(bayes_spec$coef_s2) == 2) {
      coef_s2 <- numeric(num_grp)
      coef_s2[grp_id %in% own_id] <- bayes_spec$coef_s2[1]
      coef_s2[grp_id %in% cross_id] <- bayes_spec$coef_s2[2]
      bayes_spec$coef_s2 <- coef_s2
    }
    param_prior <- bayes_spec
  } else if (prior_nm == "Horseshoe") {
    param_prior <- list()
  } else {
    param_prior <- bayes_spec
  }
  list(
    spec = bayes_spec,
    hyperparam = param_prior
  )
}

#' Initialize MCMC (Temporarily used before moving into C++)
#' @noRd
init_shrinkage_prior <- function(prior_nm, num_chains = 1, dim_data, dim_design, num_eta, num_alpha, num_grp) {
  param_init <- lapply(
    seq_len(num_chains),
    function(x) {
      list(
        init_coef = matrix(runif(dim_data * dim_design, -1, 1), ncol = dim_data),
        init_contem = exp(runif(num_eta, -1, 0)) # Cholesky factor
      )
    }
  )
  switch(
    prior_nm,
    "Minnesota" = {
      param_init
    },
    "MN_Hierarchical" = {
      lapply(
        param_init,
        function(init) {
          append(
            init,
            list(
              own_lambda = runif(1, 0, 1),
              cross_lambda = runif(1, 0, 1),
              contem_lambda = runif(1, 0, 1)
            )
          )
        }
      )
    },
    "SSVS" = {
      lapply(
        param_init,
        function(init) {
          coef_mixture <- runif(num_grp, -1, 1)
          coef_mixture <- exp(coef_mixture) / (1 + exp(coef_mixture)) # minnesota structure?
          init_coef_dummy <- rbinom(num_alpha, 1, .5) # minnesota structure?
          chol_mixture <- runif(num_eta, -1, 1)
          chol_mixture <- exp(chol_mixture) / (1 + exp(chol_mixture))
          init_coef_slab <- exp(runif(num_alpha, -1, 1))
          init_contem_slab <- exp(runif(num_eta, -1, 1))
          append(
            init,
            list(
              init_coef_dummy = init_coef_dummy,
              coef_mixture = coef_mixture,
              coef_slab = init_coef_slab,
              chol_mixture = chol_mixture,
              contem_slab = init_contem_slab,
              coef_spike_scl = runif(1, 0, 1),
              chol_spike_scl = runif(1, 0, 1)
            )
          )
        }
      )
    },
    "Horseshoe" = {
      lapply(
        param_init,
        function(init) {
          local_sparsity <- exp(runif(num_alpha, -1, 1))
          global_sparsity <- exp(runif(1, -1, 1))
          group_sparsity <- exp(runif(num_grp, -1, 1))
          contem_local_sparsity <- exp(runif(num_eta, -1, 1)) # sd = local * global
          contem_global_sparsity <- exp(runif(1, -1, 1)) # sd = local * global
          append(
            init,
            list(
              local_sparsity = local_sparsity,
              global_sparsity = global_sparsity,
              group_sparsity = group_sparsity,
              contem_local_sparsity = contem_local_sparsity,
              contem_global_sparsity = contem_global_sparsity
            )
          )
        }
      )
    },
    "NG" = {
      lapply(
        param_init,
        function(init) {
          local_sparsity <- exp(runif(num_alpha, -1, 1))
          global_sparsity <- exp(runif(1, -1, 1))
          group_sparsity <- exp(runif(num_grp, -1, 1))
          contem_local_sparsity <- exp(runif(num_eta, -1, 1)) # sd = local * global
          contem_global_sparsity <- exp(runif(1, -1, 1)) # sd = local * global
          append(
            init,
            list(
              local_shape = runif(num_grp, 0, 1),
              contem_shape = runif(1, 0, 1),
              local_sparsity = local_sparsity,
              global_sparsity = global_sparsity,
              group_sparsity = group_sparsity,
              contem_local_sparsity = contem_local_sparsity,
              contem_global_sparsity = contem_global_sparsity
            )
          )
        }
      )
    },
    "DL" = {
      lapply(
        param_init,
        function(init) {
          local_sparsity <- exp(runif(num_alpha, -1, 1))
          global_sparsity <- exp(runif(1, -1, 1))
          contem_local_sparsity <- exp(runif(num_eta, -1, 1)) # sd = local * global
          contem_global_sparsity <- exp(runif(1, -1, 1)) # sd = local * global
          append(
            init,
            list(
              local_sparsity = local_sparsity,
              global_sparsity = global_sparsity,
              contem_local_sparsity = contem_local_sparsity,
              contem_global_sparsity = contem_global_sparsity
            )
          )
        }
      )
    },
    "GDP" = {
      lapply(
        param_init,
        function(init) {
          local_sparsity <- exp(runif(num_alpha, -1, 1))
          group_rate <- exp(runif(num_grp, -1, 1))
          contem_local_sparsity <- exp(runif(num_eta, -1, 1)) # sd = local * global
          contem_local_rate <- exp(runif(num_eta, -1, 1))
          coef_shape <- runif(1, 0, 1)
          coef_rate <- runif(1, 0, 1)
          contem_shape <- runif(1, 0, 1)
          contem_rate <- runif(1, 0, 1)
          append(
            init,
            list(
              local_sparsity = local_sparsity,
              group_rate = group_rate,
              contem_local_sparsity = contem_local_sparsity,
              contem_rate = contem_local_rate,
              gamma_shape = coef_shape,
              gamma_rate = coef_rate,
              contem_gamma_shape = contem_shape,
              contem_gamma_rate = contem_rate
            )
          )
        }
      )
    }
  )
}

#' @noRd
concatenate_colnames <- function(var_name, prefix, include_mean = TRUE) {
  nm <- 
    lapply(
      prefix,
      function(lag) paste(var_name, lag, sep = "_")
    ) |> 
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
#' @param object A `bvharmod` object
#' @param ... not used
#' @details 
#' Each result of [var_lm()], [vhar_lm()], [bvar_minnesota()], [bvar_flat()], and [bvhar_minnesota()] is a subclass of `bvharmod`.
#' For example,
#' `c("varlse", "bvharmod")`.
#' @return A `list` object
#' @keywords internal
#' @noRd
split_coef <- function(object, ...) {
  if (!(is.bvharmod(object) || is.bvharirf(object))) {
    stop("Not valid method")
  }
  if (is.bvharmod(object)) {
    return(
      switch(object$type,
        "const" = {
          split.data.frame(object$coefficients[-object$df, ], gl(object$p, object$m)) |>
            lapply(t)
        },
        "none" = {
          split.data.frame(object$coefficients, gl(object$p, object$m)) |>
            lapply(t)
        }
      )
    )
  } else if (is.bvharirf(object)) {
    # 
    irf_mat <- object$coefficients
    return(
      split.data.frame(irf_mat, gl(object$lag_max + 1, ncol(irf_mat)))
    )
  } else {
    stop("Not valid method")
  }
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
    split.data.frame(t(x), gl(num_var, 1, ncol(x))) |>
    lapply(t) |>
    unlist() |> 
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
    x |> 
    split.data.frame(gl(nrow(x) / ncol(x), ncol(x)))
  if (chain == 1) {
    return(res)
  } else {
    res <- lapply(
      res,
      function(y) {
        num_var <- ncol(y) / chain
        split.data.frame(t(y), gl(num_var, 1, ncol(y))) |> 
          lapply(t) |> 
          unlist() |> 
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

#' Split Multi-chain MCMC Records
#' 
#' Preprocess multi-chain MCMC records, which is column-binded.
#'
#' @param x Parameter matrix
#' @param chain The number of the chains
#' @noRd
split_chain <- function(x, chain = 1, varname = "alpha") {
  if (chain == 1) {
    return(x)
  } else {
    # matrix format: [chain1, chain2, ...]
    # num_var <- ncol(x) / chain
    num_row <- nrow(x) / chain
    res <-
      # split.data.frame(t(x), gl(num_var, 1, ncol(x))) |>
      # lapply(t) |>
      split.data.frame(x, gl(chain, num_row)) |>
      unlist(x) |>
      array(
        # dim = c(nrow(x), chain, num_var),
        dim = c(num_row, chain, ncol(x)),
        dimnames = list(
          # iteration = seq_len(nrow(x)),
          iteration = seq_len(num_row),
          chain = seq_len(chain),
          # variable = paste0(varname, "[", seq_len(num_var), "]")
          variable = paste0(varname, "[", seq_len(ncol(x)), "]")
        )
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

#' Compute Summaries from Forecast Draws
#' 
#' @param draws Matrix in forms of rbind(step) x cbind(draws)
#' @param n_ahead Forecast step used
#' @param dim_data Dimension
#' @param num_draw MCMC draws
#' @param var_names Variable names
#' @param level level for lower and upper quantiles
#' @param med Get median instead of mean?
#' @param roll Is the `draws` the result of rolling or expanding windows?
#' 
#' @noRd 
process_forecast_draws <- function(draws, n_ahead, dim_data, num_draw, var_names, level = .05, roll = FALSE, med = FALSE) {
  if (roll) {
    if (med) {
      pred_mean <-
        draws |>
        lapply(function(res) {
          unlist(res) |>
            array(dim = c(n_ahead, dim_data, num_draw)) |>
            apply(c(1, 2), median)
        })
    } else {
      pred_mean <-
        draws |>
        lapply(function(res) {
          unlist(res) |>
            array(dim = c(n_ahead, dim_data, num_draw)) |>
            apply(c(1, 2), mean)
        })
    }
    pred_mean <- do.call(rbind, pred_mean)
    pred_se <-
      draws |>
      lapply(function(res) {
        unlist(res) |>
          array(dim = c(n_ahead, dim_data, num_draw)) |>
          apply(c(1, 2), sd)
      })
    pred_se <- do.call(rbind, pred_se)
    pred_lower <-
      draws |> 
      lapply(function(res) {
        unlist(res) |> 
          array(dim = c(n_ahead, dim_data, num_draw)) |> 
          apply(c(1, 2), quantile, probs = level / 2)
      })
    pred_lower <- do.call(rbind, pred_lower)
    pred_upper <-
      draws |>
      lapply(function(res) {
        unlist(res) |>
          array(dim = c(n_ahead, dim_data, num_draw)) |>
          apply(c(1, 2), quantile, probs = 1 - level / 2)
      })
    pred_upper <- do.call(rbind, pred_upper)
  } else {
    mcmc_distn <-
      draws |>
      unlist() |>
      array(dim = c(n_ahead, dim_data, num_draw))
    if (med) {
      pred_mean <- apply(mcmc_distn, c(1, 2), median)
    } else {
      pred_mean <- apply(mcmc_distn, c(1, 2), mean)
    }
    pred_se <- apply(mcmc_distn, c(1, 2), sd)
    pred_lower <- apply(mcmc_distn, c(1, 2), quantile, probs = level / 2)
    pred_upper <- apply(mcmc_distn, c(1, 2), quantile, probs = 1 - level / 2)
  }
  colnames(pred_mean) <- var_names
  colnames(pred_se) <- var_names
  colnames(pred_lower) <- var_names
  colnames(pred_upper) <- var_names
  if (nrow(pred_mean) == ncol(pred_mean)) {
    rownames(pred_mean) <- var_names
    rownames(pred_se) <- var_names
    rownames(pred_lower) <- var_names
    rownames(pred_upper) <- var_names
  }
  list(
    mean = pred_mean,
    sd = pred_se,
    lower = pred_lower,
    upper = pred_upper
  )
}

#' Compute Summaries from Vector Draws
#'
#' @param dim_data Dimension
#' @param level level for lower and upper quantiles
#' @param med Get median instead of mean?
#'
#' @noRd
process_vector_draws <- function(draws, dim_data, level = .05, med = FALSE) {
  mcmc_distn <- matrix(draws, ncol = dim_data)
  if (med) {
    pred_mean <- apply(mcmc_distn, 2, median)
  } else {
    pred_mean <- colMeans(mcmc_distn)
  }
  list(
    mean = pred_mean,
    sd = apply(mcmc_distn, 2, sd),
    lower = apply(mcmc_distn, 2, quantile, probs = level / 2),
    upper = apply(mcmc_distn, 2, quantile, probs = 1 - level / 2)
  )
}

#' Compute Summaries from Dynamic Spillover
#'
#' @param dim_data Dimension
#' @param level level for lower and upper quantiles
#' @param med Get median instead of mean?
#' @param var_names Variable names
#' 
#' @importFrom tibble as_tibble
#' @importFrom dplyr mutate
#' @noRd 
process_dynamic_spdraws <- function(draws, dim_data, level = .05, med = FALSE, var_names) {
  sp_draws <- lapply(
    draws,
    function(x) {
      do.call(
        cbind,
        process_vector_draws(unlist(x), dim_data = dim_data, level = level, med = med)
      ) |>
        as.data.frame() |>
        mutate(series = var_names)
    }
  )
  do.call(rbind, sp_draws) |> 
    as_tibble()
}

#' Pivot longer spillover
#' 
#' @param connect Connectedness table
#' @param col_names Column name for value
#' @noRd 
gather_spillover <- function(connect, col_names = "spillover") {
  connect |>
    as.data.frame() |>
    rownames_to_column(var = "series") |>
    pivot_longer(-"series", names_to = "shock", values_to = col_names)
}

#' Pivot longer spillover summaries
#'
#' @param distn Connectedness table distribution
#' @param prefix Column names prefix
#' 
#' @noRd
join_long_spillover <- function(connect, prefix = "spillover") {
  gather_spillover(connect$mean, col_names = prefix) |>
    left_join(gather_spillover(connect$lower, col_names = paste(prefix, "lower", sep = "_")), by = c("series", "shock")) |>
    left_join(gather_spillover(connect$upper, col_names = paste(prefix, "upper", sep = "_")), by = c("series", "shock")) |>
    left_join(gather_spillover(connect$sd, col_names = paste(prefix, "sd", sep = "_")), by = c("series", "shock"))
}

#' Define Minnesota Group Matrix
#'
#' This function creates a matrix with group index
#'
#' @param p VAR(p) or VHAR order (3 when VHAR)
#' @param dim_data Data dimension
#' @param dim_design Number of rows of coefficients matrix (kp + 1 or 3k + 1)
#' @param num_coef Length of coefficients to be restricted
#' @param minnesota Shrinkage structure
#' @param include_mean Constant term
#' @noRd
build_grpmat <- function(p, dim_data, dim_design, num_coef, minnesota, include_mean) {
  if (include_mean) {
    idx <- c(gl(p, dim_data), p + 1)
  } else {
    idx <- gl(p, dim_data)
  }
  if (p == 1) {
    glob_idmat <- matrix(1L, nrow = dim_design, ncol = dim_data)
    if (minnesota == "no") {
      return(glob_idmat)
    }
    if (include_mean) {
      glob_idmat[dim_design,] <- 0L
    }
    diag(glob_idmat[1:dim_data,]) <- 2L
    return(glob_idmat)
  }
  switch(
    minnesota,
    "no" = matrix(1L, nrow = dim_design, ncol = dim_data),
    "short" = {
      glob_idmat <- split.data.frame(
        matrix(rep(0, num_coef), ncol = dim_data),
        idx
      )
      glob_idmat[[1]] <- diag(dim_data) + 1
      id <- 1
      for (i in 2:p) {
        glob_idmat[[i]] <- matrix(i + 1, nrow = dim_data, ncol = dim_data)
        id <- id + 2
      }
      do.call(rbind, glob_idmat)
    },
    "longrun" = {
      glob_idmat <- split.data.frame(
        matrix(rep(0, num_coef), ncol = dim_data),
        idx
      )
      id <- 1
      for (i in 1:p) {
        glob_idmat[[i]] <- diag(dim_data) + id
        id <- id + 2
      }
      do.call(rbind, glob_idmat)
    }
  )
}

#' Get MCMC records as a list
#' 
#' @param object Model list
#' @param split_chain Split each chain as list
#' @noRd
get_records <- function(object, split_chain = TRUE) {
  num_chains <- 1
  if (split_chain) {
    num_chains <- object$chain
  }
  lapply(
    object$param_names,
    function(x) {
      subset_draws(object$param, variable = x) |>
        as_draws_matrix() |>
        split.data.frame(gl(num_chains, nrow(object$param) / num_chains))
    }
  ) |>
    setNames(paste(object$param_names, "record", sep = "_"))
}
