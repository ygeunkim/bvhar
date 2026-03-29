#' Impulse Response Analysis
#' 
#' Computes responses to impulses or orthogonal impulses
#' 
#' @param object Model object
#' @param lag_max Maximum lag to investigate the impulse responses (By default, `10`)
#' @param orthogonal Orthogonal impulses (`TRUE`) or just impulses (`FALSE`)
#' @param impulse_var Impulse variables character vector. If not specified, use every variable.
#' @param response_var Response variables character vector. If not specified, use every variable.
#' @param ... not used
#' @return `bvharirf` [class]
#' @export
irf <- function(object, lag_max, orthogonal, impulse_var, response_var, ...) {
  UseMethod("irf", object)
}

#' @rdname irf
#' @section Responses to forecast errors:
#' If `orthogonal = FALSE`, the function gives \eqn{W_j} VMA representation of the process such that
#' \deqn{Y_t = \sum_{j = 0}^\infty W_j \epsilon_{t - j}}
#' @section Responses to orthogonal impulses:
#' If `orthogonal = TRUE`, it gives orthogonalized VMA representation \deqn{\Theta}.
#' Based on variance decomposition (Cholesky decomposition)
#' \deqn{\Sigma = P P^T}
#' where \eqn{P} is lower triangular matrix,
#' impulse response analysis if performed under MA representation
#' \deqn{y_t = \sum_{i = 0}^\infty \Theta_i v_{t - i}}
#' Here,
#' \deqn{\Theta_i = W_i P}
#' and \eqn{v_t = P^{-1} \epsilon_t} are orthogonal.
#' @references Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
#' @importFrom dplyr mutate filter
#' @importFrom tidyr pivot_longer
#' @order 1
#' @export
irf.varlse <- function(object,
                       lag_max = 10,
                       orthogonal = TRUE,
                       impulse_var = NULL,
                       response_var = NULL,
                       ...) {
  mat_coef <- object$coefficients
  mat_irf <- compute_var_irf(
    coef_mat = mat_coef,
    lag = object$p,
    cov_mat = object$covmat,
    step = lag_max + 1,
    orthogonal = orthogonal
  )
  # preprocess-------------------
  name_var <- colnames(mat_coef)
  if (is.null(impulse_var)) {
    impulse_var <- name_var
  }
  if (is.null(response_var)) {
    response_var <- name_var
  }
  impulse_name <- rep(name_var, lag_max + 1)
  period_name <- rep(seq_len(lag_max + 1) - 1, each = object$m)
  colnames(mat_irf) <- name_var
  rownames(mat_irf) <- paste0(
    impulse_name,
    "(i=",
    period_name,
    ")"
  )
  res <- list(coefficients = mat_irf)
  res$df_long <- 
    mat_irf |> 
    as.data.frame() |> 
    mutate(
      impulse = impulse_name,
      period = period_name
    ) |> 
    pivot_longer(
      -c(period, impulse),
      names_to = "response",
      values_to = "value"
    ) |> 
    filter(impulse %in% impulse_var, response %in% response_var)
  # return----------------------
  res$lag_max <- lag_max
  res$orthogonal <- orthogonal
  res$process <- object$process
  class(res) <- "bvharirf"
  res
}

#' @rdname irf
#' @importFrom dplyr mutate
#' @importFrom tidyr pivot_longer
#' @order 1
#' @export
irf.vharlse <- function(object, 
                        lag_max = 10,
                        orthogonal = TRUE,
                        impulse_var = NULL,
                        response_var = NULL,
                        ...) {
  mat_coef <- object$coefficients
  mat_irf <- compute_vhar_irf(
    coef_mat = mat_coef,
    week = object$week,
    month = object$month,
    cov_mat = object$covmat,
    step = lag_max + 1,
    orthogonal = orthogonal
  )
  # preprocess-------------------
  name_var <- colnames(mat_coef)
  if (is.null(impulse_var)) {
    impulse_var <- name_var
  }
  if (is.null(response_var)) {
    response_var <- name_var
  }
  impulse_name <- rep(name_var, lag_max + 1)
  period_name <- rep(seq_len(lag_max + 1) - 1, each = object$m)
  colnames(mat_irf) <- name_var
  rownames(mat_irf) <- paste0(
    impulse_name,
    "(i=",
    period_name,
    ")"
  )
  res <- list(coefficients = mat_irf)
  res$df_long <- 
    mat_irf |> 
    as.data.frame() |> 
    mutate(
      impulse = impulse_name,
      period = period_name
    ) |> 
    pivot_longer(
      -c(period, impulse),
      names_to = "response",
      values_to = "value"
    ) |> 
    filter(impulse %in% impulse_var, response %in% response_var)
  # return----------------------
  res$lag_max <- lag_max
  res$orthogonal <- orthogonal
  res$process <- object$process
  class(res) <- "bvharirf"
  res
}

#' @rdname irf
#' @param level Specify alpha of confidence interval level 100(1 - alpha) percentage. By default, .05.
#' @param num_thread Number of threads
#' @param sparse `r lifecycle::badge("experimental")` Apply restriction. By default, `FALSE`.
#' Give CI level (e.g. `.05`) instead of `TRUE` to use credible interval across MCMC for restriction.
#' @param med `r lifecycle::badge("experimental")` If `TRUE`, use median of forecast draws instead of mean (default).
#' @importFrom tidyr separate
#' @importFrom dplyr rename
#' @order 1
#' @export
irf.bvarldlt <- function(object,
                         lag_max = 10,
                         orthogonal = TRUE,
                         impulse_var = NULL,
                         response_var = NULL,
                         level = .05,
                         num_thread = 1,
                         sparse = FALSE,
                         med = FALSE,
                         ...) {
  num_chains <- object$chain
  dim_data <- object$m
  num_draw <- nrow(object$param)
  # ci_lev <- 0
  # if (is.numeric(sparse)) {
  #   ci_lev <- sparse
  #   sparse <- FALSE
  # }
  fit_ls <- get_records(object, TRUE)
  irf_res <- compute_varldlt_irf(
    num_chains = num_chains,
    lag = object$p,
    step = lag_max + 1,
    fit_record = fit_ls,
    sparse = sparse,
    nthreads = num_thread
  ) # list of dim * step x num_draw * dim
  # preprocess-------------------
  name_var <- colnames(object$coefficients)
  if (is.null(impulse_var)) {
    impulse_var <- name_var
  }
  if (is.null(response_var)) {
    response_var <- name_var
  }
  res <- process_irf_draws(
    irf_res,
    dim_data = dim_data,
    lag_max = lag_max,
    num_draw = num_draw,
    var_names = name_var,
    impulse_var = impulse_var,
    response_var = response_var,
    level = level,
    med = med
  )
  # return----------------------
  res$lag_max <- lag_max
  res$orthogonal <- orthogonal
  res$process <- object$process
  class(res) <- "bvharirf"
  res
}

#' @rdname irf
#' @order 1
#' @export
irf.bvharldlt <- function(object,
                         lag_max = 10,
                         orthogonal = TRUE,
                         impulse_var = NULL,
                         response_var = NULL,
                         level = .05,
                         num_thread = 1,
                         sparse = FALSE,
                         med = FALSE,
                         ...) {
  num_chains <- object$chain
  dim_data <- object$m
  num_draw <- nrow(object$param)
  # ci_lev <- 0
  # if (is.numeric(sparse)) {
  #   ci_lev <- sparse
  #   sparse <- FALSE
  # }
  fit_ls <- get_records(object, TRUE)
  irf_res <- compute_vharldlt_irf(
    num_chains = num_chains,
    week = object$week,
    month = object$month,
    step = lag_max + 1,
    fit_record = fit_ls,
    sparse = sparse,
    nthreads = num_thread
  ) # list of dim * step x num_draw * dim
  # preprocess-------------------
  name_var <- colnames(object$coefficients)
  if (is.null(impulse_var)) {
    impulse_var <- name_var
  }
  if (is.null(response_var)) {
    response_var <- name_var
  }
  res <- process_irf_draws(
    irf_res,
    dim_data = dim_data,
    lag_max = lag_max,
    num_draw = num_draw,
    var_names = name_var,
    impulse_var = impulse_var,
    response_var = response_var,
    level = level,
    med = med
  )
  # return----------------------
  res$lag_max <- lag_max
  res$orthogonal <- orthogonal
  res$process <- object$process
  class(res) <- "bvharirf"
  res
}
