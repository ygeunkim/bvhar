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
#' @seealso [VARtoVMA()]
#' @importFrom dplyr mutate filter
#' @importFrom tidyr pivot_longer
#' @order 1
#' @export
irf.varlse <- function(object,
                       lag_max = 10,
                       orthogonal = TRUE,
                       impulse_var,
                       response_var,
                       ...) {
  mat_coef <- object$coefficients
  mat_irf <- matrix()
  if (orthogonal) {
    mat_irf <- VARcoeftoVMA_ortho(
      var_coef = mat_coef, 
      var_covmat = object$covmat, 
      var_lag = object$p,
      lag_max = lag_max
    )
  } else {
    mat_irf <- VARcoeftoVMA(
      var_coef = mat_coef,
      var_lag = object$p,
      lag_max = lag_max
    )
  }
  # preprocess-------------------
  name_var <- colnames(mat_coef)
  if (missing(impulse_var)) {
    impulse_var <- name_var
  }
  if (missing(response_var)) {
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
#' @seealso [VHARtoVMA()]
#' @importFrom dplyr mutate
#' @importFrom tidyr pivot_longer
#' @order 1
#' @export
irf.vharlse <- function(object, 
                        lag_max = 10,
                        orthogonal = TRUE,
                        impulse_var,
                        response_var,
                        ...) {
  mat_coef <- object$coefficients
  mat_irf <- matrix()
  if (orthogonal) {
    mat_irf <- VHARcoeftoVMA_ortho(
      vhar_coef = mat_coef, 
      vhar_covmat = object$covmat, 
      HARtrans_mat = object$HARtrans,
      lag_max = lag_max,
      month = object$month
    )
  } else {
    mat_irf <- VHARcoeftoVMA(
      vhar_coef = mat_coef,
      HARtrans_mat = object$HARtrans,
      lag_max = lag_max,
      month = object$month
    )
  }
  # preprocess-------------------
  name_var <- colnames(mat_coef)
  if (missing(impulse_var)) {
    impulse_var <- name_var
  }
  if (missing(response_var)) {
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
