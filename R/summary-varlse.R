#' Summarizing Vector Autoregressive Model
#' 
#' `summary` method for `varlse` class.
#' 
#' @param object A `varlse` object
#' @param ... not used
#' 
#' @return `summary.varlse` [class] additionally computes the following
#' \item{`names`}{Variable names}
#' \item{`totobs`}{Total number of the observation}
#' \item{`obs`}{Sample size used when training = `totobs` - `p`}
#' \item{`p`}{Lag of VAR}
#' \item{`coefficients`}{Coefficient Matrix}
#' \item{`call`}{Matched call}
#' \item{`process`}{Process: VAR}
#' \item{`covmat`}{Covariance matrix of the residuals}
#' \item{`corrmat`}{Correlation matrix of the residuals}
#' \item{`roots`}{Roots of characteristic polynomials}
#' \item{`is_stable`}{Whether the process is stable or not based on `roots`}
#' \item{`log_lik`}{log-likelihood}
#' \item{`ic`}{Information criteria vector}
#' \itemize{
#'     \item{`AIC`} - AIC
#'     \item{`BIC`} - BIC
#'     \item{`HQ`} - HQ
#'     \item{`FPE`} - FPE
#' }
#' 
#' @references 
#' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
#' 
#' @importFrom stats cor pt
#' @importFrom tibble add_column
#' @importFrom dplyr mutate
#' @order 1
#' @export
summary.varlse <- function(object, ...) {
  var_name <- colnames(object$y0)
  cov_resid <- object$covmat
  coef_mat <- object$coefficients
  # inference-----------------------------
  var_stat <- infer_var(object)
  var_coef <- var_stat$summary_stat
  colnames(var_coef) <- c("estimate", "std.error", "statistic")
  term_name <- lapply(
    var_name,
    function(x) paste(rownames(coef_mat), x, sep = ".")
  ) |> 
    unlist()
  var_coef <- 
    var_coef |> 
    as.data.frame() |> 
    add_column(
      term = term_name,
      .before = 1
    ) |> 
    mutate(p.value = 2 * (1 - pt(abs(statistic), df = var_stat$df)))
  log_lik <- logLik(object)
  res <- list(
    names = var_name,
    totobs = object$totobs,
    obs = object$obs,
    p = object$p,
    coefficients = var_coef,
    df = var_stat$df,
    call = object$call,
    process = object$process,
    type = object$type,
    covmat = cov_resid,
    corrmat = cor(object$residuals),
    roots = stableroot(object),
    is_stable = is.stable(object),
    log_lik = log_lik,
    ic = c(
      AIC = AIC(log_lik),
      BIC = BIC(log_lik),
      HQ = HQ(log_lik)
    )
  )
  class(res) <- "summary.varlse"
  res
}
