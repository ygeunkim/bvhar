#' Summarizing Vector HAR Model
#' 
#' `summary` method for `vharlse` class.
#' 
#' @param object A `vharlse` object
#' @param ... not used
#' 
#' @return `summary.vharlse` [class] additionally computes the following
#' \item{`names`}{Variable names}
#' \item{`totobs`}{Total number of the observation}
#' \item{`obs`}{Sample size used when training = `totobs` - `p`}
#' \item{`p`}{3}
#' \item{`week`}{Order for weekly term}
#' \item{`month`}{Order for monthly term}
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
#' @references 
#' Lütkepohl, H. (2007). *New Introduction to Multiple Time Series Analysis*. Springer Publishing.
#' 
#' Corsi, F. (2008). *A Simple Approximate Long-Memory Model of Realized Volatility*. Journal of Financial Econometrics, 7(2), 174-196.
#' 
#' Baek, C. and Park, M. (2021). *Sparse vector heterogeneous autoregressive modeling for realized volatility*. J. Korean Stat. Soc. 50, 495-510.
#' 
#' @importFrom stats cor pt
#' @importFrom tibble add_column
#' @importFrom dplyr mutate
#' @order 1
#' @export
summary.vharlse <- function(object, ...) {
  vhar_name <- colnames(object$y0)
  cov_resid <- object$covmat
  coef_mat <- object$coefficients
  # inference-------------------------------
  vhar_stat <- infer_vhar(object)
  vhar_coef <- vhar_stat$summary_stat
  colnames(vhar_coef) <- c("estimate", "std.error", "statistic")
  term_name <- lapply(
    vhar_name,
    function(x) paste(rownames(coef_mat), x, sep = ".")
  ) |> 
    unlist()
  vhar_coef <- 
    vhar_coef |> 
    as.data.frame() |> 
    add_column(
      term = term_name,
      .before = 1
    ) |> 
    mutate(p.value = 2 * (1 - pt(abs(statistic), df = vhar_stat$df)))
  log_lik <- logLik(object)
  res <- list(
    names = vhar_name,
    totobs = object$totobs,
    obs = object$obs,
    p = object$p,
    week = object$week,
    month = object$month,
    # coefficients = phihat_mat,
    coefficients = vhar_coef,
    call = object$call,
    process = object$process,
    type = object$type,
    covmat = cov_resid,
    corrmat = cor(object$residuals),
    log_lik = log_lik,
    ic = c(
      AIC = AIC(log_lik),
      BIC = BIC(log_lik),
      HQ = HQ(log_lik)
    )
  )
  class(res) <- "summary.vharlse"
  res
}
