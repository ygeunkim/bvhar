#' @rdname vhar_lm
#' @param x \code{vharlse} object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.vharlse <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # split the matrix for the print: Phi(d), Phi(w), Phi(m)
  # phihat_mat <- switch(
  #   x$type,
  #   "const" = {
  #     split.data.frame(x$coefficients[-(3 * x$m + 1),], gl(3, x$m)) |> 
  #       lapply(t)
  #   },
  #   "none" = {
  #     split.data.frame(x$coefficients, gl(3, x$m)) |> 
  #       lapply(t)
  #   }
  # )
  phihat_mat <- split_coef(x)
  names(phihat_mat) <- c("day", "week", "month")
  cat("VHAR Estimation")
  cat("====================================================\n\n")
  for (i in 1:x$p) {
    cat(paste0("LSE for ", names(phihat_mat)[i], ":\n"))
    # Phi(d), Phi(w), Phi(m)---------
    print.default(
      phihat_mat[[i]],
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n\n")
  }
  # const term----------------------
  if (x$type == "const") {
    intercept <- x$coefficients[x$df,]
    cat("LSE for constant:\n")
    print.default(
      intercept,
      digits = digits,
      print.gap = 2L,
      quote = FALSE
    )
    cat("\n\n")
  }
  cat("--------------------------------------------------\n")
  cat("*_day, *_week, *_month of the Coefficient matrix: daily, weekly, and monthly term in the VHAR model\n\n")
  invisible(x)
}

#' @rdname vhar_lm
#' @exportS3Method knitr::knit_print
knit_print.vharlse <- function(x, ...) {
  print(x)
}

#' @rdname summary.vharlse
#' @param x `summary.vharlse` object
#' @param digits digit option to print
#' @param signif_code Check significant rows (Default: `TRUE`)
#' @param ... not used
#' @importFrom tidyr separate
#' @importFrom dplyr case_when
#' @order 2
#' @export
print.summary.vharlse <- function(x, digits = max(3L, getOption("digits") - 3L), signif_code = TRUE, ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  # for VAR(p)--------------------------------------------------------------------
  cat("VHAR Estimation\n")
  cat("====================================================\n")
  # variables--------------------------
  cat("Variables: ")
  cat(paste(x$names, collapse = ", "))
  if (x$type == "const") {
    cat("\nwith const added\n")
  }
  cat(
    paste("Number of variables:", "m =", length(x$names))
  )
  cat("\n")
  # obs num----------------------------
  cat(
    paste("Observation size:", "T =", x$totobs)
  )
  cat("\n")
  cat(
    paste("Number of sample used for fitting:", "n = T - p =", x$obs)
  )
  # coefficients-----------------------
  coef_mat <- x$coefficients
  dim_data <- ncol(x$covmat)
  dim_design <- nrow(coef_mat) / dim_data
  coef_mat <- 
    coef_mat |> 
    separate(term, into = c("term", "variable"), sep = "\\.") |> 
    split.data.frame(f = gl(dim_data, dim_design))
  if (signif_code) {
    sig_star <- numeric(dim_design)
    p_val <- numeric(dim_design)
  }
  cat("\n====================================================\n")
  for (i in 1:length(coef_mat)) {
    cat(paste0(unique(coef_mat[[i]]$variable), " variable", ":\n"))
    if (signif_code) {
      p_val <- coef_mat[[i]][, "p.value"]
      sig_star <- case_when(
        p_val <= .001 ~ "***",
        p_val > .001 & p_val <= .01 ~ "**",
        p_val > .01 & p_val <= .05 ~ "*",
        p_val > .05 & p_val <= .1 ~ ".",
        p_val > .1 ~ " "
      )
      coef_mat[[i]][, " "] <- sig_star
    }
    # print Phi-----------------------
    print(
      coef_mat[[i]][,-2], # without variable name column since it is printed in the header
      digits = digits,
      print.gap = 2L,
      quote = FALSE,
      right = FALSE,
      row.names = FALSE
    )
    cat("\n")
  }
  if (signif_code) {
    cat(paste0("---\n", "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1", "\n"))
  }
  cat("\n")
  # df---------------------------------------
  cat("------------------------------------------------------\n")
  cat(paste("Degrees of freedom:", "df =", x$df))
  cat("\n\n")
  # cov and corr-----------------------------
  cat("LS Estimate for Covariance matrix:\n")
  print(x$covmat)
  cat("\nLS Estimate for Correlation matrix:\n")
  print(x$corrmat)
  # information criteria--------------------
  cat("\n====================================================\n")
  cat("log-likelihood:\n")
  print(x$log_lik)
  cat("Information criteria:\n")
  print(x$ic)
}

#' @rdname summary.vharlse
#' @exportS3Method knitr::knit_print
knit_print.summary.vharlse <- function(x, ...) {
  print(x)
}
