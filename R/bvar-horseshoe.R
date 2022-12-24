#' Fitting Bayesian VAR(p) of Horseshoe Prior
#' 
#' This function fits BVAR(p) with horseshoe prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param p VAR lag
#' @param num_iter MCMC iteration number
#' @param num_warm Number of warm-up (burn-in). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param init_spec Horseshoe initialization specification by [init_horseshoe()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @return `bvar_horseshoe` returns an object named [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{coefficients}{Coefficient Matrix}
#'   \item{p}{Lag of VAR}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' @references 
#' Bhattacharya, A., Chakraborty, A., & Mallick, B. K. (2016). *Fast sampling with Gaussian scale mixture priors in high-dimensional regression*. Biometrika, 103(4), 985–991. doi:[10.1093/biomet/asw042](https://doi.org/10.1093/biomet/asw042)
#' 
#' Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). *The horseshoe estimator for sparse signals*. Biometrika, 97(2), 465–480. doi:[10.1093/biomet/asq017](https://doi.org/10.1093/biomet/asq017)
#' 
#' Makalic, E., & Schmidt, D. F. (2016). *A Simple Sampler for the Horseshoe Estimator*. IEEE Signal Processing Letters, 23(1), 179–182. doi:[10.1109/lsp.2015.2503725](https://doi.org/10.1109/LSP.2015.2503725)
#' @importFrom posterior as_draws_df bind_draws
#' @order 1
#' @export
bvar_horseshoe <- function(y,
                           p,
                           num_iter = 1000, 
                           num_warm = floor(num_iter / 2),
                           thinning = 1,
                           init_spec = init_horseshoe(),
                           include_mean = TRUE,
                           verbose = FALSE) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  # model specification---------------
  if (!is.horseshoeinit(init_spec)) {
    stop("Provide 'horseshoeinit' for 'init_spec'.")
  }
  # MCMC iterations-------------------
  if (num_iter < 1) {
    stop("Iterate more than 1 times for MCMC.")
  }
  if (num_iter < num_warm) {
    stop("'num_iter' should be larger than 'num_warm'.")
  }
  if (thinning < 1) {
    stop("'thinning' should be non-negative.")
  }
  # Y0 = X0 A + Z---------------------
  Y0 <- build_y0(y, p, p + 1)
  dim_data <- ncol(y)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
  }
  colnames(Y0) <- name_var
  X0 <- build_design(y, p, include_mean)
  name_lag <- concatenate_colnames(name_var, 1:p, include_mean)
  colnames(X0) <- name_lag
  # Initial vectors-------------------
  dim_design <- ncol(X0)
  if (init_spec$chain == 1) {
    if (length(init_spec$init_local) != dim_design) {
      stop("Length of the vector 'init_local' should be dim * p or dim * p + 1.")
    }
    if (ncol(init_spec$init_priorvar) != dim_data) {
      stop("Dimension of the matrix 'init_priorvar' should be dim x dim.")
    }
    init_local <- init_spec$init_local
    init_global <- init_spec$init_global
    init_priorvar <- init_spec$init_priorvar
  } else {
    if (is.matrix(init_spec$init_local[[1]]) &&
        !(nrow(init_spec$init_local[[1]]) == dim_design || ncol(init_spec$init_local[[1]]) == dim_data)) {
      stop("Dimension of the matrix 'init_local' should be (dim * p) x dim or (dim * p + 1) x dim.")
    }
    init_local <- unlist(init_spec$init_local)
    init_global <- init_spec$init_global
    init_priorvar <- init_spec$init_priorvar
  }
  # MCMC-----------------------------
  res <- estimate_horseshoe_niw(
    num_iter = num_iter,
    num_warm = num_warm,
    x = X0,
    y = Y0,
    init_local = init_local,
    init_global = init_global,
    init_priorvar = init_priorvar,
    chain = init_spec$chain,
    display_progress = verbose
  )
  # preprocess the results-----------
  thin_id <- seq(from = 1, to = num_iter - num_warm, by = thinning)
  if (res$chain > 1) {
    # 
  } else {
    res$alpha_record <- res$alpha_record[thin_id,]
    colnames(res$alpha_record) <- paste0("alpha[", seq_len(ncol(res$alpha_record)), "]")
    res$alpha_posterior <- 
      colMeans(res$alpha_record) %>% 
      matrix(ncol = dim_data)
    colnames(res$alpha_posterior) <- name_var
    rownames(res$alpha_posterior) <- name_lag
    res$alpha_record <- as_draws_df(res$alpha_record)
    res$lambda_record <- res$lambda_record[thin_id,]
    colnames(res$lambda_record) <- paste0("lambda[", seq_len(ncol(res$lambda_record)), "]")
    res$lambda_record <- as_draws_df(res$lambda_record)
    res$tau_record <- as.matrix(res$tau_record[thin_id])
    colnames(res$tau_record) <- "tau"
    res$tau_record <- as_draws_df(res$tau_record)
    # res$nu_record <- as.matrix(res$nu_record[thin_id])
    # colnames(res$nu_record) <- "nu"
    # res$nu_record <- as_draws_df(res$nu_record)
    # res$xi_record <- as.matrix(res$xi_record[thin_id])
    # colnames(res$xi_record) <- "xi"
    # res$xi_record <- as_draws_df(res$xi_record)
    res$psi_record <- split_psirecord(res$psi_record, varname = "psi")
    res$psi_record <- res$psi_record[thin_id]
    res$psi_posterior <- Reduce("+", res$psi_record) / length(res$psi_record)
    colnames(res$psi_posterior) <- name_var
    rownames(res$psi_posterior) <- name_var
    # diagonal of precision
    res$omega_record <- 
      lapply(res$psi_record, diag) %>% 
      do.call(rbind, .)
    colnames(res$omega_record) <- paste0("omega[", seq_len(ncol(res$omega_record)), "]")
    res$omega_record <- as_draws_df(res$omega_record)
    # upper diagonal of precision
    res$eta_record <-
      lapply(res$psi_record, function(x) x[upper.tri(x, diag = FALSE)]) %>%
      do.call(rbind, .)
    colnames(res$eta_record) <- paste0("eta[", seq_len(ncol(res$eta_record)), "]")
    res$eta_record <- as_draws_df(res$eta_record)
  }
  res$param <- bind_draws(
    res$alpha_record,
    res$lambda_record,
    res$tau_record,
    res$omega_record,
    res$eta_record,
    res$nu_record,
    res$xi_record
  )
  # variables------------
  res$df <- ncol(X0)
  res$p <- p
  res$m <- ncol(y)
  res$obs <- nrow(Y0)
  res$totobs <- nrow(y)
  # model-----------------
  res$call <- match.call()
  res$process <- paste(init_spec$process, init_spec$prior, sep = "_")
  res$type <- ifelse(include_mean, "const", "none")
  res$spec <- init_spec
  res$iter <- num_iter
  res$burn <- num_warm
  res$thin <- thinning
  # data------------------
  res$y0 <- Y0
  res$design <- X0
  res$y <- y
  # return S3 object-----------------
  class(res) <- c("bvarhs", "bvharmod")
  res
}

#' @rdname bvar_horseshoe
#' @param x `bvarhs` object
#' @param digits digit option to print
#' @param ... not used
#' @order 2
#' @export
print.bvarhs <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(
    "Call:\n",
    paste(deparse(x$call), sep="\n", collapse = "\n"), "\n\n", sep = ""
  )
  cat(sprintf("BVAR(%i) with Horseshoe Prior\n", x$p))
  cat("Fitted by Metropolis algorithm\n")
  cat(paste0("Total number of iteration: ", x$iter, "\n"))
  cat(paste0("Number of warm-up: ", x$burn, "\n"))
  if (x$thin > 1) {
    cat(paste0("Thinning: ", x$thin, "\n"))
  }
  cat("====================================================\n\n")
  print(
    x$param,
    digits = digits,
    print.gap = 2L,
    quote = FALSE
  )
}

#' @rdname bvar_horseshoe
#' @param x `bvarhs` object
#' @param ... not used
#' @order 3
#' @export
knit_print.bvarhs <- function(x, ...) {
  print(x)
}
