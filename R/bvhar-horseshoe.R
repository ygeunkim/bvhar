#' Fitting Bayesian VHAR of Horseshoe Prior
#' 
#' This function fits VHAR with horseshoe prior.
#' 
#' @param y Time series data of which columns indicate the variables
#' @param har Numeric vector for weekly and monthly order. By default, `c(5, 22)`.
#' @param num_iter MCMC iteration number
#' @param num_burn Number of burn-in (warm-up). Half of the iteration is the default choice.
#' @param thinning Thinning every thinning-th iteration
#' @param bayes_spec Horseshoe initialization specification by [set_horseshoe()].
#' @param include_mean Add constant term (Default: `TRUE`) or not (`FALSE`)
#' @param sparsity Type of handling sparsity (Default, rowwise `"row"`) or (vectorized `"vec"`).
#' @param verbose Print the progress bar in the console. By default, `FALSE`.
#' @return `bvhar_horseshoe` returns an object named `bvarhs` [class].
#' It is a list with the following components:
#' 
#' \describe{
#'   \item{phi_record}{MCMC trace for vectorized coefficients (alpha \eqn{\phi}) with [posterior::draws_df] format.}
#'   \item{lambda_record}{MCMC trace for local shrinkage level (lambda \eqn{\lambda}) with [posterior::draws_df] format.}
#'   \item{tau_record}{MCMC trace for global shrinkage level (tau \eqn{\tau}) with [posterior::draws_df] format.}
#'   \item{psi_record}{MCMC trace for precision matrix (psi \eqn{\Psi}) with [list] format.}
#'   \item{chain}{The numer of chains}
#'   \item{coefficients}{Posterior mean of VHAR coefficients.}
#'   \item{psi_posterior}{Posterior mean of precision matrix \eqn{\Psi}}
#'   \item{covmat}{Posterior mean of covariance matrix}
#'   \item{omega_record}{MCMC trace for diagonal element of \eqn{\Psi} (omega) with [posterior::draws_df] format.}
#'   \item{eta_record}{MCMC trace for upper triangular element of \eqn{\Psi} (eta) with [posterior::draws_df] format.}
#'   \item{param}{[posterior::draws_df] with every variable: alpha, lambda, tau, omega, and eta}
#'   \item{df}{Numer of Coefficients: `3m + 1` or `3m`}
#'   \item{p}{3 (The number of terms. It contains this element for usage in other functions.)}
#'   \item{m}{Dimension of the data}
#'   \item{obs}{Sample size used when training = `totobs` - `p`}
#'   \item{totobs}{Total number of the observation}
#'   \item{call}{Matched call}
#'   \item{process}{Description of the model, e.g. `"VHAR_Horseshoe"`}
#'   \item{type}{include constant term (`"const"`) or not (`"none"`)}
#'   \item{algo}{Usual Gibbs sampling (`"gibbs"`) or fast sampling (`"fast"`)}
#'   \item{spec}{Horseshoe specification defined by [set_horseshoe()]}
#'   \item{iter}{Total iterations}
#'   \item{burn}{Burn-in}
#'   \item{thin}{Thinning}
#'   \item{HARtrans}{VHAR linear transformation matrix}
#'   \item{y0}{\eqn{Y_0}}
#'   \item{design}{\eqn{X_0}}
#'   \item{y}{Raw input}
#' }
#' @references 
#' Bai, R., & Ghosh, M. (2018). High-dimensional multivariate posterior consistency under global–local shrinkage priors. Journal of Multivariate Analysis, 167, 157–170. doi:[10.1016/j.jmva.2018.04.010](https://doi.org/10.1016/j.jmva.2018.04.010)
#' 
#' Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). *The horseshoe estimator for sparse signals*. Biometrika, 97(2), 465–480. doi:[10.1093/biomet/asq017](https://doi.org/10.1093/biomet/asq017)
#' 
#' Makalic, E., & Schmidt, D. F. (2016). *A Simple Sampler for the Horseshoe Estimator*. IEEE Signal Processing Letters, 23(1), 179–182. doi:[10.1109/lsp.2015.2503725](https://doi.org/10.1109/LSP.2015.2503725)
#' @importFrom posterior as_draws_df bind_draws
#' @importFrom stats cov
#' @order 1
#' @export
bvhar_horseshoe <- function(y,
                            har = c(5, 22),
                            num_iter = 1000, 
                            num_burn = floor(num_iter / 2),
                            thinning = 1,
                            bayes_spec = set_horseshoe(),
                            include_mean = TRUE,
                            sparsity = c("row", "vec"),
                            verbose = FALSE) {
  if (!all(apply(y, 2, is.numeric))) {
    stop("Every column must be numeric class.")
  }
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }
  # model specification---------------
  if (!is.horseshoespec(bayes_spec)) {
    stop("Provide 'horseshoespec' for 'bayes_spec'.")
  }
  # MCMC iterations-------------------
  if (num_iter < 1) {
    stop("Iterate more than 1 times for MCMC.")
  }
  if (num_iter < num_burn) {
    stop("'num_iter' should be larger than 'num_burn'.")
  }
  if (thinning < 1) {
    stop("'thinning' should be non-negative.")
  }
  # Y0 = X1 Phi + Z-------------------
  Y0 <- build_y0(y, har[2], har[2] + 1)
  dim_data <- ncol(y)
  if (!is.null(colnames(y))) {
    name_var <- colnames(y)
  } else {
    name_var <- paste0("y", seq_len(dim_data))
  }
  colnames(Y0) <- name_var
  X0 <- build_design(y, har[2], include_mean)
  colnames(X0) <- concatenate_colnames(name_var, 1:har[2], include_mean)
  hartrans_mat <- scale_har(dim_data, har[1], har[2], include_mean)
  name_har <- concatenate_colnames(name_var, c("day", "week", "month"), include_mean)
  X1 <- X0 %*% t(hartrans_mat)
  colnames(X1) <- name_har
  # Initial vectors-------------------
  dim_har <- ncol(X1)
  num_restrict <- switch (sparsity,
    "row" = ifelse(include_mean, dim_data * 3 + 1, dim_data * 3),
    "vec" = ifelse(include_mean, dim_data^2 * 3 + 1, dim_data^2 * 3)
  )
  if (length(bayes_spec$local_sparsity) != dim_har) {
    if (length(bayes_spec$local_sparsity) == 1) {
      bayes_spec$local_sparsity <- rep(bayes_spec$local_sparsity, num_restrict)
    } else {
      stop("Length of the vector 'local_sparsity' should be dim * 3 or dim * 3 + 1.")
    }
  }
  init_local <- bayes_spec$local_sparsity
  init_global <- bayes_spec$global_sparsity
  # MCMC-----------------------------
  res <- switch (sparsity,
    "row" = {
      estimate_bvar_horseshoe(
        num_iter = num_iter,
        num_burn = num_burn,
        x = X1,
        y = Y0,
        init_local = init_local,
        init_global = init_global,
        init_priorvar = cov(y),
        blocked_gibbs = 2,
        display_progress = verbose
      )
    },
    "vec" = {
      estimate_sur_horseshoe(
        num_iter = num_iter,
        num_burn = num_burn,
        x = X1,
        y = Y0,
        init_local = init_local,
        init_global = init_global,
        display_progress = verbose
      )
    }
  )
  # res <- estimate_bvar_horseshoe(
  #   num_iter = num_iter,
  #   num_burn = num_burn,
  #   x = X1,
  #   y = Y0,
  #   init_local = init_local,
  #   init_global = init_global,
  #   chain = 1,
  #   display_progress = verbose
  # )
  # preprocess the results-----------
  names(res) <- gsub(pattern = "^alpha", replacement = "phi", x = names(res))
  thin_id <- seq(from = 1, to = num_iter - num_burn, by = thinning)
  res$phi_record <- res$phi_record[thin_id,]
  colnames(res$phi_record) <- paste0("phi[", seq_len(ncol(res$phi_record)), "]")
  res$coefficients <- 
    colMeans(res$phi_record) %>% 
    matrix(ncol = dim_data)
  colnames(res$coefficients) <- name_var
  rownames(res$coefficients) <- name_har
  res$phi_record <- as_draws_df(res$phi_record)
  res$tau_record <- as.matrix(res$tau_record[thin_id])
  colnames(res$tau_record) <- "tau"
  res$tau_record <- as_draws_df(res$tau_record)
  if (sparsity == "row") {
    res$lambda_record <- res$lambda_record[thin_id,]
    colnames(res$lambda_record) <- paste0("lambda[", seq_len(ncol(res$lambda_record)), "]")
    res$lambda_record <- as_draws_df(res$lambda_record)
    res$psi_record <- split_psirecord(res$psi_record, varname = "psi")
    res$psi_record <- res$psi_record[thin_id]
    res$psi_posterior <- Reduce("+", res$psi_record) / length(res$psi_record)
    colnames(res$psi_posterior) <- name_var
    rownames(res$psi_posterior) <- name_var
    res$covmat <- solve(res$psi_posterior)
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
    # Parameters-----------------
    res$param <- bind_draws(
      res$phi_record,
      res$lambda_record,
      res$tau_record,
      res$omega_record,
      res$eta_record
    )
  } else {
    res$lambda_record <- as.matrix(res$lambda_record[thin_id])
    colnames(res$lambda_record) <- "lambda"
    res$lambda_record <- as_draws_df(res$lambda_record)
    res$covmat <- mean(res$sigma) * diag(dim_data)
    res$psi_posterior <- diag(dim_data) / mean(res$sigma)
    colnames(res$covmat) <- name_var
    rownames(res$covmat) <- name_var
    colnames(res$psi_posterior) <- name_var
    rownames(res$psi_posterior) <- name_var
    res$sigma_record <- as.matrix(res$sigma_record[thin_id])
    colnames(res$sigma_record) <- "sigma"
    res$sigma_record <- as_draws_df(res$sigma_record)
    # Parameters-----------------
    res$param <- bind_draws(
      res$phi_record,
      res$lambda_record,
      res$tau_record,
      res$sigma_record
    )
  }
  # variables------------
  res$df <- ncol(X0)
  res$p <- 3
  res$week <- har[1]
  res$month <- har[2]
  res$m <- ncol(y)
  res$obs <- nrow(Y0)
  res$totobs <- nrow(y)
  # model-----------------
  res$call <- match.call()
  res$process <- paste("VHAR", bayes_spec$prior, sep = "_")
  res$type <- ifelse(include_mean, "const", "none")
  # res$algo <- ifelse(fast_sampling, "fast", "gibbs")
  res$spec <- bayes_spec
  res$iter <- num_iter
  res$burn <- num_burn
  res$thin <- thinning
  # data------------------
  res$HARtrans <- hartrans_mat
  res$y0 <- Y0
  res$design <- X0
  res$y <- y
  # return S3 object-----------------
  class(res) <- c("bvharhs", "hsmod", "bvharsp")
  res
}
